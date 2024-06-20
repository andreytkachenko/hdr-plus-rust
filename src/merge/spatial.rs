pub struct MergeSpatial<B: MergeBackend> {
    backend: B,
}

impl<B: MergeBackend> MergeSpatial<B> {
    /// Convenience function for the spatial merging approach
    ///
    /// Supports non-Bayer raw files
    fn align_merge_spatial_domain(
        ref_idx: Int,
        mosaic_pattern_width: Int,
        search_distance: Int,
        tile_size: Int,
        noise_reduction: f64,
        uniform_exposure: bool,
        exposure_bias: &[i32],
        black_level: &[&[i32]],
        color_factors: &[&[f64]],
        images: &[Image],
        hotpixel_weight_image: &Image,
        final_image: &mut Image,
    ) -> Result {
        log::info!("Merging in the spatial domain...");

        let kernel_size = 16; // kernel size of binomial filtering used for blurring the image

        // derive normalized robustness value: four steps in noise_reduction (-4.0 in this case) yield an increase
        // by a factor of two in the robustness norm with the idea that the sd of shot noise increases by a factor
        // of sqrt(2) per iso level
        let robustness_rev = 0.5 * (36.0 - f64(i32(noise_reduction + 0.5)));
        let robustness = 0.12 * f32::pow(1.3, robustness_rev) - 0.4529822;

        // set original texture size
        let image_width_orig = images[ref_idx].width;
        let image_height_orig = images[ref_idx].height;

        // set alignment params
        let min_image_dim = i32::min(image_width_orig, image_height_orig);
        let mut downscale_factor_array = vec![mosaic_pattern_width];
        let mut search_dist_array = vec![2];
        let mut tile_size_array = vec![tile_size];
        let mut res = min_image_dim / downscale_factor_array[0];

        // This loop generates lists, which include information on different parameters required for the alignment
        // on different resolution levels. For each pyramid level, the downscale factor compared to the neighboring
        // resolution, the search distance, tile size, resolution (only for lowest level) and total downscale factor
        // compared to the original resolution (only for lowest level) are calculated.
        while res > search_distance {
            downscale_factor_array.push(2);
            search_dist_array.push(2);
            tile_size_array.push(max(tile_size_array.last().unwrap() / 2, 8));
            res /= 2;
        }

        // calculate padding for extension of the image frame with zeros
        // For the alignment, the frame may be extended further by pad_align due to the following reason: the alignment
        // is performed on different resolution levels and alignment vectors are upscaled by a simple multiplication by 2.
        // As a consequence, the frame at all resolution levels has to be a multiple of the tile sizes of these resolution levels.
        let tile_factor =
            *tile_size_array.last().unwrap() * downscale_factor_array.into_iter().fold(1, i32::mul);

        let mut pad_align_x = f32::ceil(image_width_orig as f32 / tile_factor as f32) as i32;
        pad_align_x = (pad_align_x * tile_factor as i32 - image_width_orig) / 2;

        let mut pad_align_y = f32::ceil(image_height_orig as f32 / tile_factor as f32) as i32;
        pad_align_y = (pad_align_y * tile_factor as i32 - image_height_orig) / 2;

        // prepare reference texture by correcting hot pixels, equalizing exposure and extending the texture
        let mut ref_image = images[ref_idx].prepare(
            hotpixel_weight_image,
            pad_align_x,
            pad_align_x,
            pad_align_y,
            pad_align_y,
            0,
            black_level[ref_idx],
            mosaic_pattern_width,
        );

        let ref_image_cropped = ref_image.crop(pad_align_x, pad_align_x, pad_align_y, pad_align_y);

        let black_level_mean = f64(black_level[ref_idx].iter().copied().fold(0, i32::add))
            / f64(black_level[ref_idx].len());

        // build reference pyramid
        let ref_pyramid = ref_image.build_pyramid(
            downscale_factor_array,
            black_level_mean,
            color_factors[ref_idx],
        );

        // blur reference texure and estimate noise standard deviation
        // -  the computation is done here to avoid repeating the same computation in 'robust_merge()'
        let ref_texture_blurred = ref_image_cropped.blur(mosaic_pattern_width, kernel_size);

        let noise_sd =
            ref_image_cropped.estimate_color_noise(ref_texture_blurred, mosaic_pattern_width);

        // iterate over comparison images
        for comp_idx in 0..images.len() {
            // add the reference texture to the output
            if comp_idx == ref_idx {
                final_image.add(ref_image_cropped, images.len());

                // DispatchQueue.main.async { progress.int += Int(80000000/Double(textures.count)) }
                continue;
            }

            // prepare comparison texture by correcting hot pixels, equalizing exposure and extending the texture
            let comp_texture = images[comp_idx].prepare(
                hotpixel_weight_image,
                pad_align_x,
                pad_align_x,
                pad_align_y,
                pad_align_y,
                (exposure_bias[ref_idx] - exposure_bias[comp_idx]),
                black_level[comp_idx],
                mosaic_pattern_width,
            );

            black_level_mean = f64(black_level[comp_idx].sum()) / f64(black_level[comp_idx].len());

            // align comparison texture
            let mut aligned_texture = align_texture(
                ref_pyramid,
                comp_texture,
                downscale_factor_array,
                tile_size_array,
                search_dist_array,
                (exposure_bias[comp_idx] == exposure_bias[ref_idx]),
                black_level_mean,
                color_factors[comp_idx],
            );

            aligned.crop_mut(pad_align_x, pad_align_x, pad_align_y, pad_align_y);

            // robust-merge the texture
            let merged_texture = self.robust_merge(
                ref_image_cropped,
                ref_texture_blurred,
                aligned_texture,
                kernel_size,
                robustness,
                noise_sd,
                mosaic_pattern_width,
            );

            // add robust-merged texture to the output image
            final_image.add(merged_texture, images.len());

            // sync GUI progress
            // DispatchQueue.main.async { progress.int += Int(80000000/Double(textures.count)) }
        }

        Ok(())
    }

    fn estimate_color_noise(
        texture: MTLTexture,
        texture_blurred: MTLTexture,
        mosaic_pattern_width: Int,
    ) -> f32 {
        texture
            // compute the color difference of each mosaic superpixel between the original and the blurred texture
            .color_difference(texture_blurred, mosaic_pattern_width)
            // compute the average of the difference between the original and the blurred texture
            .global_mean(mosaic_pattern_width)
    }

    fn robust_merge(
        ref_image: Image,
        ref_texture_blurred: Image,
        comp_texture: Image,
        kernel_size: i32,
        robustness: f64,
        noise_sd: MTLBuffer,
        mosaic_pattern_width: i32,
    ) -> Result<Image, Error> {
        // blur comparison texture
        let comp_texture_blurred = comp_texture.blur(mosaic_pattern_width, kernel_size)?;

        // compute the color difference of each superpixel between the blurred reference and the comparison textures
        let image_diff =
            ref_texture_blurred.color_difference(comp_texture_blurred, mosaic_pattern_width)?;

        // create a weight texture
        let weight_image = self
            .backend
            .create_image(image_diff.width, image_diff.height);

        // compute merge weight
        self.backend
            .compute_merge_weight(&image_diff, weight_image, noise_sd, robustness)?;

        // upsample merge weight to full image resolution
        let weight_image_upsampled =
            weight_image.upsample(ref_image.width, ref_image.height, Bilinear);

        // average the input textures based on the weight
        ref_image.add_weighted(comp_texture, weight_image_upsampled)
    }
}
