pub struct FrequencyMerge<B> {
    backend: B,
}

impl<B> FrequencyMerge<B> {
    /// Convenience function for the frequency-based merging approach.
    ///
    /// Perform the merging 4 times with a slight displacement between the frame to supress artifacts in the merging process.
    /// The shift is equal to to the tile size used in the merging process, which later translates into tile_size_merge / 2
    /// when each color channel is processed independently.
    ///
    /// Currently only supports Bayer raw files
    fn align_merge_frequency_domain(
        ref_idx: usize,
        mosaic_pattern_width: u32,
        search_distance: u32,
        tile_size: u32,
        noise_reduction: f64,
        uniform_exposure: bool,
        exposure_bias: &[i32],
        white_level: i32,
        black_level: &[&[i32]],
        color_factors: &[&[Double]],
        images: &[Image<B>],
        hotpixel_weight_image: &Image<B>,
        final_texture: &mut Image<B>,
    ) -> Result<(), Error> {
        log::info!("Merging in the frequency domain...");

        // The tile size for merging in frequency domain is set to 8x8 for all tile sizes used for alignment.
        // The smaller tile size leads to a reduction of artifacts at specular highlights at the expense of a
        // slightly reduced suppression of low-frequency noise in the shadows. The fixed value of 8 is supported
        // by the highly-optimized fast Fourier transform. A slow, but easier to understand discrete Fourier
        // transform is also provided for values larger than 8.
        //
        // see https://graphics.stanford.edu/papers/hdrp/hasinoff-hdrplus-sigasia16.pdf for more details
        let tile_size_merge = 8;

        // These corrections account for the fact that bursts with exposure bracketing include images with longer
        // exposure times, which exhibit a better signal-to-noise ratio. Thus the expected noise level n used in
        // the merging equation d^2/(d^2 + n) has to be reduced to get a comparable noise reduction strength on
        // average (exposure_corr1). Furthermore, a second correction (exposure_corr2) takes into account that
        // images with longer exposure get slightly larger weights than images with shorter exposure (applied as
        // an increased value for the parameter max_motion_norm => see call of function merge_frequency_domain).
        // The simplified formulas do not correctly include read noise (a square is ignored) and as a consequence,
        // merging weights in the shadows will be very slightly overestimated. As the shadows are typically lifted
        // heavily in HDR shots, this effect may be even preferable.
        let mut exposure_corr1 = 0.0;
        let mut exposure_corr2 = 0.0;

        for comp_idx in 0..exposure_bias.len() {
            let exposure_factor = f64::pow(
                2.0,
                (f64(exposure_bias[comp_idx] - exposure_bias[ref_idx]) / 100.0),
            );

            exposure_corr1 += (0.5 + 0.5 / exposure_factor);
            exposure_corr2 += f64::min(4.0, exposure_factor);
        }

        exposure_corr1 /= f64(exposure_bias.len());
        exposure_corr2 /= f64(exposure_bias.len());

        // derive normalized robustness value: two steps in noise_reduction (-2.0 in this case) yield an increase by
        // a factor of two in the robustness norm with the idea that the variance of shot noise increases by a factor
        // of two per iso level
        let robustness_rev =
            0.5 * (if uniform_exposure { 26.5 } else { 28.5 } - f64(i32(noise_reduction + 0.5)));
        let robustness_norm =
            exposure_corr1 / exposure_corr2 * f64::pow(2.0, (-robustness_rev + 7.5));

        // derive estimate of read noise with the idea that read noise increases approx. by a factor of three (stronger
        // than increase in shot noise) per iso level to increase noise reduction in darker regions relative to bright regions
        let read_noise = f64::pow(f64::pow(2.0, (-robustness_rev + 10.0)), 1.6);

        // derive a maximum value for the motion norm with the idea that denoising can be stronger in static regions
        // with good alignment compared to regions with motion factors from Google paper: daylight = 1, night = 6,
        // darker night = 14, extreme low-light = 25. We use a continuous value derived from the robustness value
        // to cover a similar range as proposed in the paper
        //
        // see https://graphics.stanford.edu/papers/night-sight-sigasia19/night-sight-sigasia19.pdf for more details
        let max_motion_norm = f64::max(1.0, f64::pow(1.3, (11.0 - robustness_rev)));

        // set original texture size
        let texture_width_orig = images[ref_idx].width();
        let texture_height_orig = images[ref_idx].height();

        // set alignment params
        let min_image_dim = min(texture_width_orig, texture_height_orig);
        let mut downscale_factor_array = vec![mosaic_pattern_width];
        let mut search_dist_array = vec![2];
        let mut tile_size_array = vec![tile_size];
        let mut res = min_image_dim / downscale_factor_array[0];

        // This loop generates lists, which include information on different parameters required for the
        // alignment on different resolution levels. For each pyramid level, the downscale factor compared
        // to the neighboring resolution, the search distance, tile size, resolution (only for lowest level)
        // and total downscale factor compared to the original resolution (only for lowest level) are calculated.
        while res > search_distance {
            downscale_factor_array.push(2);
            search_dist_array.push(2);
            tile_size_array.push(max(tile_size_array.last().unwrap() / 2, 8));
            res /= 2;
        }

        // calculate padding for extension of the image frame with zeros
        // The minimum size of the frame for the frequency merging has to be texture size + tile_size_merge
        // as 4 frames shifted by tile_size_merge in x, y and x, y are processed in four consecutive runs.
        // For the alignment, the frame may be extended further by pad_align due to the following reason:
        // the alignment is performed on different resolution levels and alignment vectors are upscaled by
        // a simple multiplication by 2. As a consequence, the frame at all resolution levels has to be a
        // multiple of the tile sizes of these resolution levels.
        let tile_factor =
            tile_size_array.last().unwrap() * downscale_factor_array.into_iter().copied().product();

        let pad_align_x =
            f32::ceil((texture_width_orig + tile_size_merge) as f32 / tile_factor as f32) as i32;

        let pad_align_x =
            (pad_align_x * tile_factor as i32 - texture_width_orig - tile_size_merge) / 2;

        let pad_align_y =
            f32::ceil((texture_height_orig + tile_size_merge) as f32 / tile_factor as f32) as i32;

        let pad_align_y =
            (pad_align_y * tile_factor as i32 - texture_height_orig - tile_size_merge) / 2;

        // calculate padding for the merging in the frequency domain, which can be applied to the actual image frame + a smaller margin compared to the alignment
        let crop_merge_x = f32::floor(pad_align_x as f32 / (2 * tile_size_merge) as f32) as i32;
        let crop_merge_x = crop_merge_x * 2 * tile_size_merge;

        let crop_merge_y = f32::floor(pad_align_y as f32 / (2 * tile_size_merge) as f32) as i32;
        let crop_merge_y = crop_merge_y * 2 * tile_size_merge;

        let pad_merge_x = pad_align_x - crop_merge_x;
        let pad_merge_y = pad_align_y - crop_merge_y;

        // set tile information needed for the merging
        let tile_info_merge = TileInfo {
            tile_size,
            tile_size_merge,
            search_dist: 0,
            n_tiles_x: (texture_width_orig + tile_size_merge + 2 * pad_merge_x)
                / (2 * tile_size_merge),
            n_tiles_y: (texture_height_orig + tile_size_merge + 2 * pad_merge_y)
                / (2 * tile_size_merge),
            n_pos_1d: 0,
            n_pos_2d: 0,
        };

        for i in 1..=4 {
            // let t0 = DispatchTime.now().uptimeNanoseconds;

            // set shift values
            let shift_left = if i % 2 == 0 { tile_size_merge } else { 0 };
            let shift_right = if i % 2 == 1 { tile_size_merge } else { 0 };
            let shift_top = if i < 3 { tile_size_merge } else { 0 };
            let shift_bottom = if i >= 3 { tile_size_merge } else { 0 };

            // add shifts for artifact suppression
            let pad_left = pad_align_x + shift_left;
            let pad_right = pad_align_x + shift_right;
            let pad_top = pad_align_y + shift_top;
            let pad_bottom = pad_align_y + shift_bottom;

            // prepare reference texture by correcting hot pixels, equalizing exposure and extending the texture
            let ref_image = images[ref_idx].prepare(
                hotpixel_weight_image,
                pad_left,
                pad_right,
                pad_top,
                pad_bottom,
                0,
                black_level[ref_idx],
                mosaic_pattern_width,
            );
            // convert reference texture into RGBA pixel format that SIMD instructions can be applied
            let ref_image_rgba = ref_image.convert_to_rgba(crop_merge_x, crop_merge_y);

            let mut black_level_mean =
                black_level[ref_idx].sum() as f64 / black_level[ref_idx].len() as f64;

            // build reference pyramid
            let ref_pyramid = ref_image.build_pyramid(
                downscale_factor_array,
                black_level_mean,
                color_factors[ref_idx],
            );

            // estimate noise level of tiles
            let rms_image = self
                .backend
                .calculate_rms_rgba(ref_image_rgba, tile_info_merge);

            // generate texture to accumulate the total mismatch
            let total_mismatch_image = rms_image.clone_zeroed();

            // transform reference texture into the frequency domain
            let ref_texture_ft = self.backend.forward_ft(ref_image_rgba, tile_info_merge);

            // add reference texture to the final texture
            let final_texture_ft = ref_texture_ft.copy_image();

            // iterate over comparison images
            for comp_idx in 0..images.len() {
                if comp_idx == ref_idx {
                    continue;
                }

                // prepare comparison texture by correcting hot pixels, equalizing exposure and extending the texture
                let comp_image = images[comp_idx].prepare(
                    hotpixel_weight_image,
                    pad_left,
                    pad_right,
                    pad_top,
                    pad_bottom,
                    (exposure_bias[ref_idx] - exposure_bias[comp_idx]),
                    black_level[comp_idx],
                    mosaic_pattern_width,
                );

                let black_level_mean =
                    black_level[comp_idx].sum() as f64 / black_level[comp_idx].len() as f64;

                // align comparison texture
                let aligned_image_rgba = self
                    .aligner
                    .align(
                        ref_pyramid,
                        comp_image,
                        downscale_factor_array,
                        tile_size_array,
                        search_dist_array,
                        (exposure_bias[comp_idx] == exposure_bias[ref_idx]),
                        black_level_mean,
                        color_factors[comp_idx],
                    )
                    .convert_to_rgba(crop_merge_x, crop_merge_y);

                // calculate exposure factor between reference texture and aligned texture
                let exposure_factor = f64::pow(
                    2.0,
                    ((exposure_bias[comp_idx] - exposure_bias[ref_idx]) as f64 / 100.0),
                );

                // calculate mismatch texture
                let mismatch_image = self.backend.calculate_mismatch_rgba(
                    aligned_image_rgba,
                    ref_image_rgba,
                    rms_image,
                    exposure_factor,
                    tile_info_merge,
                );

                // normalize mismatch texture
                let mean_mismatch = mismatch_image
                    .crop(
                        shift_left / tile_size_merge,
                        shift_right / tile_size_merge,
                        shift_top / tile_size_merge,
                        shift_bottom / tile_size_merge,
                    )
                    .global_mean(mosaic_pattern_width);

                self.backend
                    .normalize_mismatch(mismatch_image, mean_mismatch);

                // add mismatch texture to the total, accumulated mismatch texture
                total_mismatch_image.add(mismatch_image, images.len());

                let highlights_norm_texture = self.calculate_highlights_norm_rgba(
                    aligned_image_rgba,
                    exposure_factor,
                    tile_info_merge,
                    if white_level == -1 {
                        1000000
                    } else {
                        white_level
                    },
                    black_level_mean,
                );

                // transform aligned comparison texture into the frequency domain
                let aligned_texture_ft = self.forward_ft(aligned_image_rgba, tile_info_merge);

                // adapt max motion norm for images with bracketed exposure
                let max_motion_norm_exposure = if uniform_exposure {
                    max_motion_norm
                } else {
                    f64::min(4.0, exposure_factor) * max_motion_norm.sqrt()
                };

                // merge aligned comparison texture with reference texture in the frequency domain
                self.backend.merge_frequency_domain(
                    ref_texture_ft,
                    aligned_texture_ft,
                    final_texture_ft,
                    rms_image,
                    mismatch_image,
                    highlights_norm_texture,
                    robustness_norm,
                    read_noise,
                    max_motion_norm_exposure,
                    uniform_exposure,
                    tile_info_merge,
                );

                // sync GUI progress
                // DispatchQueue.main.async { progress.int += Int(80000000/Double(4*(textures.count-1))) }
            }
            // apply simple deconvolution to slightly correct potential blurring from misalignment of bursts
            self.deconvolute_frequency_domain(
                final_texture_ft,
                total_mismatch_image,
                tile_info_merge,
            );

            // transform output texture back to image domain
            let mut output_texture =
                self.backward_ft(final_texture_ft, tile_info_merge, images.len());

            // reduce potential artifacts at tile borders
            self.reduce_artifacts_tile_border(
                output_texture,
                ref_image_rgba,
                tile_info_merge,
                black_level[ref_idx],
            );

            // convert back to the 2x2 pixel structure and crop to original size
            let output_texture = output_texture.convert_to_bayer().crop(
                pad_left - crop_merge_x,
                pad_right - crop_merge_x,
                pad_top - crop_merge_y,
                pad_bottom - crop_merge_y,
            );

            // add output texture to the final texture to collect all textures of the four iterations
            final_texture.add_texture(output_texture, 1);

            // log::info!("Align+merge (\(i)/4): ", Float(DispatchTime.now().uptimeNanoseconds - t0) / 1_000_000_000)
        }

        Ok(())
    }
}
