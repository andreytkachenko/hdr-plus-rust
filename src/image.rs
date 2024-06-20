pub struct Image<B: ImageBackend> {
    _b: B,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpsampleType {
    Bilinear,
    NearestNeighbour,
}

impl<B: ImageBackend> Image<B> {
    pub fn add(&mut self, other: &Image) {}

    /// Find hotpixels based on the idea that they will be the same pixels in all frames of the burst.
    fn find_hotpixels(
        &mut self,
        textures: &[Image],
        black_level: &[&[i32]],
        iso_exposure_time: &[f64],
        noise_reduction: f64,
        mosaic_pattern_width: u32,
    ) {
        if mosaic_pattern_width != 2 && mosaic_pattern_width != 6 {
            return;
        }

        // calculate hot pixel correction strength based on ISO value, exposure time and number of frames in the burst
        let mut correction_strength = if iso_exposure_time[0] > 0.0 {
            let correction_strength = iso_exposure_time.sum();

            (
                // TODO: This needs an explanation
                f64::min(
                    80,
                    f64::max(
                        5.0,
                        correction_strength / (textures.len() as f64).sqrt()
                            * if noise_reduction == 23.0 { 0.25 } else { 1.00 },
                    ),
                ) - 5.0
            ) / 75.0
        } else {
            1.0
        };

        // only apply hot pixel correction if correction strength is larger than 0.001
        if correction_strength > 0.001 {
            // generate simple average of all textures
            let mut average_texture = textures[0].zeros_like();

            // iterate over all images
            for comp_idx in 0..textures.len() {
                average_texture.add(textures[comp_idx], textures.len());
            }

            // calculate mean value specific for each color channel
            let mean_texture_buffer = average_texture.mean_per_subpixel(mosaic_pattern_width);

            // standard parameters if black level is not available / available
            let hot_pixel_multiplicator = if (black_level[0][0] == -1) { 2.0 } else { 1.0 };
            let mut hot_pixel_threshold = if (black_level[0][0] == -1) { 1.0 } else { 2.0 };
            // X-Trans sensor has more spacing between nearest pixels of same color, need a more relaxed threshold.
            if mosaic_pattern_width == 6 {
                hot_pixel_threshold *= 1.4;
            }

            // Calculate mean black level for each color channel
            let mosaic_pattern_size = mosaic_pattern_width as usize * mosaic_pattern_width as usize;
            let mut black_levels_mean = vec![0.0; mosaic_pattern_size];

            if black_level[0][0] != 1 {
                for channel_idx in 0..mosaic_pattern_size {
                    for img_idx in 0..textures.len() {
                        black_levels_mean[channel_idx] += black_level[img_idx][channel_idx] as f32
                    }

                    black_levels_mean[channel_idx] /= textures.len() as f32
                }
            }

            match mosaic_pattern_width {
                4 => self.backend.find_hotpixels_bayer(
                    average_texture,
                    self,
                    mean_texture_buffer,
                    black_levels_buffer,
                    hot_pixel_threshold,
                    hot_pixel_multiplicator,
                    correction_strength,
                ),
                6 => self.backend.find_hotpixels_xtrans(
                    average_texture,
                    self,
                    mean_texture_buffer,
                    black_levels_buffer,
                    hot_pixel_threshold,
                    hot_pixel_multiplicator,
                    correction_strength,
                ),
                _ => return,
            }

            // -4 in width and height represent that hotpixel correction is not applied on a 2-pixel wide border around the image.
            // This is done so that the algorithm is simpler and comparing neighbours don't have to handle the edge cases.
        }
    }

    #[inline]
    pub fn prepare(
        &self,
        hotpixel_weight_texture: &Image<B>,
        pad_left: u32,
        pad_right: u32,
        pad_top: u32,
        pad_bottom: u32,
        exposure_diff: i32,
        black_level: &[i32],
        mosaic_pattern_width: u32,
    ) -> Image<B> {
        // always use pixel format float32 with increased precision that merging is performed with best possible precision
        let mut out_texture = self.backend.create_image(
            self.width() + pad_left + pad_right,
            self.height() + pad_top + pad_bottom,
        );
        out_texture.fill_zeros();

        self.backend.prepare_texture_bayer(
            self,
            hotpixel_weight_texture,
            black_level,
            pad_left,
            pad_top,
            exposure_diff,
        );

        out_texture
    }

    #[inline]
    pub fn normalize_texture(&self, norm_texture: &mut Image<B>, norm_scalar: f32) {
        self.backend
            .normalize_texture(&self.inner, &mut norm_texture.inner, norm_scalar);
    }

    #[inline]
    pub fn copy_zeroed(&self) -> Image<B> {
        let mut out_texture = self.backend.create_image(self.width(), self.height());
        out_texture.fill_zeros();
        out_texture
    }

    /// Function to calculate the average pixel value over the whole of a texture.
    /// If `per_sub_pixel` is `true`, then each subpixel in the mosaic pattern will have an independent average calculated, producing a `pattern_width * pattern_width` buffer.
    /// If it's `false`, then a single value will be calculated for the whole texture.
    pub fn global_mean(&self, mosaic_pattern_width: u32) -> Image<B> {
        // Create output texture from the y-axis blurring
        let mut summed_y = self.backend.create_image_buffer(
            PixelFormat::R32Float,
            self.width(),
            mosaic_pattern_width,
        );

        // Sum each subpixel of the mosaic vertically along columns, creating a (width, mosaic_pattern_width) sized image
        self.backend.sum_rect_columns_float(
            &mut summed_y,
            self,
            self.height(),
            mosaic_pattern_width,
        );

        // Sum along the row
        // If `per_sub_pixel` is true, then the result is per sub pixel, otherwise a single value is calculated
        let mut sub_buffer =
            vec![0u32; mosaic_pattern_width as usize * mosaic_pattern_width as usize];

        self.backend.sum_row(
            &mut sum_buffer,
            summed_y,
            summed_y.width(),
            mosaic_pattern_width,
        );

        // Calculate the average from the sum
        // let state       = per_sub_pixel ? divide_buffer_state                       : sum_divide_buffer_state
        // let buffer_size = per_sub_pixel ? mosaic_pattern_width*mosaic_pattern_width : 1
        // let avg_buffer  = device.makeBuffer(length: buffer_size*MemoryLayout<Float32>.size, options: .storageModeShared)!
        // command_encoder.setComputePipelineState(state)
        // // If doing per-subpixel, the total number of pixels of each subpixel is 1/(mosaic_pattern_withh)^2 times the total
        // let num_pixels_per_value = Float(in_texture.width * in_texture.height) / (per_sub_pixel ? Float(mosaic_pattern_width*mosaic_pattern_width) : 1.0)
        // let threads_per_grid_divisor = MTLSize(width: buffer_size, height: 1, depth: 1)
        // let threads_per_thread_group_divisor = get_threads_per_thread_group(state, threads_per_grid_divisor)

        // command_encoder.setBuffer(sum_buffer,                                   offset: 0,                            index: 0)
        // command_encoder.setBuffer(avg_buffer,                                   offset: 0,                            index: 1)
        // command_encoder.setBytes([num_pixels_per_value],                        length: MemoryLayout<Float32>.stride, index: 2)
        // command_encoder.setBytes([mosaic_pattern_width*mosaic_pattern_width],   length: MemoryLayout<Int>.stride,     index: 3)
        // command_encoder.dispatchThreads(threads_per_grid_divisor, threadsPerThreadgroup: threads_per_thread_group_divisor)

        // command_encoder.endEncoding()
        // command_buffer.commit()

        return avg_buffer;
    }

    #[inline]
    pub fn upsample(&self, width: Int, height: Int, mode: UpsampleType) -> Image<B> {
        let scale_x = width as f64 / self.width() as f64;
        let scale_y = height as f64 / self.height() as f64;

        // create output texture
        let output_texture = self.copy_zeroed();

        match mode {
            UpsampleType::Bilinear => self.backend.upsample_bilinear_float(
                &self.inner,
                &mut output_texture.inner,
                scale_x,
                scale_y,
            ),
            UpsampleType::NearestNeighbour => self.backend.upsample_nearest_int(
                &self.inner,
                &mut output_texture.inner,
                scale_x,
                scale_y,
            ),
        }

        output_texture
    }
}
