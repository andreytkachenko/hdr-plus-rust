use encase::internal::Result;
use rawloader::RawImage;

use crate::error::AlignmentError;

/// all the relevant information about image tiles in a single struct
pub struct TileInfo {
    pub tile_size: u32,
    pub tile_size_merge: u32,
    pub search_dist: u32,
    pub n_tiles_x: u32,
    pub n_tiles_y: u32,
    pub n_pos_1d: u32,
    pub n_pos_2d: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TileSize {
    Small = 16,
    Medium = 32,
    Large = 64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchDistance {
    Small = 128,
    Medium = 64,
    Large = 32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergingAlgorithm {
    Fast,
    HighQuality,
}

pub enum ExposureControl {
    Off,
    LinearFullRange,
    Linear1EV,
    Curve0EV,
    Curve1EV,
}

pub enum BitDepth {
    Native,
    Bit16,
}

pub trait Image {
    fn mosaic_pattern(&self) -> MosaicPattern;
    fn white_level(&self) -> WhiteLevel;
    fn black_level(&self) -> WhiteLevel;
    fn exposure_bias(&self);
    fn iso_exposure_time(&self);
    fn color_factors(&self);
}

pub trait Backend {
    type Image: Image;
    type Error: std::error::Error;

    fn load_image(&self, raw: &RawImage) -> Result<Self::Image, Self::Error>;
}

pub struct Denoise<B: Backend> {
    algo: MergingAlgorithm,
    tile_size: TileSize,
    search_distance: SearchDistance,
    noise_reduction: f64,
    exposure_control: ExposureControl,
    output_bit_depth: BitDepth,
    backend: B,
}

impl<B: Backend> Default for Denoise<B> {
    fn default() -> Self {
        Self {
            algo: MergingAlgorithm::Fast,
            tile_size: TileSize::Medium,
            search_distance: SearchDistance::Medium,
            noise_reduction: 13.0,
            exposure_control: ExposureControl::LinearFullRange,
            output_bit_depth: BitDepth::Native,
            backend: B::default(),
        }
    }
}

impl<B: Backend> Denoise<B> {
    /// Convenience function for temporal averaging.
    fn calculate_temporal_average(
        progress: ProcessingProgress,
        mosaic_pattern_width: i32,
        exposure_bias: &[i32],
        white_level: i32,
        black_level: &[&[i32]],
        uniform_exposure: bool,
        color_factors: &[&[f64]],
        images: &[Image],
        hotpixel_weight_image: &Image,
        final_image: &mut Image,
    ) -> Result<(), Error> {
        // find index of image with shortest exposure
        let mut exp_idx = 0;
        for comp_idx in 0..exposure_bias.len() {
            if exposure_bias[comp_idx] < exposure_bias[exp_idx] {
                exp_idx = comp_idx;
            }
        }

        let mut comp_image = self
            .backend
            .create_image_f32(final_image.width(), final_image.height())?;

        // if color_factor is NOT available, it is set to a negative value earlier in the pipeline
        if white_level != -1
            && black_level[0][0] != -1
            && color_factors[0][0] > 0
            && mosaic_pattern_width == 2
        {
            // Temporal averaging with extrapolation of highlights or exposure weighting

            // The averaging is performed in two steps:
            // 1. add all frames of the burst
            // 2. divide the resulting sum of frames by a pixel-specific norm to get the final image

            // The pixel-specific norm has two contributions: norm(x, y) = norm_texture(x, y) + norm_scalar
            // - pixel-specific values calculated by add_texture_exposure() accounting for the different weight given to each pixel based on its brightness
            // - a global scalar if a given frame has the same exposure as the reference frame, which has the darkest exposure
            //
            // For the latter, a single scalar is used as it is more efficient than adding a single, uniform value to each individual pixel in norm_texture

            let norm_image = self
                .backend
                .create_image_f32(final_image.width(), final_image.height())?;

            let mut norm_scalar = 0;

            // 1. add up all frames of the burst
            for comp_idx in 0..images.len() {
                comp_image.prepare_texture(
                    &images[comp_idx],
                    &hotpixel_weight_image,
                    0,
                    0,
                    0,
                    0,
                    exposure_bias[exp_idx] - exposure_bias[comp_idx],
                    black_level[comp_idx],
                    mosaic_pattern_width,
                );

                if exposure_bias[comp_idx] == exposure_bias[exp_idx] {
                    final_image.add_texture_highlights(
                        comp_image,
                        white_level,
                        black_level[comp_idx],
                        color_factors[comp_idx],
                    );
                    norm_scalar += 1;
                } else {
                    final_image.add_texture_exposure(
                        comp_image,
                        norm_image,
                        exposure_bias[comp_idx] - exposure_bias[exp_idx],
                        white_level,
                        black_level[comp_idx],
                        color_factors[comp_idx],
                        mosaic_pattern_width,
                    );
                }

                // DispatchQueue.main.async { progress.int += Int(80_000_000/Double(textures.count)) }
            }

            // 2. divide the sum of frames by a per-pixel norm to get the final image
            norm_image.normalize_texture(final_image, norm_scalar);
        } else {
            // simple temporal averaging
            for comp_idx in 0..images.len() {
                comp_image.prepare_texture(
                    &images[comp_idx],
                    &hotpixel_weight_image,
                    0,
                    0,
                    0,
                    0,
                    0,
                    black_level[comp_idx],
                    mosaic_pattern_width,
                );

                final_image.add_texture(&comp_image, images.len());
                // DispatchQueue.main.async { progress.int += Int(80_000_000/Double(textures.count)) }
            }
        }

        Ok(())
    }

    /// Main function of the app.
    pub fn run(&mut self, images: &[RawImage], ref_idx: Option<usize>) -> Result<(), Error> {
        let mem_info = self.get_memory_info();
        let disk_info = self.get_disk_info();

        // Maximum size for the caches
        let texture_cache_max_size_mb = f64::min(
            10_000.0,
            0.15 * f64(mem_info.physical_memory_size()) / 1000.0 / 1000.0,
        );
        /// The initial value is set to be twice that of the in-memory cache.
        /// It is capped between 4–10 GB, but never allowed to be greater that 15% of the total free disk space.
        /// There is a hard cap of 15% of the total system free disk space.
        let max_dngfolder_size_gb = f64::min(
            10.0,
            f64::min(
                0.15 * disk_info.free_space(),
                f64::max(4.0, 2.0 * texture_cache_max_size_mb / 1000.0),
            ),
        );

        let uniform_exposure = false; // TODO check image exposures - true if all images has same iso_exposure_time

        let images = images
            .into_iter()
            .try_map(|img| self.backend.load_image(img))?;

        let black_level = get_black_leves(images);
        let white_level = get_white_leves(images);
        let color_factors = get_color_factors(images);
        let iso_exposure_time = get_iso_exposure_time(images);
        let mosaic_pattern = get_mosaic_pattern(images);

        let ref_idx = if Some(idx) = ref_idx {
            idx
        } else {
            if uniform_exposure {
                0usize
            } else {
                // TODO find with lowest iso_exposure_time
                todo!();
            }
        };

        // check for non-Bayer sensors that the exposure of images is uniform
        if !uniform_exposure && mosaic_pattern.width != 2 {
            return Err(AlignmentError::NonBayerExposureBracketing);
        }

        if uniform_exposure
            && self.exposure_control != ExposureControl::Off
            && mosaic_pattern.width != 2
        {
            // DispatchQueue.main.async { progress.show_nonbayer_exposure_alert = true }
            self.exposure_control = ExposureControl::Off;
        }

        if uniform_exposure && self.exposure_control != ExposureControl::Off {
            let white_level = vec![-1, white_level.len()];
            let black_level = vec![
                vec![-1, mosaic_pattern.width * mosaic_pattern.width],
                black_level.len(),
            ];
            let color_factors = vec![vec![-1.0, color_factors[0].len()], color_factors.len()];
            let iso_exposure_time = vec![-1.0, iso_exposure_time.len()];
        }

        if self.merging_algorithm == MergingAlgorithm::HighQuality && mosaic_pattern_width != 2 {
            // DispatchQueue.main.async { progress.show_nonbayer_hq_alert = true }
            self.merging_algorithm = MergingAlgorithm::Fast;
        }

        if self.output_bit_depth == BitDepth::Bit16 && mosaic_pattern_width != 2 {
            // DispatchQueue.main.async { progress.show_nonbayer_bit_depth_alert = true }
            self.output_bit_depth = BitDepth::Native;
        }

        let final_image = self.backend.create_image_f32();
        let hotpixel_weight_image = self.backend.create_image_f16();
        hotpixel_weight_image.find_hotpixels(
            textures,
            black_level,
            iso_exposure_time,
            noise_reduction,
            mosaic_pattern_width,
        );

        if noise_reduction == 23.0 {
            sellf.calculate_temporal_average(
                progress,
                mosaic_pattern_width,
                exposure_bias,
                white_level[ref_idx],
                black_level,
                uniform_exposure,
                color_factors,
                textures,
                hotpixel_weight_texture,
                final_texture,
            )?;
        } else if self.merging_algorithm == MergingAlgorithm::HighQuality {
            self.align_merge_frequency_domain(
                progress,
                ref_idx,
                mosaic_pattern_width,
                search_distance,
                tile_size,
                noise_reduction,
                uniform_exposure,
                exposure_bias,
                white_level[ref_idx],
                black_level,
                color_factors,
                textures,
                hotpixel_weight_texture,
                final_texture,
            )?;
        } else {
            self.align_merge_spatial_domain(
                progress,
                ref_idx,
                mosaic_pattern_width,
                search_distance,
                tile_size,
                noise_reduction,
                uniform_exposure,
                exposure_bias,
                black_level,
                color_factors,
                textures,
                hotpixel_weight_texture,
                final_texture,
            )?;
        }

        if (mosaic_pattern_width == 2 && self.exposure_control != ExposureControl::Off) {
            final_texture.correct_exposure(
                white_level[ref_idx],
                black_level,
                exposure_control,
                exposure_bias,
                uniform_exposure,
                color_factors,
                ref_idx,
                mosaic_pattern_width,
            )?
        }

        // apply scaling to 16 bit
        let factor_16bit = if (self.output_bit_depth == BitDepth::Bit16
            && mosaic_pattern_width == 2
            && self.exposure_control != ExposureControl::Off)
        {
            i32(f64::pow(2.0, 16.0 - f64::ceil(f64::log2(f64(white_level[ref_idx])))) + 0.5)
        } else {
            1
        };

        let output_texture_uint16 = convert_float_to_uint16(
            final_texture,
            if white_level[ref_idx] == -1 {
                1000000
            } else {
                factor_16bit * white_level[ref_idx]
            },
            black_level[ref_idx],
            factor_16bit,
            mosaic_pattern_width,
        )?;

        // save output_texture_uint16 to raw dng

        Ok(())
    }
}
