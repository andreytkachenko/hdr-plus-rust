mod wgpu;

pub trait Backend {
    type ImageBuffer;
    type Error;
}

pub trait ImageBackend: Backend {
    /// Add Texture
    fn add_texture(&self);

    /// Add Texture (Exposure)
    fn add_texture_exposure(&self);

    /// Add Texture (Highlights)
    fn add_texture_highlights(&self);

    /// Add Texture (UInt16
    fn add_texture_uint16(&self);

    /// Add Texture (Weighted)
    fn add_texture_weighted(&self);

    /// Blur Mosaic Texture
    fn blur_mosaic_texture(&self);

    /// Calculate Highlight Weights
    fn calculate_weight_highlights(&self);

    /// Convert Float to UInt16
    fn convert_float_to_uint16(&self);

    /// Convert RGBA to Bayer
    fn convert_to_bayer(&self);

    /// Covert Bayer to RGBA
    fn convert_to_rgba(&self);

    /// Copy Texture
    fn copy_texture(&self);

    /// Crop Texture
    fn crop_texture(&self);

    /// Divide Buffer Per Sub Pixel
    fn divide_buffer(&self);

    /// Sum and Divide Buffer Total
    fn sum_divide_buffer(&self);

    /// Fill With Zeros
    fn fill_with_zeros(&self);

    /// Find Hotpixels (Bayer)
    fn find_hotpixels_bayer(&self);

    /// Find Hotpixels (XTrans)
    fn find_hotpixels_xtrans(&self);

    /// Normalize Texture
    fn normalize_texture(&self);

    /// Prepare Texture (Bayer)
    fn prepare_texture_bayer(&self);

    /// Sum Along Columns Inside A Rect (Float)
    fn sum_rect_columns_float(&self);

    /// Sum Along Columns Inside A Rect (UInt)
    fn sum_rect_columns_uint(&self);

    /// Sum Along Rows
    fn sum_row(&self);

    /// Upsample (Bilinear) (Float)
    fn upsample_bilinear_float(
        &self,
        into: &mut Self::ImageBuffer,
        scale_x: f32,
        scale_y: f32,
    ) -> Result<(), Self::Error>;

    /// Upsample (Nearest Neighbour) (Int)
    fn upsample_nearest_int(
        &self,
        into: &mut Self::ImageBuffer,
        scale_x: f32,
        scale_y: f32,
    ) -> Result<(), Self::Error>;

    /// Avg Pool
    fn avg_pool(
        &self,
        scale: u32,
        black_level_mean: f64,
        normalization: bool,
        color_factors3: Vec<f64>,
    ) -> Result<Self::ImageBuffer, Self::Error>;

    /// Avg Pool (Normalized)
    fn avg_pool_normalization(&self);
}

pub trait ExposureBackend {
    // Correct Exposure (Non-linear)
    fn correct_exposure(&self);

    // Correct Exposure (Linear)
    fn correct_exposure_linear(&self);

    // Maximum (X-Direction)
    fn max_x(&self);

    // Maximum (Y-Direction)
    fn max_y(&self);
}

pub trait SpatialMergeBackend {
    /// Color Difference
    fn color_difference(&self);

    /// Compute Merging Weight
    fn compute_merge_weight(&self);
}

pub trait FrequencyMergeBackend {
    /// Frequency Domain Merge
    fn merge_frequency_domain(&self);

    /// Calculate Abs Diff RGBA
    fn calculate_abs_diff_rgba(&self);

    /// Calculate Highlights Norm RGBA
    fn calculate_highlights_norm_rgba(&self);

    /// Calculate Mismatch RGBA
    fn calculate_mismatch_rgba(&self);

    /// Calculate RMS RGBA
    fn calculate_rms_rgba(&self);

    /// Deconvolute Frequency Domain
    fn deconvolute_frequency_domain(&self);

    /// Normalize Mismatch
    fn normalize_mismatch(&self);

    /// Reduce Artifacts at Tile Borders
    fn reduce_artifacts_tile_border(&self);

    /// Backwards Optimized Fast Fourier Transform
    fn backward_dft(&self);

    /// Backwards Discrete Fourier Transform
    fn backward_fft(&self);

    /// Forwards Optimized Fast Fourier Transform
    fn forward_dft(&self);

    /// Forwards Discrete Fourier Transform
    fn forward_fft(&self);
}

pub trait AlignBackend {
    /// Compute Tile Difference
    fn compute_tile_differences(&self);

    /// Compute Tile Difference (N=25)
    fn compute_tile_differences25(&self);

    /// Compute Tile Difference (N=25) (Exposure)
    fn compute_tile_differences_exposure25(&self);

    /// Correct Upsampling Error
    fn correct_upsampling_error(&self);

    /// Find Best Tile Alignment
    fn find_best_tile_alignment(&self);

    /// Warp Texture (Bayer)
    fn warp_texture_bayer(&self);

    /// Warp Texture (XTrans)
    fn warp_texture_xtrans(&self);
}
