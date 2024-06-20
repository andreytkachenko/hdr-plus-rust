trait ExposureBackend {
    /// Correct Exposure (Non-linear)
    fn correct_exposure(&self);

    /// Correct Exposure (Linear)
    fn correct_exposure_linear(&self);

    /// Maximum (X-Direction)
    fn max_x(&self);

    /// Maximum (Y-Direction)
    fn max_y(&self);
}

pub struct Exposure<B: ExposureBackend> {
    _b: B,
}

impl<B: ExposureBackend> Exposure<B> {
    fn correct_exposure(
        final_texture: Image,
        white_level: i32,
        black_level: &[&[i32]],
        exposure_control: String,
        exposure_bias: &[i32],
        uniform_exposure: bool,
        color_factors: &[&[f64]],
        ref_idx: i32,
        mosaic_pattern_width: i32,
    ) {
    }

    fn texture_max(in_texture: Image) -> MTLBuffer {}
}
