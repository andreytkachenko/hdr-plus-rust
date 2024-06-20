#[derive(Debug, thiserror::Error)]
pub enum AlignmentError {
    #[error("LessThanTwoImages")]
    LessThanTwoImages,

    #[error("InconsistentExtensions")]
    InconsistentExtensions,

    #[error("InconsistentResolutions")]
    InconsistentResolutions,

    #[error("ConversionFailed")]
    ConversionFailed,

    #[error("MissingDngConverter")]
    MissingDngConverter,

    #[error("NonBayerExposureBracketing")]
    NonBayerExposureBracketing,
}
