/// File: mod.rs
/// This file stitches the analysis module together

mod analyzer;
mod spectral_analysis_tools;
mod audio_analysis_tools;

#[doc(inline)]
pub use analyzer::*;
#[doc(inline)]
pub use audio_analysis_tools::*;
#[doc(inline)]
pub use spectral_analysis_tools::{make_power_spectrum, spectral_centroid, spectral_entropy, spectral_flatness, 
    spectral_kurtosis, spectral_roll_off_point, spectral_skewness, spectral_slope, spectral_slope_region, 
    spectral_variance};
