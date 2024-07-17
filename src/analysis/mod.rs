//! # Analysis
//! The `analysis` module contains functionality for audio and spectrum analysis.
//! Some analysis tools are based on formulas from Florian Eyben, "Real-Time Speech and Music Classification," Springer, 2016.

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
