//! # Audio and spectral analysis
//! The `analysis` module contains functionality for audio and spectrum analysis.
//! Some analysis tools are based on formulas from Florian Eyben, "Real-Time Speech and Music Classification," Springer, 2016.

mod spectrum_analyzer;
mod spectral_analysis_tools;
mod audio_analysis_tools;
pub mod mel;

#[doc(inline)]
pub use spectrum_analyzer::*;
#[doc(inline)]
pub use audio_analysis_tools::*;
#[doc(inline)]
pub use spectral_analysis_tools::{alpha_ratio, autocorrelation, autocorrelation_power_spectrum, 
    hammarberg_index, harmonicity, make_power_spectrogram, make_power_spectrum, make_log_spectrogram, 
    make_log_spectrum, normalize_spectrogram, normalize_spectrum, spectral_centroid, spectral_difference, 
    spectral_entropy, spectral_flatness, spectral_flux, spectral_kurtosis, spectral_roll_off_point, 
    spectral_skewness, spectral_slope, spectral_slope_region, spectral_variance};
