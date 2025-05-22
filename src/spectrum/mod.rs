//! # Spectrum
//! The `spectrum` module has a collection of spectral functionality, including
//! convenience real FFT/IFFT implementations, and a real STFT/ISTFT pair.

mod fft_tools;
mod fft;
mod transformations;

#[doc(inline)]
pub use fft_tools::*;
#[doc(inline)]
pub use fft::*;
#[doc(inline)]
pub use transformations::*;
