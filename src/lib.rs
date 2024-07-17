//! # aus
//! This is a crate for audio processing and analysis in Rust, combining new functionality with aggregated 
//! functionality from other existing crates. For example, this crate provides wrappers for `rustfft`, 
//! allowing a FFT to be performed with a single function call. It also has a STFT/ISTFT function pair. 
//! It also has built-in window generation in the style of `numpy`. And there are implementations of spectral 
//! feature extraction, such as calculating spectral centroid, entropy, slope, etc.

mod audiofile;
mod window;
pub mod spectrum;
pub mod analysis;
pub mod grain;
pub mod mp;
pub mod operations;
pub mod synthesis;
pub mod tuning;

#[doc(inline)]
pub use audiofile::*;
#[doc(inline)]
pub use window::*;
