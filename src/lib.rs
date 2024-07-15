// File: lib.rs
// This file stitches the crate together

mod audiofile;
mod window;
pub mod spectrum;
pub mod analysis;
pub mod grain;
pub mod mp;
pub mod operations;
pub mod tuning;

#[doc(inline)]
pub use audiofile::*;
#[doc(inline)]
pub use window::*;
