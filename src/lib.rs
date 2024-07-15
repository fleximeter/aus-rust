// File: lib.rs
// This file stitches the crate together

#![crate_name = "audiorust"]

mod audiofile;
pub mod spectrum;
pub mod analysis;
pub mod grain;
pub mod mp;
pub mod operations;
pub mod tuning;

pub use audiofile::*;
