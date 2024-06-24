// File: tests.rs
// This file contains functionality for testing the package.

use crate::audiofile;
use crate::operations;
use crate::spectrum;

/// Basic tests of level adjustment
pub fn basic_tests() {
    let path = String::from("D:\\Recording\\grains.wav");
    let mut audio = audiofile::read(&path);
    operations::adjust_level(&mut audio.samples, -36.0);
    operations::fade_in(&mut audio.samples[0], spectrum::WindowType::Hanning, 88200);
    operations::fade_out(&mut audio.samples[0], spectrum::WindowType::Hanning, 88200);
    audiofile::write(String::from("D:\\Recording\\out.wav"), &audio)
    // let (magnitude_spectrum, phase_spectrum) = spectrum::complex_to_polar_rfft(spectrum::rfft(&mut audio.samples[0][0..2048].to_vec(), 2048));
    // println!("{:?}", magnitude_spectrum);
}