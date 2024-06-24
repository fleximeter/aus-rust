// File: tests.rs
// This file contains functionality for testing the package.

use crate::audiofile;
use crate::operations;
use crate::spectrum;
use crate::analysis;
use crate::tuning;
use crate::grain;

/// Tests of level adjustment and fade in / fade out
pub fn basic_tests1() {
    let path = String::from("D:\\Recording\\grains.wav");
    let mut audio = audiofile::read(&path);
    operations::adjust_level(&mut audio.samples, -12.0);
    operations::fade_in(&mut audio.samples[0], spectrum::WindowType::Hanning, 44100 * 4);
    operations::fade_out(&mut audio.samples[0], spectrum::WindowType::Hanning, 44100 * 4);
    audiofile::write(String::from("D:\\Recording\\out1.wav"), &audio)
    // let (magnitude_spectrum, phase_spectrum) = spectrum::complex_to_polar_rfft(spectrum::rfft(&mut audio.samples[0][0..2048].to_vec(), 2048));
    // println!("{:?}", magnitude_spectrum);
}

/// Test force equal energy
pub fn basic_tests2() {
    let path = String::from("D:\\Recording\\out1.wav");
    let mut audio = audiofile::read(&path);
    operations::force_equal_energy(&mut audio.samples[0], -16.0, 16384);
    audiofile::write(String::from("D:\\Recording\\out2.wav"), &audio)
    
}

/// Test force equal energy
pub fn basic_tests3() {
    let path = String::from("D:\\Recording\\Samples\\Iowa\\Viola.pizz.mono.2444.1\\samples\\sample.48.Viola.pizz.sulC.ff.C3B3.mono.wav");
    let mut audio = audiofile::read(&path);
    let mut spectrogram = spectrum::rstft(&mut audio.samples[0], 1024, 512, spectrum::WindowType::Bartlett);
    let output_audio = spectrum::irstft(&mut spectrogram, 1024, 512, spectrum::WindowType::Bartlett);
    let mut output_audio_channels: Vec<Vec<f64>> = Vec::with_capacity(1);
    output_audio_channels.push(output_audio);
    let output_audiofile: audiofile::AudioFile = audiofile::AudioFile {
        audio_format: audiofile::AudioFormat::S24,
        bits_per_sample: 24,
        duration: 0.0,
        num_channels: 1,
        num_frames: output_audio_channels[0].len(),
        sample_rate: 44100,
        samples: output_audio_channels
    };
    audiofile::write(String::from("D:\\Recording\\out3.wav"), &output_audiofile);    
}
