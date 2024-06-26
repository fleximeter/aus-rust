// File: tests.rs
// This file contains functionality for testing the package.

use crate::audiofile;
use crate::operations;
use crate::spectrum;
use crate::analysis;
use crate::tuning;
use crate::grain;
use crate::mp;
use num::Complex;

/// Tests of level adjustment and fade in / fade out
pub fn basic_tests1() {
    let path = String::from("D:\\Recording\\grains.wav");
    let mut audio = match audiofile::read(&path) {
        Ok(x) => x,
        Err(err) => panic!("could not read audio")
    };
    operations::adjust_level(&mut audio.samples, -12.0);
    operations::fade_in(&mut audio.samples[0], spectrum::WindowType::Hanning, 44100 * 4);
    operations::fade_out(&mut audio.samples[0], spectrum::WindowType::Hanning, 44100 * 4);
    match audiofile::write(String::from("D:\\Recording\\out1.wav"), &audio) {
        Ok(x) => (),
        Err(err) => panic!("could not write audio")
    }
    // let (magnitude_spectrum, phase_spectrum) = spectrum::complex_to_polar_rfft(spectrum::rfft(&mut audio.samples[0][0..2048].to_vec(), 2048));
    // println!("{:?}", magnitude_spectrum);
}

/// Test force equal energy
pub fn basic_tests2() {
    let path = String::from("D:\\Recording\\out1.wav");
    let mut audio = match audiofile::read(&path) {
        Ok(x) => x,
        Err(err) => panic!("could not read audio")
    };
    operations::force_equal_energy(&mut audio.samples[0], -16.0, 16384);
    match audiofile::write(String::from("D:\\Recording\\out2.wav"), &audio) {
        Ok(x) => (),
        Err(err) => panic!("could not write audio")
    }    
}

/// Test STFT/ISTFT
pub fn basic_tests3() {
    let fft_size: usize = 4096;
    let path = String::from("D:\\Recording\\Samples\\Iowa\\Viola.pizz.mono.2444.1\\samples\\sample.48.Viola.pizz.sulC.ff.C3B3.mono.wav");
    let mut audio = match audiofile::read(&path) {
        Ok(x) => x,
        Err(err) => panic!("could not read audio")
    };
    let mut spectrogram: Vec<Vec<Complex<f64>>> = match spectrum::rstft(&mut audio.samples[0], fft_size, fft_size / 2, spectrum::WindowType::Hamming) {
        Ok(x) => x,
        Err(_err) => Vec::new()
    };
    let output_audio: Vec<f64> = match spectrum::irstft(&mut spectrogram, fft_size, fft_size / 2, spectrum::WindowType::Hamming) {
        Ok(x) => x,
        Err(_err) => Vec::new()
    };
    let mut output_audio_channels: Vec<Vec<f64>> = Vec::with_capacity(1);
    output_audio_channels.push(output_audio);
    let mut output_audiofile: audiofile::AudioFile = audio.copy_header();
    output_audiofile.num_frames = output_audio_channels[0].len();
    output_audiofile.duration = output_audiofile.num_frames as f64 / output_audiofile.sample_rate as f64;
    output_audiofile.samples = output_audio_channels;
    match audiofile::write(String::from("D:\\Recording\\out3.wav"), &audio) {
        Ok(x) => (),
        Err(err) => panic!("could not write audio")
    }}

/// Test analysis on an audio file.
/// This test does not use multithreading, so it will probably take much longer.
pub fn basic_tests4() {
    let fft_size: usize = 4096;
    let path = String::from("D:\\Recording\\grains.wav");
    let mut audio = match audiofile::read(&path) {
        Ok(x) => x,
        Err(err) => panic!("could not read audio")
    };
    let stft_imaginary_spectrum: Vec<Vec<Complex<f64>>> = match spectrum::rstft(&mut audio.samples[0], fft_size, fft_size / 2, spectrum::WindowType::Hamming) {
        Ok(x) => x,
        Err(_err) => Vec::new()
    };
    let (stft_magnitude_spectrum, _) = spectrum::complex_to_polar_rstft(stft_imaginary_spectrum);
    let mut analyses: Vec<analysis::Analysis> = Vec::with_capacity(stft_magnitude_spectrum.len());
    for i in 0..stft_magnitude_spectrum.len() {
        analyses.push(analysis::analyzer(&stft_magnitude_spectrum[i], fft_size, audio.sample_rate as u16));
    }
}

/// Test stochastic exchange with STFT/ISTFT
pub fn basic_tests5() {
    let fft_size: usize = 4096;
    let path = String::from("D:\\Recording\\Samples\\freesound\\creative_commons_0\\wind_chimes\\eq\\217800__minian89__wind_chimes_eq.wav");
    let mut audio = match audiofile::read(&path) {
        Ok(x) => x,
        Err(err) => panic!("could not read audio")
    };
    audiofile::mixdown(&mut audio);
    let mut spectrogram: Vec<Vec<Complex<f64>>> = match spectrum::rstft(&mut audio.samples[0], fft_size, fft_size / 2, spectrum::WindowType::Hamming) {
        Ok(x) => x,
        Err(_err) => Vec::new()
    };
    
    // Perform spectral operations here
    operations::stochastic_exchange_stft(&mut spectrogram, 20);
    
    // Perform ISTFT and add fade in/out
    let mut output_audio: Vec<f64> = match spectrum::irstft(&mut spectrogram, fft_size, fft_size / 2, spectrum::WindowType::Hamming) {
        Ok(x) => x,
        Err(_err) => Vec::new()
    };
    operations::fade_in(&mut output_audio, spectrum::WindowType::Hanning, 1000);
    operations::fade_out(&mut output_audio, spectrum::WindowType::Hanning, 1000);

    // Generate the output audio file
    let mut output_audio_channels: Vec<Vec<f64>> = Vec::with_capacity(1);
    output_audio_channels.push(output_audio);
    let mut output_audiofile: audiofile::AudioFile = audio.copy_header();
    output_audiofile.num_channels = 1;
    output_audiofile.num_frames = output_audio_channels[0].len();
    output_audiofile.duration = output_audiofile.num_frames as f64 / output_audiofile.sample_rate as f64;
    output_audiofile.samples = output_audio_channels;
    match audiofile::write(String::from("D:\\Recording\\out5.wav"), &audio) {
        Ok(x) => (),
        Err(err) => panic!("could not write audio")
    }
}

/// Test multithreaded spectral analyzer
pub fn basic_tests6() {
    let fft_size: usize = 4096;
    let path = String::from("D:\\Recording\\grains.wav");
    let mut audio = match audiofile::read(&path) {
        Ok(x) => x,
        Err(err) => panic!("could not read audio")
    };
    let a = mp::stft_analysis(&mut audio.samples[0], fft_size, audio.sample_rate as u16);
}
