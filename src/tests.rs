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

/// Test STFT/ISTFT
pub fn basic_tests3() {
    let path = String::from("D:\\Recording\\Samples\\Iowa\\Viola.pizz.mono.2444.1\\samples\\sample.48.Viola.pizz.sulC.ff.C3B3.mono.wav");
    let mut audio = audiofile::read(&path);
    let mut spectrogram = spectrum::rstft(&mut audio.samples[0], 4096, 2048, spectrum::WindowType::Hamming);
    let output_audio = spectrum::irstft(&mut spectrogram, 4096, 2048, spectrum::WindowType::Hamming);
    let mut output_audio_channels: Vec<Vec<f64>> = Vec::with_capacity(1);
    output_audio_channels.push(output_audio);
    let mut output_audiofile: audiofile::AudioFile = audio.copy_header();
    output_audiofile.num_frames = output_audio_channels[0].len();
    output_audiofile.duration = output_audiofile.num_frames as f64 / output_audiofile.sample_rate as f64;
    output_audiofile.samples = output_audio_channels;
    audiofile::write(String::from("D:\\Recording\\out3.wav"), &output_audiofile);    
}

/// Test analysis
pub fn basic_tests4() {
    let path = String::from("D:\\Recording\\grains.wav");
    let mut audio = audiofile::read(&path);
    let a = analysis::analyzer(&mut audio.samples[0], 4096, audio.sample_rate as u16);
}

/// Test overlap add
pub fn basic_tests5() {
    let mut audio_chunks: Vec<Vec<f64>> = Vec::with_capacity(10);
    for i in 0..10 {
        let chunk = spectrum::hamming(2048);
        audio_chunks.push(chunk);
    }
    let mut audio = spectrum::overlap_add(&audio_chunks, 2048, 1024);

    // Adjust levels
    let mut maxval = 0.0;
    for i in 0..audio.len() {
        maxval = f64::max(audio[i], maxval);
    }

    for i in 0..audio.len() {
        audio[i] /= maxval * 2.0;
    }

    let mut audio_channels: Vec<Vec<f64>> = Vec::with_capacity(1);
    audio_channels.push(audio);
    let a = audiofile::AudioFile {
        audio_format: audiofile::AudioFormat::S24,
        bits_per_sample: 24,
        duration: 0.0,
        num_channels: 1,
        num_frames: audio_channels[0].len(),
        sample_rate: 44100,
        samples: audio_channels
    };
    audiofile::write(String::from("D:\\Recording\\out4.wav"), &a);
}