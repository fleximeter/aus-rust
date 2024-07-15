/// File: tests.rs
/// This file contains functionality for testing the package.

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
        Err(_) => panic!("could not read audio")
    };
    operations::adjust_level(&mut audio.samples[0], -12.0);
    operations::fade_in(&mut audio.samples[0], spectrum::WindowType::Hanning, 44100 * 4);
    operations::fade_out(&mut audio.samples[0], spectrum::WindowType::Hanning, 44100 * 4);
    let path: String = String::from("D:\\Recording\\out1.wav");
    match audiofile::write(&path, &audio) {
        Ok(_) => (),
        Err(_) => panic!("could not write audio")
    }
    // let (magnitude_spectrum, phase_spectrum) = spectrum::complex_to_polar_rfft(spectrum::rfft(&mut audio.samples[0][0..2048].to_vec(), 2048));
    // println!("{:?}", magnitude_spectrum);
}

/// Test force equal energy
pub fn basic_tests2() {
    let path = String::from("D:\\Recording\\out1.wav");
    let mut audio = match audiofile::read(&path) {
        Ok(x) => x,
        Err(_) => panic!("could not read audio")
    };
    operations::force_equal_energy(&mut audio.samples[0], -16.0, 16384);
    let path: String = String::from("D:\\Recording\\out2.wav");
    match audiofile::write(&path, &audio) {
        Ok(_) => (),
        Err(_) => panic!("could not write audio")
    }    
}

/// Test STFT/ISTFT
pub fn basic_tests3() {
    let fft_size: usize = 4096;
    let hop_size: usize = fft_size / 2;
    let window_type = spectrum::WindowType::Hamming;
    let path = String::from("D:\\Recording\\Samples\\Iowa\\Viola.pizz.mono.2444.1\\samples\\sample.48.Viola.pizz.sulC.ff.C3B3.mono.wav");
    let mut audio = match audiofile::read(&path) {
        Ok(x) => x,
        Err(_) => panic!("could not read audio")
    };
    let mut spectrogram: Vec<Vec<Complex<f64>>> = spectrum::rstft(&mut audio.samples[0], fft_size, hop_size, window_type);
    let (mags, phases) = spectrum::complex_to_polar_rstft(&spectrogram);
    let mut output_spectrogram = spectrum::polar_to_complex_rstft(&mags, &phases).unwrap();
    let output_audio = spectrum::irstft(&output_spectrogram, fft_size, hop_size, window_type).unwrap();

    let output_audiofile = audiofile::AudioFile::new_mono(audiofile::AudioFormat::S24, 44100, output_audio);
    let path: String = String::from("D:\\Recording\\out3.wav");
    match audiofile::write(&path, &output_audiofile) {
        Ok(_) => (),
        Err(_) => panic!("could not write audio")
    }
}

/// Test analysis on an audio file.
/// This test does not use multithreading, so it will probably take much longer.
pub fn basic_tests4() {
    let fft_size: usize = 4096;
    let hop_size: usize = fft_size / 2;
    let window_type = spectrum::WindowType::Hamming;
    let path = String::from("D:\\Recording\\grains.wav");
    let mut audio = match audiofile::read(&path) {
        Ok(x) => x,
        Err(_) => panic!("could not read audio")
    };
    let stft_imaginary_spectrum: Vec<Vec<Complex<f64>>> = spectrum::rstft(&mut audio.samples[0], fft_size, hop_size, window_type);
    let (stft_magnitude_spectrum, _) = spectrum::complex_to_polar_rstft(&stft_imaginary_spectrum);
    let mut analyses: Vec<analysis::Analysis> = Vec::with_capacity(stft_magnitude_spectrum.len());
    for i in 0..stft_magnitude_spectrum.len() {
        analyses.push(analysis::analyzer(&stft_magnitude_spectrum[i], fft_size, audio.sample_rate));
    }
}

/// Test stochastic exchange with STFT/ISTFT
pub fn basic_tests5() {
    let fft_size: usize = 4096;
    let hop_size: usize = fft_size / 2;
    let window_type = spectrum::WindowType::Hamming;
    let path = String::from("D:\\Recording\\Samples\\freesound\\creative_commons_0\\wind_chimes\\eq\\217800__minian89__wind_chimes_eq.wav");
    let mut audio = match audiofile::read(&path) {
        Ok(x) => x,
        Err(_) => panic!("could not read audio")
    };
    audiofile::mixdown(&mut audio);
    let mut spectrogram: Vec<Vec<Complex<f64>>> = spectrum::rstft(&mut audio.samples[0], fft_size, hop_size, window_type);
    let (mut magnitude_spectrogram, mut phase_spectrogram) = spectrum::complex_to_polar_rstft(&spectrogram);

    // Perform spectral operations here
    spectrum::stft_exchange_frames_stochastic(&mut magnitude_spectrogram, &mut phase_spectrogram, 20);
    
    // Perform ISTFT and add fade in/out
    let output_spectrogram = spectrum::polar_to_complex_rstft(&magnitude_spectrogram, &phase_spectrogram).unwrap();
    let mut output_audio: Vec<f64> = spectrum::irstft(&output_spectrogram, fft_size, hop_size, window_type).unwrap();
    operations::fade_in(&mut output_audio, spectrum::WindowType::Hanning, 1000);
    operations::fade_out(&mut output_audio, spectrum::WindowType::Hanning, 1000);

    // Generate the output audio file
    let output_audiofile = audiofile::AudioFile::new_mono(audiofile::AudioFormat::S24, 44100, output_audio);
    let path: String = String::from("D:\\Recording\\out5.wav");
    match audiofile::write(&path, &output_audiofile) {
        Ok(_) => (),
        Err(_) => panic!("could not write audio")
    }
}

/// Test multithreaded spectral analyzer
pub fn basic_tests6() {
    let fft_size: usize = 4096;
    let path = String::from("D:\\Recording\\grains.wav");
    let mut audio = match audiofile::read(&path) {
        Ok(x) => x,
        Err(_) => panic!("could not read audio")
    };
    let _ = mp::stft_analysis(&mut audio.samples[0], fft_size, audio.sample_rate);
}

/// Test spectral freeze
pub fn basic_tests7() {
    let fft_size: usize = 4096;
    let hop_size: usize = fft_size / 2;
    let window_type = spectrum::WindowType::Hamming;
    let path = String::from("D:\\Recording\\Samples\\Iowa\\Cello.arco.mono.2444.1\\samples_ff\\sample_Cello.arco.ff.sulC.C2B2.wav_0.wav");
    let mut audio = match audiofile::read(&path) {
        Ok(x) => x,
        Err(_) => panic!("could not read audio")
    };
    let spectrogram = spectrum::rstft(&mut audio.samples[0], fft_size, hop_size, window_type);
    let (mag, phase) = spectrum::complex_to_polar_rstft(&spectrogram);
    let (freeze_mag, freeze_phase) = spectrum::fft_freeze(&mag[8], &phase[8], 50, fft_size / 2);
    let mut freeze_spectrogram = spectrum::polar_to_complex_rstft(&freeze_mag, &freeze_phase).unwrap();
    let mut output_audio = spectrum::irstft(&mut freeze_spectrogram, fft_size, fft_size / 2, spectrum::WindowType::Hamming).unwrap();

    // Fade in and out at beginning and end
    operations::fade_in(&mut output_audio, spectrum::WindowType::Hanning, 10000);
    operations::fade_out(&mut output_audio, spectrum::WindowType::Hanning, 10000);

    // Make output audio file
    let output_audiofile = audiofile::AudioFile::new_mono(audiofile::AudioFormat::S24, 44100, output_audio);
    let path: String = String::from("D:\\Recording\\out7.wav");
    match audiofile::write(&path, &output_audiofile) {
        Ok(_) => (),
        Err(_) => panic!("could not write audio")
    }
}

/// Test convolution
pub fn basic_tests8() {
    let fft_size: usize = 2048;

    let audio_path = String::from("D:\\Recording\\Samples\\Iowa\\Cello.arco.mono.2444.1\\samples_ff\\sample_Cello.arco.ff.sulC.C2B2.wav_0.wav");
    let ir_path = String::from("D:\\Recording\\Samples\\Impulse_Responses\\560377__manysounds__1000-liter-wine-tank-plate-spring-impulse.wav");
    let mut audio = match audiofile::read(&audio_path) {
        Ok(x) => x,
        Err(_) => panic!("could not read audio")
    };
    let mut ir = match audiofile::read(&ir_path) {
        Ok(x) => x,
        Err(_) => panic!("could not read audio")
    };

    let zeros: Vec<f64> = vec![0.0; ir.samples[0].len()];
    audio.samples[0].extend(&zeros);
    
    let mut output_audio = spectrum::partitioned_convolution(&mut audio.samples[0], &mut ir.samples[0], fft_size).unwrap();

    // Fade in and out at beginning and end
    operations::adjust_level(&mut output_audio, -6.0);
    operations::fade_in(&mut output_audio, spectrum::WindowType::Hanning, 10000);
    operations::fade_out(&mut output_audio, spectrum::WindowType::Hanning, 10000);

    // Make output audio file
    let output_audiofile = audiofile::AudioFile::new_mono(audiofile::AudioFormat::S24, 44100, output_audio);
    let path: String = String::from("D:\\Recording\\out8.wav");
    match audiofile::write(&path, &output_audiofile) {
        Ok(_) => (),
        Err(_) => ()
    }
}

/// Test pyin
pub fn basic_tests9() {
    let fft_size: usize = 2048;

    let audio_path = String::from("D:\\Recording\\Samples\\freesound\\creative_commons_0\\granulation\\159130__cms4f__flute-play-c-11.wav");
    let mut audio = match audiofile::read(&audio_path) {
        Ok(x) => x,
        Err(_) => panic!("could not read audio")
    };
    
    //let result = analysis::pyin_pitch_estimator(&audio.samples[0], audio.sample_rate, 50.0, 500.0, fft_size);
    let result = analysis::yin_pitch_estimator(&audio.samples[0][20000..22048], audio.sample_rate, fft_size);
    println!("Result: {}, confidence: {}", result.0, result.1);
    //println!("{}", analysis::pyin_pitch_estimator_single(&audio.samples[0], audio.sample_rate, 50.0, 500.0));
}
