// File: spectrum.rs
// This file contains spectral resource functionality.

use realfft::RealFftPlanner;

/// Creates a Bartlett window of size m
pub fn bartlett(m: usize) -> Vec<f64>{
    let mut window: Vec<f64> = Vec::with_capacity(m);
    for i in 0..m {
        window[i] = (2.0 / (m as f64 - 1.0)) * ((m as f64 - 1.0) / 2.0 - f64::abs(i as f64 - (m as f64 - 1.0) / 2.0));
    }
    window
}

/// Creates a Blackman window of size m
pub fn blackman(m: usize) -> Vec<f64>{
    let mut window: Vec<f64> = Vec::with_capacity(m);
    for i in 0..m {
        window[i] = 0.42 - 0.5 * f64::cos((2.0 * std::f64::consts::PI * i as f64) / (m as f64)) 
            + 0.08 * f64::cos((4.0 * std::f64::consts::PI * i as f64) / (m as f64));
    }
    window
}

/// Creates a Hanning window of size m
pub fn hanning(m: usize) -> Vec<f64>{
    let mut window: Vec<f64> = Vec::with_capacity(m);
    for i in 0..m {
        window[i] = 0.5 - 0.5 * f64::cos((2.0 * std::f64::consts::PI * i as f64) / (m as f64 - 1.0));
    }
    window
}

/// Creates a Hamming window of size m
pub fn hamming(m: usize) -> Vec<f64>{
    let mut window: Vec<f64> = Vec::with_capacity(m);
    for i in 0..m {
        window[i] = 0.54 - 0.46 * f64::cos((2.0 * std::f64::consts::PI * i as f64) / (m as f64 - 1.0));
    }
    window
}

/// Calculates the real FFT of a chunk of audio.
/// 
/// The input audio must be a 1D vector already of the appropriate size.
/// This function will return the magnitude and phase spectrum.
pub fn rfft(audio: &mut Vec<f64>, fft_size: usize) -> (Vec<f64>, Vec<f64>) {
    let mut real_planner = RealFftPlanner::<f64>::new();
    let r2c = real_planner.plan_fft_forward(fft_size);
    let mut input = audio;
    let mut spectrum = r2c.make_output_vec();
    assert_eq!(input.len(), fft_size);
    assert_eq!(spectrum.len(), fft_size / 2 + 1);
    r2c.process(&mut input, &mut spectrum).unwrap();
    let mut magnitude_spectrum = vec![0.0 as f64; spectrum.len()];
    let mut phase_spectrum = vec![0.0 as f64; spectrum.len()];
    for i in 0..spectrum.len() {
        magnitude_spectrum[i] = f64::sqrt(f64::powf(spectrum[i].re, 2.0) + f64::powf(spectrum[i].im, 2.0));
        phase_spectrum[i] = f64::atan2(spectrum[i].im, spectrum[i].re);
    }
    (magnitude_spectrum, phase_spectrum)
}

/// Gets the corresponding frequencies for rFFT data
pub fn rfftfreq(fft_size: usize, sample_rate: u16) -> Vec<f64> {
    let mut freqs = vec![0.0 as f64; fft_size / 2 + 1];
    let f_0 = sample_rate as f64 / fft_size as f64;
    for i in 1..freqs.len() {
        freqs[i] = f_0 * i as f64;
    }
    freqs
}

/// Calculates the real STFT of a chunk of audio.
/// 
/// The input audio must be a 1D vector already of the appropriate size.
/// This function will return the magnitude and phase spectrum.
pub fn rstft(audio: &mut Vec<f64>, fft_size: usize, hop_size: usize) -> (Vec<f64>, Vec<f64>) {
    let mut real_planner = RealFftPlanner::<f64>::new();
    let r2c = real_planner.plan_fft_forward(fft_size);
    let mut input = audio;
    let mut spectrum = r2c.make_output_vec();
    assert_eq!(input.len(), fft_size);
    assert_eq!(spectrum.len(), fft_size / 2 + 1);
    r2c.process(&mut input, &mut spectrum).unwrap();
    let mut magnitude_spectrum = vec![0.0 as f64; spectrum.len()];
    let mut phase_spectrum = vec![0.0 as f64; spectrum.len()];
    for i in 0..spectrum.len() {
        magnitude_spectrum[i] = f64::sqrt(f64::powf(spectrum[i].re, 2.0) + f64::powf(spectrum[i].im, 2.0));
        phase_spectrum[i] = f64::atan2(spectrum[i].im, spectrum[i].re);
    }
    (magnitude_spectrum, phase_spectrum)
}
