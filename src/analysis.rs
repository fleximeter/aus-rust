// File: analysis.rs
// This file contains functionality for analyzing audio.

use realfft::RealFftPlanner;

/// Extracts the RMS energy of the signal
/// Reference: Eyben, pp. 21-22
pub fn energy(audio: &Vec<Vec<f64>>) -> f64 {
    let mut sumsquare: f64 = 0.0;
    for i in 0..audio.len() {
        for j in 0..audio[i].len() {
            sumsquare += audio[i][j].powf(2.0);
        }
    }
    if audio[0].len() < 1 {
        return 0.0;
    } else {
        return f64::sqrt(1.0 / audio[0].len() as f64 * sumsquare);
    }
}

/// Calculates the spectral centroid from provided magnitude spectrum.
/// It requires the sum of the magnitude spectrum as a parameter, since
/// this is a value that might be reused.
/// Reference: Eyben, pp. 39-40
pub fn spectral_centroid(magnitude_spectrum: &Vec<f64>, magnitude_freqs: &Vec<f64>, magnitude_spectrum_sum: f64) -> f64 {
    let mut sum: f64 = 0.0;
    for i in 0..magnitude_spectrum.len() {
        sum += magnitude_spectrum[i] * magnitude_freqs[i];
    }
    if magnitude_spectrum_sum == 0.0 {
        return 0.0;
    } else {
        return sum / magnitude_spectrum_sum;
    }
}

/// Calculates the spectral entropy.
/// It requires the power spectrum mass function (PMF).
/// Reference: Eyben, pp. 23, 40, 41
pub fn spectral_entropy(spectrum_pmf: &Vec<f64>) -> f64 {
    let mut entropy: f64 = 0.0;
    for i in 0..spectrum_pmf.len() {
        entropy += spectrum_pmf[i] * spectrum_pmf[i].log2();
    }
    -entropy
}

/// Calculates the spectral flatness.
/// It requires the power spectrum mass function (PMF).
/// Reference: Eyben, p. 39, https://en.wikipedia.org/wiki/Spectral_flatness
pub fn spectral_flatness(magnitude_spectrum: &Vec<f64>, magnitude_spectrum_sum: f64) -> f64 {
    let mut log_spectrum_sum: f64 = 0.0;
    for i in 0..magnitude_spectrum.len() {
        log_spectrum_sum += magnitude_spectrum[i].ln();
    }
    log_spectrum_sum /= magnitude_spectrum.len() as f64;
    f64::exp(log_spectrum_sum) * magnitude_spectrum.len() as f64 / magnitude_spectrum_sum
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

/// Calculates the zero crossing rate.
/// Reference: Eyben, p. 20
pub fn zero_crossing_rate(audio: &Vec<f64>, sample_rate: u16) -> f64 {
    let mut num_zc: f64 = 0.0;
    for i in 1..audio.len() {
        if audio[i-1] * audio[i] < 0.0 {
            num_zc += 1.0;
        } else if i < audio.len() - 1 && audio[i+1] < 0.0 && audio[i] == 0.0 {
            num_zc += 1.0;
        }
    }
    num_zc as f64 * sample_rate as f64 / audio.len() as f64
}
