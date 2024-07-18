// File: analyzer.rs
// This file contains functionality for analyzing audio.

use super::spectral_analysis_tools::*;
use crate::spectrum;

/// Represents a spectral analysis of a FFT frame. Contains computed spectral features.
#[derive(Copy, Clone)]
pub struct Analysis {
    pub spectral_centroid: f64,
    pub spectral_variance: f64,
    pub spectral_skewness: f64,
    pub spectral_kurtosis: f64,
    pub spectral_entropy: f64,
    pub spectral_flatness: f64,
    pub spectral_roll_off_50: f64,
    pub spectral_roll_off_75: f64,
    pub spectral_roll_off_90: f64,
    pub spectral_roll_off_95: f64,
    pub spectral_slope: f64,
    pub spectral_slope_0_1_khz: f64,
    pub spectral_slope_1_5_khz: f64,
    pub spectral_slope_0_5_khz: f64,
}

/// Performs a suite of spectral analysis tools on a provided rFFT magnitude spectrum.
/// This function is more efficient than calculating the spectral features separately.
/// It returns an `Analysis` struct containing the analysis.
///
/// # Example
///
/// ```
/// use aus::{spectrum, analysis};
/// let fft_size = 2048;
/// let audio = aus::read("myaudio.wav");
/// let audio_chunk = audio.samples[:fft_size];
/// let imaginary_spectrum = spectrum::rfft(&audio_chunk, fft_size);
/// let (magnitude_spectrum, phase_spectrum) = spectrum::complex_to_polar_rfft(&imaginary_spectrum);
/// let audio_analysis = analysis::analyzer(&magnitude_spectrum, fft_size, audio.sample_rate);
/// ```
pub fn analyzer(magnitude_spectrum: &Vec<f64>, fft_size: usize, sample_rate: u32) -> Analysis {
    let power_spectrum = make_power_spectrum(&magnitude_spectrum);
    let magnitude_spectrum_sum = magnitude_spectrum.iter().sum();
    let power_spectrum_sum = power_spectrum.iter().sum();
    let spectrum_pmf = make_spectrum_pmf(&power_spectrum, power_spectrum_sum);
    let rfft_freqs = spectrum::rfftfreq(fft_size, sample_rate);
    let analysis_spectral_centroid = compute_spectral_centroid(&magnitude_spectrum, &rfft_freqs, magnitude_spectrum_sum);
    let analysis_spectral_variance = compute_spectral_variance(&spectrum_pmf, &rfft_freqs, analysis_spectral_centroid);
    let analysis_spectral_skewness = compute_spectral_skewness(&spectrum_pmf, &rfft_freqs, analysis_spectral_centroid, analysis_spectral_variance);
    let analysis_spectral_kurtosis = compute_spectral_kurtosis(&spectrum_pmf, &rfft_freqs, analysis_spectral_centroid, analysis_spectral_variance);
    let analysis_spectral_entropy = compute_spectral_entropy(&spectrum_pmf);
    let analysis_spectral_flatness = compute_spectral_flatness(&magnitude_spectrum, magnitude_spectrum_sum);
    let analysis_spectral_roll_off_50 = compute_spectral_roll_off_point(&power_spectrum, &rfft_freqs, power_spectrum_sum, 0.5);
    let analysis_spectral_roll_off_75 = compute_spectral_roll_off_point(&power_spectrum, &rfft_freqs, power_spectrum_sum, 0.75);
    let analysis_spectral_roll_off_90 = compute_spectral_roll_off_point(&power_spectrum, &rfft_freqs, power_spectrum_sum, 0.9);
    let analysis_spectral_roll_off_95 = compute_spectral_roll_off_point(&power_spectrum, &rfft_freqs, power_spectrum_sum, 0.95);
    let analysis_spectral_slope = compute_spectral_slope(&power_spectrum, power_spectrum_sum);

    // Eyben notes an author that recommends computing the slope of these spectral bands separately.
    let analysis_spectral_slope_0_1_khz = compute_spectral_slope_region(&power_spectrum, &rfft_freqs, 0.0, 1000.0, sample_rate);
    let analysis_spectral_slope_1_5_khz = compute_spectral_slope_region(&power_spectrum, &rfft_freqs, 1000.0, 5000.0, sample_rate);
    let analysis_spectral_slope_0_5_khz = compute_spectral_slope_region(&power_spectrum, &rfft_freqs, 0.0, 5000.0, sample_rate);
    
    let analysis = Analysis {
        spectral_centroid: analysis_spectral_centroid,
        spectral_entropy: analysis_spectral_entropy,
        spectral_flatness: analysis_spectral_flatness,
        spectral_kurtosis: analysis_spectral_kurtosis,
        spectral_roll_off_50: analysis_spectral_roll_off_50,
        spectral_roll_off_75: analysis_spectral_roll_off_75,
        spectral_roll_off_90: analysis_spectral_roll_off_90,
        spectral_roll_off_95: analysis_spectral_roll_off_95,
        spectral_skewness: analysis_spectral_skewness,
        spectral_slope: analysis_spectral_slope,
        spectral_slope_0_1_khz: analysis_spectral_slope_0_1_khz,
        spectral_slope_0_5_khz: analysis_spectral_slope_0_5_khz,
        spectral_slope_1_5_khz: analysis_spectral_slope_1_5_khz,
        spectral_variance: analysis_spectral_variance,
    };
    analysis
}

#[cfg(test)]
mod test {
    use super::*;
    
    #[test]
    /// Test analysis on an audio file.
    /// This test does not use multithreading, so it will probably take much longer.
    pub fn basic_tests4() {
        let fft_size: usize = 4096;
        let hop_size: usize = fft_size / 2;
        let window_type = crate::WindowType::Hamming;
        let path = String::from("D:\\Recording\\grains.wav");
        let mut audio = match crate::read(&path) {
            Ok(x) => x,
            Err(_) => panic!("could not read audio")
        };
        let stft_imaginary_spectrum: Vec<Vec<num::Complex<f64>>> = crate::spectrum::rstft(&mut audio.samples[0], fft_size, hop_size, window_type);
        let (stft_magnitude_spectrum, _) = crate::spectrum::complex_to_polar_rstft(&stft_imaginary_spectrum);
        let mut analyses: Vec<Analysis> = Vec::with_capacity(stft_magnitude_spectrum.len());
        for i in 0..stft_magnitude_spectrum.len() {
            analyses.push(analyzer(&stft_magnitude_spectrum[i], fft_size, audio.sample_rate));
        }
    }
}