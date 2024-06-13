// File: analysis.rs
// This file contains functionality for analyzing audio.

// Extracts the RMS energy of the signal
// :param audio: A NumPy array of audio samples
// :return: The RMS energy of the signal
// Reference: Eyben, pp. 21-22
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

// Calculates the spectral centroid from provided magnitude spectrum
// :param magnitude_spectrum: The magnitude spectrum
// :param magnitude_freqs: The magnitude frequencies
// :param magnitude_spectrum_sum: The sum of the magnitude spectrum
// :return: The spectral centroid
// Reference: Eyben, pp. 39-40
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

