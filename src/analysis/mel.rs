//! # Mel cepstrum
//! The `analysis::mel` module contains functionality for Mel spectrum and MFCC analysis.

use crate::{spectrum, util};

/// Computes the Mel equivalent of a frequency in Hz.
/// 
/// $$
/// f^{(mel)}=2595 \log_{10}{\left(1+\frac{f}{700}\right)}
/// $$
#[inline]
pub fn freq_to_mel(freq: f64) -> f64 {
    2595.0 * f64::log10(1.0 + freq / 700.0)
}

/// Computes the frequency equivalent in Hz of a Mel.
/// 
/// $$
/// f = 700 \left(10^{\frac{f^{(mel)}}{2595}} - 1\right)
/// $$
#[inline]
pub fn mel_to_freq(mel: f64) -> f64 {
    700.0 * (f64::powf(10.0, mel / 2595.0) - 1.0)
}

/// Computes the Mel spectrum from a given magnitude or power spectrum. 
/// You need to specify the lower and upper Mel bounds. 
/// You also need to specify the number of filters (this determines the size of the resulting Mel spectrum).
/// 
/// # Example
/// ```
/// use aus::{spectrum, analysis};
/// let fft_size = 2048;
/// let audio = aus::read("myfile.wav").unwrap();
/// let audio_chunk = &audio.samples[0][..fft_size];
/// let rfft_freqs = spectrum::rfftfreq(fft_size, audio.sample_rate);
/// let imaginary_spectrum = spectrum::rfft(&audio_chunk, fft_size);
/// let (magnitude_spectrum, phase_spectrum) = spectrum::complex_to_polar_rfft(&imaginary_spectrum);
/// let power_spectrum = analysis::make_power_spectrum(&magnitude_spectrum);
/// let mel_spectrum = analysis::mel::make_mel_spectrum(&power_spectrum, analysis::mel::freq_to_mel(20.0), analysis::mel::freq_to_mel(8000.0), 40, &rfft_freqs);
pub fn make_mel_spectrum(spectrum: &[f64], lower_mel: f64, upper_mel: f64, num_filters: usize, fft_freqs: &[f64]) -> Vec<f64> {
    // Holds the real FFT frequency indices corresponding to the filter points
    let mut spectrum_indices_for_filters: Vec<usize> = vec![0; num_filters + 2];
    
    // Compute the FFT frequency indices for `num_filters + 1` points
    let slope = (upper_mel - lower_mel) / (num_filters as f64 + 1.0);
    for i in 0..spectrum_indices_for_filters.len() {
        // Get the frequency corresponding to this point in the filterbank
        let mel = lower_mel + slope * i as f64;
        let freq = mel_to_freq(mel);

        // Find the index of the closest real FFT frequency
        let idx = match util::ordered_search(&fft_freqs, freq) {
            Some(x) => x,
            None => 0
        };
        spectrum_indices_for_filters[i] = idx;
    }

    // Each filter contains the index range for which it is applied.
    let mut filterbanks: Vec<(Vec<f64>, usize, usize)> = Vec::new();

    // Realize the spectrum filterbank.
    // Each filter in the filterbank is a triangular filter that is evenly spaced on the Mel scale.
    // The filters are applied in the frequency domain.
    for i in 0..num_filters {
        // The filter size will be as small as possible (no extra zero samples to either side)
        let mut filter: Vec<f64> = vec![0.0; spectrum_indices_for_filters[i+2] - spectrum_indices_for_filters[i] + 1];
        
        // compute the left half of the filter
        for j in spectrum_indices_for_filters[i]..spectrum_indices_for_filters[i+1] {
            filter[j-spectrum_indices_for_filters[i]] = (j - spectrum_indices_for_filters[i]) as f64 / (spectrum_indices_for_filters[i+1] - spectrum_indices_for_filters[i]) as f64;
        }

        // the amplitude in the center of the filter is 1
        filter[spectrum_indices_for_filters[i+1] - spectrum_indices_for_filters[i] + 1] = 1.0;

        // compute the right half of the filter
        for j in spectrum_indices_for_filters[i+1]+1..spectrum_indices_for_filters[i+2] + 1 {
            filter[j-spectrum_indices_for_filters[i]] = (spectrum_indices_for_filters[i+2] - j) as f64 / (spectrum_indices_for_filters[i+2] - spectrum_indices_for_filters[i+1]) as f64;
        }

        // track the spectrum indices corresponding to the filter
        filterbanks.push((filter, spectrum_indices_for_filters[i], spectrum_indices_for_filters[i+2] + 1));
    }

    // Filter the spectrum
    let mut filtered_spectrum: Vec<f64> = vec![0.0; filterbanks.len()];
    for i in 0..filterbanks.len() {
        filtered_spectrum[i] = util::dot_product(&spectrum[filterbanks[i].1..filterbanks[i].2], &filterbanks[i].0);
    }
    filtered_spectrum
}


/// Derives the Mel frequency cepstral coefficients (MFCCs) given a Mel spectrum.
/// Eyben's advice is to use a 20-8000Hz filterbank, a 26-band spectrum, and discard all MFCCs except 12-16. (Eyben, 60-61)
/// 
/// If you provide a `lifter` value greater than 0.0, liftering will be applied to the MFCCs
/// (this approach is borrowed from `librosa`: <https://librosa.org/doc/main/generated/librosa.feature.mfcc.html>).
/// 
/// The MFCCs are derived by converting the Mel spectrum to a log Mel spectrum, then applying the Discrete Cosine Transform Type II.
/// Liftering is optional.
/// 
/// # Example
/// This example covers the entire process for calculating the MFCCs from a FFT frame.
/// ```
/// use aus::{spectrum, analysis};
/// let fft_size = 2048;
/// let audio = aus::read("myfile.wav").unwrap();
/// let rfft_freqs = spectrum::rfftfreq(fft_size, audio.sample_rate);
/// let audio_chunk = &audio.samples[0][..fft_size];
/// let imaginary_spectrum = spectrum::rfft(&audio_chunk, fft_size);
/// let (magnitude_spectrum, phase_spectrum) = spectrum::complex_to_polar_rfft(&imaginary_spectrum);
/// let power_spectrum = analysis::make_power_spectrum(&magnitude_spectrum);
/// let mel_spectrum = analysis::mel::make_mel_spectrum(&power_spectrum, analysis::mel::freq_to_mel(20.0), analysis::mel::freq_to_mel(8000.0), 26, &rfft_freqs);
/// let log_spectrum: Vec<f64> = analysis::make_log_spectrum(&mel_spectrum, 10e-8);
/// let mfccs = analysis::mel::mfcc(&log_spectrum, 2.0); // then use indices 11-15
/// ```
pub fn mfcc(log_spectrum: &[f64], lifter: f64) -> Vec<f64> {
    let mut mfccs = spectrum::dct2(&log_spectrum);
    // Perform "liftering"
    if lifter > 0.0 {
        for k in 0..mfccs.len() {
            mfccs[k] *= 1.0 + lifter / 2.0 * f64::sin(std::f64::consts::PI * k as f64 / lifter); 
        }
    }
    mfccs
}
