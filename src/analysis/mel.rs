//! # Mel cepstrum
//! The `analysis::mel` module contains functionality for Mel spectrum and MFCC analysis.
//! 
//! To produce the Mel spectrum of a given magnitude spectrum, you need to run the `make_filterbanks` function to generate
//! the Mel filterbanks, then run the `filter_rfft_spectrum` function to generate the spectrum. To get the MFCCs, you
//! additionally run the `mfcc` function.

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

/// Computes the filterbanks for a Mel range and number of filters.
/// The filterbank contains the magnitude spectrum representation of each triangular filter.
/// You perform the filtering operation by computing the dot product of the real FFT
/// magnitude spectrum and the filter for each filter in the filterbank.
/// 
/// # Example
/// ```
/// use aus::{spectrum, analysis};
/// let fft_size = 2048;
/// let audio = aus::read("myfile.wav").unwrap();
/// let rfft_freqs = spectrum::rfftfreq(fft_size, audio.sample_rate);
/// let filterbanks = analysis::mel::make_filterbanks(analysis::mel::freq_to_mel(20.0), analysis::mel::freq_to_mel(8000.0), 40, fft_size, &rfft_freqs);
pub fn make_filterbanks(lower_mel: f64, upper_mel: f64, num_filters: usize, fft_size: usize, fft_freqs: &[f64]) -> Vec<Vec<f64>> {
    // Holds the real FFT frequency indices corresponding to the filter points
    let mut indices: Vec<usize> = vec![0; num_filters + 2];
    
    // Compute the FFT frequency indices for `num_filters + 1` points
    let slope = (upper_mel - lower_mel) / (num_filters as f64 + 1.0);
    for i in 0..indices.len() {
        // Get the frequency corresponding to this point in the filterbank
        let mel = lower_mel + slope * i as f64;
        let freq = mel_to_freq(mel);

        // Find the index of the closest real FFT frequency
        let idx = match util::ordered_search(&fft_freqs, freq) {
            Some(x) => x,
            None => 0
        };
        indices[i] = idx;
    }

    // Realize the filterbank
    let mut filterbanks: Vec<Vec<f64>> = Vec::new();
    for i in 0..num_filters {
        let mut filter: Vec<f64> = vec![0.0; fft_size / 2 + 1];
        filter[indices[i] + 1] = 1.0;
        for j in indices[i]..indices[i+1] {
            filter[j] = (j - indices[i]) as f64 / (indices[i+1] - indices[i]) as f64;
        }
        for j in indices[i+1]+1..indices[i+2] {
            filter[j] = (indices[i+2] - j) as f64 / (indices[i+2] - indices[i+1]) as f64;
        }
        filterbanks.push(filter);
    }

    filterbanks
}

/// Filters a real FFT spectrum with a filterbank. When used with Mel filterbanks, this function produces the Mel spectrum.
/// 
/// # Example
/// This example covers the entire process for making a Mel spectrum from a FFT frame.
/// ```
/// use aus::{spectrum, analysis};
/// let fft_size = 2048;
/// let audio = aus::read("myfile.wav").unwrap();
/// let rfft_freqs = spectrum::rfftfreq(fft_size, audio.sample_rate);
/// let audio_chunk = &audio.samples[0][..fft_size];
/// let imaginary_spectrum = spectrum::rfft(&audio_chunk, fft_size);
/// let (magnitude_spectrum, phase_spectrum) = spectrum::complex_to_polar_rfft(&imaginary_spectrum);
/// let power_spectrum = analysis::make_power_spectrum(&magnitude_spectrum);
/// let filterbanks = analysis::mel::make_filterbanks(analysis::mel::freq_to_mel(20.0), analysis::mel::freq_to_mel(8000.0), 40, fft_size, &rfft_freqs);
/// let spectrum = analysis::mel::filter_rfft_spectrum(&power_spectrum, &filterbanks);
/// ```
pub fn filter_rfft_spectrum(magnitude_spectrum: &[f64], filterbanks: &[Vec<f64>]) -> Vec<f64> {
    let mut filtered_spectrum: Vec<f64> = vec![0.0; filterbanks.len()];
    for i in 0..filterbanks.len() {
        filtered_spectrum[i] = util::dot_product(&magnitude_spectrum, &filterbanks[i]);
    }
    filtered_spectrum
}

/// Derives the Mel frequency cepstral coefficients (MFCCs) given a Mel spectrum.
/// Eyben's advice is to use a 20-8000Hz filterbank, a 26-band spectrum, and discard all MFCCs except 12-16. (Eyben, 60-61)
/// If you provide a `lifter` value greater than 0.0, liftering will be applied to the MFCCs
/// (this approach is borrowed from `librosa`: https://librosa.org/doc/main/generated/librosa.feature.mfcc.html).
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
/// let filterbanks = analysis::mel::make_filterbanks(analysis::mel::freq_to_mel(20.0), analysis::mel::freq_to_mel(8000.0), 26, fft_size, &rfft_freqs);
/// let mel_spectrum = analysis::mel::filter_rfft_spectrum(&magnitude_spectrum, &filterbanks);
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
