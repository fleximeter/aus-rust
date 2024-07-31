// File: spectral_analysis_tools.rs
// 
// This file contains functionality for computing spectral features.
// 
// Note that the "compute..." functions are for use with the "analyzer" function.
// These functions help to reduce the number of duplicate calculations needed 
// (such as computing the sum of the magnitude spectrum, etc.) There are separate 
// functions that allow these operations to be run individually. The compute functions
// are not exposed via the parent module.
//
// Many of the spectral features extracted here are based on the formulas provided in
// Florian Eyben, "Real-Time Speech and Music Classification by Large Audio Feature Space Extraction," Springer, 2016.

use num::Complex;
use rustfft::FftPlanner;
use crate::spectrum::SpectrumError;

/// Computes the autocorrelation of a signal of length `fft_size` using the FFT method described in Eyben, 45.
/// You must zero-pad the audio before calling this function if the `audio` length does not match the `fft_size`.
/// This autocorrelation method performs `N/2` zero-padding to the left and right as described in Eyben, 45.
/// The resulting `f64` vector contains only the computed values for `tau >= 0` and has length `fft_size`.
/// 
/// # Example
/// 
/// ```
/// use aus::{read, analysis::autocorrelation};
/// let fft_size: usize = 2048;
/// let audio = read("myfile.wav").unwrap();
/// let auto = autocorrelation(&audio.samples[0][..fft_size], fft_size).unwrap();
/// ```
pub fn autocorrelation(audio: &[f64], fft_size: usize) -> Result<Vec<f64>, SpectrumError> {
    if audio.len() != fft_size {
        return Err(SpectrumError{ error_msg: String::from("The audio length does not match the FFT size.")});
    }

    // prepare fft
    let padded_fft_size = fft_size * 2;
    let mut auto: Vec<f64> = vec![0.0; fft_size];
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(padded_fft_size);
    let ifft = planner.plan_fft_inverse(padded_fft_size);

    // prepare input audio vector
    let mut spectrum: Vec<Complex<f64>> = Vec::with_capacity(padded_fft_size);
    for _ in 0..fft_size / 2 {
        spectrum.push(Complex{re: 0.0, im: 0.0})
    }
    for i in 0..audio.len() {
        spectrum.push(Complex{re: audio[i], im: 0.0});
    }
    for _ in 0..fft_size / 2 {
        spectrum.push(Complex{re: 0.0, im: 0.0})
    }

    // compute autocorrelation
    fft.process(&mut spectrum);

    for i in 0..padded_fft_size {
        spectrum[i] = spectrum[i] * spectrum[i].conj();
    }

    ifft.process(&mut spectrum);

    // copy result into output vector
    for i in 0..fft_size {
        auto[i] = spectrum[i + fft_size].re;
    }
    
    Ok(auto)
}

/// Calculates the spectral centroid from provided magnitude spectrum.
/// It requires the sum of the magnitude spectrum as a parameter, since
/// this is a value that might be reused.
/// (Eyben, pp. 39-40)
/// 
/// This function is for efficient batch calculation, if you want to 
/// calculate all spectral features at once with the analyzer function.
pub fn compute_spectral_centroid(magnitude_spectrum: &Vec<f64>, rfft_freqs: &Vec<f64>, magnitude_spectrum_sum: f64) -> f64 {
    let mut sum: f64 = 0.0;
    for i in 0..magnitude_spectrum.len() {
        sum += magnitude_spectrum[i] * rfft_freqs[i];
    }
    return sum / magnitude_spectrum_sum;
}

/// Calculates the spectral entropy.
/// It requires the power spectrum mass function (PMF).
/// (Reference: Eyben, pp. 23, 40, 41)
/// 
/// This function is for efficient batch calculation, if you want to 
/// calculate all spectral features at once with the analyzer function.
pub fn compute_spectral_entropy(spectrum_pmf: &Vec<f64>) -> f64 {
    let mut entropy: f64 = 0.0;
    for i in 0..spectrum_pmf.len() {
        entropy += spectrum_pmf[i] * spectrum_pmf[i].log2();
    }
    -entropy
}

/// Calculates the spectral flatness.
/// It requires the power spectrum mass function (PMF).
/// (Eyben, p. 39, https://en.wikipedia.org/wiki/Spectral_flatness)
/// 
/// This function is for efficient batch calculation, if you want to 
/// calculate all spectral features at once with the analyzer function.
pub fn compute_spectral_flatness(magnitude_spectrum: &Vec<f64>, magnitude_spectrum_sum: f64) -> f64 {
    let mut log_spectrum_sum: f64 = 0.0;
    for i in 0..magnitude_spectrum.len() {
        log_spectrum_sum += magnitude_spectrum[i].ln();
    }
    log_spectrum_sum /= magnitude_spectrum.len() as f64;
    f64::exp(log_spectrum_sum) * magnitude_spectrum.len() as f64 / magnitude_spectrum_sum
}

/// Calculates the spectral kurtosis.
/// Requires the spectrum power mass function (PMF), RFFT magnitude frequencies, and spectral centroid
/// (Eyben, pp. 23, 39-40)
/// 
/// This function is for efficient batch calculation, if you want to 
/// calculate all spectral features at once with the analyzer function.
pub fn compute_spectral_kurtosis(spectrum_pmf: &Vec<f64>, rfft_freqs: &Vec<f64>, spectral_centroid: f64, spectral_variance: f64) -> f64 {
    let mut spectral_kurtosis: f64 = 0.0;
    for i in 0..spectrum_pmf.len() {
        spectral_kurtosis += f64::powf(rfft_freqs[i] - spectral_centroid, 4.0) * spectrum_pmf[i];
    }
    spectral_kurtosis /= spectral_variance.powf(2.0);
    spectral_kurtosis
}

/// Calculates the spectral roll off frequency from provided power spectrum.
/// The parameter `n` (0.0 <= n <= 1.00) indicates the roll-off point we wish to calculate .
/// (Eyben, p. 41)
/// 
/// This function is for efficient batch calculation, if you want to 
/// calculate all spectral features at once with the analyzer function.
pub fn compute_spectral_roll_off_point(power_spectrum: &Vec<f64>, rfft_freqs: &Vec<f64>, power_spectrum_sum: f64, n: f64) -> f64 {
    let mut i: i64 = -1;
    let mut cumulative_energy = 0.0;
    while cumulative_energy < n && i < rfft_freqs.len() as i64 - 1 {
        i += 1;
        cumulative_energy += power_spectrum[i as usize] / power_spectrum_sum;
    }
    rfft_freqs[i as usize]
}

/// Calculates the spectral skewness. 
/// Requires the spectrum power mass function (PMF), RFFT magnitude frequencies, and spectral centroid.
/// (Eyben, pp. 23, 39-40)
/// 
/// This function is for efficient batch calculation, if you want to 
/// calculate all spectral features at once with the analyzer function.
pub fn compute_spectral_skewness(spectrum_pmf: &Vec<f64>, rfft_freqs: &Vec<f64>, spectral_centroid: f64, spectral_variance: f64) -> f64 {
    let mut spectral_skewness: f64 = 0.0;
    for i in 0..spectrum_pmf.len() {
        spectral_skewness += f64::powf(rfft_freqs[i] - spectral_centroid, 3.0) * spectrum_pmf[i];
    }
    spectral_skewness /= spectral_variance.powf(1.5);
    spectral_skewness
}

/// Calculates the spectral slope from provided power spectrum.
/// (Eyben, pp. 35-38)
/// 
/// This function is for efficient batch calculation, if you want to 
/// calculate all spectral features at once with the analyzer function.
pub fn compute_spectral_slope(power_spectrum: &Vec<f64>, power_spectrum_sum: f64) -> f64 {
    let n = power_spectrum.len() as f64;

    // power spectrum will be the Y vector; we need to make an X vector
    let mut x: Vec<f64> = vec![0.0; power_spectrum.len()];
    for i in 0..power_spectrum.len() {
        x[i] = i as f64;
    }
    
    // the sum of x, and sum of x^2
    let sum_x = n * (n - 1.0) / 2.0;
    let sum_x_2 = n * (n - 1.0) * (2.0 * n - 1.0) / 6.0;

    let slope = (n * dot_product(&power_spectrum, &x) - sum_x * power_spectrum_sum) / (n * sum_x_2 - sum_x.powf(2.0));
    slope
}

/// Calculates the spectral slope from provided power spectrum, between the frequencies
/// specified. The frequencies specified do not have to correspond to exact bin indices.
/// (Eyben, pp. 35-38)
/// 
/// This function is for efficient batch calculation, if you want to 
/// calculate all spectral features at once with the analyzer function.
pub fn compute_spectral_slope_region(power_spectrum: &Vec<f64>, rfft_freqs: &Vec<f64>, f_lower: f64, f_upper: f64, sample_rate: u32) -> f64 {
    let fundamental_freq = sample_rate as f64 / ((power_spectrum.len() - 1) as f64 * 2.0);
    
    // The approximate bin indices for the lower and upper frequencies specified
    let m_fl = f_lower / fundamental_freq;
    let m_fu = f_upper / fundamental_freq;
    
    let n = power_spectrum.len() as f64;
    let m_fl_ceil = m_fl.ceil() as usize;
    let m_fl_floor = m_fl.floor() as usize;
    let m_fu_ceil = m_fu.ceil() as usize;
    let m_fu_floor = m_fu.floor() as usize;
    
    // these complicated formulas come from Eyben, p.37. The idea
    // is to use interpolation in case the lower and upper frequencies
    // do not correspond to exact bin indices.
    
    // calculate the sum_x
    let sum_x: f64 = f_lower + rfft_freqs[m_fl_ceil..m_fu_floor].iter().sum::<f64>() as f64 + f_upper;

    // calculate the sum_y
    let sum_y = power_spectrum[m_fl_floor] + (m_fl - m_fl_floor as f64) * 
        (power_spectrum[m_fl_ceil] - power_spectrum[m_fl_floor]) + 
        power_spectrum[m_fl_ceil..m_fu_floor].iter().sum::<f64>() + 
        power_spectrum[m_fu_floor] + (m_fu - m_fu_floor as f64) * 
        (power_spectrum[m_fu_ceil] - power_spectrum[m_fu_floor]);
    
    // calculate sum_x^2
    let sum_x_2 = f_lower.powf(2.0) + 
        dot_product(&rfft_freqs[m_fl_ceil..m_fu_floor], 
            &rfft_freqs[m_fl_ceil..m_fu_floor]) + 
        f_upper.powf(2.0);

    // calculate sum_xy
    let sum_xy = f_lower * (power_spectrum[m_fl_floor] + (m_fl - m_fl_floor as f64) * (power_spectrum[m_fl_ceil] - power_spectrum[m_fl_floor])) +
        dot_product(&power_spectrum[m_fl_ceil..m_fu_floor], &rfft_freqs[m_fl_ceil..m_fu_floor]) +
        f_upper * (power_spectrum[m_fu_floor] + (m_fu - m_fu_floor as f64) * (power_spectrum[m_fu_ceil] - power_spectrum[m_fu_floor]));

    let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x_2 - sum_x.powf(2.0));
    slope
}

/// Calculates the spectral variance.
/// Requires the spectrum power mass function (PMF), RFFT magnitude frequencies, and spectral centroid
/// (Eyben, pp. 23, 39-40)
/// 
/// This function is for efficient batch calculation, if you want to 
/// calculate all spectral features at once with the analyzer function.
pub fn compute_spectral_variance(spectrum_pmf: &Vec<f64>, rfft_freqs: &Vec<f64>, spectral_centroid: f64) -> f64 {
    let mut spectral_variance: f64 = 0.0;
    for i in 0..spectrum_pmf.len() {
        spectral_variance += f64::powf(rfft_freqs[i] - spectral_centroid, 2.0) * spectrum_pmf[i];
    }
    spectral_variance
}

/// Simple dot product function, implemented for code readability rather than using zip(), etc.
/// No vector length checks are performed - make sure that both vectors have the same length before
/// calling this function.
/// 
/// # Panics
/// This function will panic if `vec2` is shorter than `vec1`.
#[inline]
pub fn dot_product(vec1: &[f64], vec2: &[f64]) -> f64 {
    let mut sum = 0.0;
    for i in 0..vec1.len() {
        sum += vec1[i] * vec2[i];
    }
    sum
}

/// Creates a power spectrum based on a provided magnitude spectrum.
/// 
/// # Example
/// 
/// ```
/// use aus::{spectrum, analysis};
/// let fft_size = 2048;
/// let audio = aus::read("myaudio.wav").unwrap();
/// let imaginary_spectrum = spectrum::rfft(&audio.samples[0][..fft_size], fft_size);
/// let (magnitude_spectrum, phase_spectrum) = spectrum::complex_to_polar_rfft(&imaginary_spectrum);
/// let power_spectrum = analysis::make_power_spectrum(&magnitude_spectrum);
/// ```
pub fn make_power_spectrum(magnitude_spectrum: &Vec<f64>) -> Vec<f64> {
    let mut power_spec: Vec<f64> = vec![0.0; magnitude_spectrum.len()];
    for i in 0..magnitude_spectrum.len() {
        power_spec[i] = magnitude_spectrum[i].powf(2.0);
    }
    power_spec
}

/// Generates the spectrum power mass function (PMF) based on provided power spectrum 
/// and sum of power spectrum.
/// (Eyben, p. 40)
pub fn make_spectrum_pmf(power_spectrum: &Vec<f64>, power_spectrum_sum: f64) -> Vec<f64> {
    let mut pmf_vector: Vec<f64> = vec![0.0; power_spectrum.len()];
    for i in 0..power_spectrum.len() {
        pmf_vector[i] = power_spectrum[i] / power_spectrum_sum;
    }
    pmf_vector
}

/// Calculates the spectral centroid from provided magnitude spectrum.
/// (Eyben, pp. 39-40)
///
/// # Example
///
/// ```
/// use aus::{spectrum, analysis};
/// let fft_size = 2048;
/// let audio = aus::read("myaudio.wav").unwrap();
/// let imaginary_spectrum = spectrum::rfft(&audio.samples[0][..fft_size], fft_size);
/// let (magnitude_spectrum, phase_spectrum) = spectrum::complex_to_polar_rfft(&imaginary_spectrum);
/// let freqs = spectrum::rfftfreq(fft_size, audio.sample_rate);
/// let centroid = analysis::spectral_centroid(&magnitude_spectrum, &freqs);
/// ```
pub fn spectral_centroid(magnitude_spectrum: &Vec<f64>, rfft_freqs: &Vec<f64>) -> f64 {
    let magnitude_spectrum_sum: f64 = magnitude_spectrum.iter().sum();
    compute_spectral_centroid(magnitude_spectrum, rfft_freqs, magnitude_spectrum_sum)
}

/// Calculates the spectral entropy from provided magnitude spectrum.
/// (Eyben, pp. 23, 40, 41)
///
/// # Example
///
/// ```
/// use aus::{spectrum, analysis};
/// let fft_size = 2048;
/// let audio = aus::read("myaudio.wav").unwrap();
/// let imaginary_spectrum = spectrum::rfft(&audio.samples[0][..fft_size], fft_size);
/// let (magnitude_spectrum, phase_spectrum) = spectrum::complex_to_polar_rfft(&imaginary_spectrum);
/// let entropy = analysis::spectral_entropy(&magnitude_spectrum);
/// ```
pub fn spectral_entropy(magnitude_spectrum: &Vec<f64>) -> f64 {
    let power_spectrum = make_power_spectrum(magnitude_spectrum);
    let spectrum_pmf = make_spectrum_pmf(&power_spectrum, power_spectrum.iter().sum());
    compute_spectral_entropy(&spectrum_pmf)
}

/// Calculates the spectral flatness from provided magnitude spectrum.
/// (Eyben, p. 39, https://en.wikipedia.org/wiki/Spectral_flatness)
///
/// # Example
///
/// ```
/// use aus::{spectrum, analysis};
/// let fft_size = 2048;
/// let audio = aus::read("myaudio.wav").unwrap();
/// let imaginary_spectrum = spectrum::rfft(&audio.samples[0][..fft_size], fft_size);
/// let (magnitude_spectrum, phase_spectrum) = spectrum::complex_to_polar_rfft(&imaginary_spectrum);
/// let flatness = analysis::spectral_flatness(&magnitude_spectrum);
/// ```
pub fn spectral_flatness(magnitude_spectrum: &Vec<f64>) -> f64 {
    let magnitude_spectrum_sum: f64 = magnitude_spectrum.iter().sum();
    compute_spectral_flatness(magnitude_spectrum, magnitude_spectrum_sum)
}

/// Calculates the spectral kurtosis from provided magnitude spectrum and real FFT frequency list.
/// (Eyben, pp. 23, 39-40)
///
/// # Example
///
/// ```
/// use aus::{spectrum, analysis};
/// let fft_size = 2048;
/// let audio = aus::read("myaudio.wav").unwrap();
/// let imaginary_spectrum = spectrum::rfft(&audio.samples[0][..fft_size], fft_size);
/// let (magnitude_spectrum, phase_spectrum) = spectrum::complex_to_polar_rfft(&imaginary_spectrum);
/// let freqs = spectrum::rfftfreq(fft_size, audio.sample_rate);
/// let kurtosis = analysis::spectral_kurtosis(&magnitude_spectrum, &freqs);
/// ```
pub fn spectral_kurtosis(magnitude_spectrum: &Vec<f64>, rfft_freqs: &Vec<f64>) -> f64 {
    let power_spectrum = make_power_spectrum(magnitude_spectrum);
    let spectrum_pmf = make_spectrum_pmf(&power_spectrum, power_spectrum.iter().sum());
    let spectral_centroid = compute_spectral_centroid(magnitude_spectrum, rfft_freqs, magnitude_spectrum.iter().sum());
    let spectral_variance = compute_spectral_variance(&spectrum_pmf, rfft_freqs, spectral_centroid);
    compute_spectral_kurtosis(&spectrum_pmf, rfft_freqs, spectral_centroid, spectral_variance)
}

/// Calculates the spectral roll off frequency from provided magnitude spectrum, real FFT frequency list, and roll-off point.
/// The parameter `n` (0.0 <= n <= 1.00) indicates the roll-off point we wish to calculate.
/// (Eyben, p. 41)
///
/// # Example
///
/// ```
/// use aus::{spectrum, analysis};
/// let fft_size = 2048;
/// let audio = aus::read("myaudio.wav").unwrap();
/// let imaginary_spectrum = spectrum::rfft(&audio.samples[0][..fft_size], fft_size);
/// let (magnitude_spectrum, phase_spectrum) = spectrum::complex_to_polar_rfft(&imaginary_spectrum);
/// let freqs = spectrum::rfftfreq(fft_size, audio.sample_rate);
/// let roll_off = analysis::spectral_roll_off_point(&magnitude_spectrum, &freqs, 0.75);
/// ```
pub fn spectral_roll_off_point(magnitude_spectrum: &Vec<f64>, rfft_freqs: &Vec<f64>, n: f64) -> f64 {
    let power_spectrum = make_power_spectrum(magnitude_spectrum);
    let power_spectrum_sum: f64 = power_spectrum.iter().sum();
    compute_spectral_roll_off_point(&power_spectrum, rfft_freqs, power_spectrum_sum, n)
}

/// Calculates the spectral skewness from provided magnitude spectrum and real FFT frequency list.
/// (Eyben, pp. 23, 39-40)
///
/// # Example
///
/// ```
/// use aus::{spectrum, analysis};
/// let fft_size = 2048;
/// let audio = aus::read("myaudio.wav").unwrap();
/// let imaginary_spectrum = spectrum::rfft(&audio.samples[0][..fft_size], fft_size);
/// let (magnitude_spectrum, phase_spectrum) = spectrum::complex_to_polar_rfft(&imaginary_spectrum);
/// let freqs = spectrum::rfftfreq(fft_size, audio.sample_rate);
/// let skewness = analysis::spectral_skewness(&magnitude_spectrum, &freqs);
/// ```
pub fn spectral_skewness(magnitude_spectrum: &Vec<f64>, rfft_freqs: &Vec<f64>) -> f64 {
    let power_spectrum = make_power_spectrum(magnitude_spectrum);
    let spectrum_pmf = make_spectrum_pmf(&power_spectrum, power_spectrum.iter().sum());
    let spectral_centroid = compute_spectral_centroid(magnitude_spectrum, rfft_freqs, magnitude_spectrum.iter().sum());
    let spectral_variance = compute_spectral_variance(&spectrum_pmf, rfft_freqs, spectral_centroid);
    compute_spectral_skewness(&spectrum_pmf, rfft_freqs, spectral_centroid, spectral_variance)
}

/// Calculates the spectral slope from provided magnitude spectrum.
/// (Eyben, pp. 35-38)
///
/// # Example
///
/// ```
/// use aus::{spectrum, analysis};
/// let fft_size = 2048;
/// let audio = aus::read("myaudio.wav").unwrap();
/// let imaginary_spectrum = spectrum::rfft(&audio.samples[0][..fft_size], fft_size);
/// let (magnitude_spectrum, phase_spectrum) = spectrum::complex_to_polar_rfft(&imaginary_spectrum);
/// let slope = analysis::spectral_slope(&magnitude_spectrum);
/// ```
pub fn spectral_slope(magnitude_spectrum: &Vec<f64>) -> f64 {
    let power_spectrum = make_power_spectrum(magnitude_spectrum);
    let power_spectrum_sum: f64 = power_spectrum.iter().sum();
    compute_spectral_slope(&power_spectrum, power_spectrum_sum)
}

/// Calculates the spectral slope from provided magnitude spectrum, between the frequencies
/// specified. The frequencies specified do not have to align with the frequencies in the real FFT frequency list.
/// (Eyben, pp. 35-38)
///
/// # Example
///
/// ```
/// use aus::{spectrum, analysis};
/// let fft_size = 2048;
/// let audio = aus::read("myaudio.wav").unwrap();
/// let imaginary_spectrum = spectrum::rfft(&audio.samples[0][..fft_size], fft_size);
/// let (magnitude_spectrum, phase_spectrum) = spectrum::complex_to_polar_rfft(&imaginary_spectrum);
/// let freqs = spectrum::rfftfreq(fft_size, audio.sample_rate);
/// let slope = analysis::spectral_slope_region(&magnitude_spectrum, &freqs, 105.0, 852.0, audio.sample_rate);
/// ```
pub fn spectral_slope_region(magnitude_spectrum: &Vec<f64>, rfft_freqs: &Vec<f64>, f_lower: f64, f_upper: f64, sample_rate: u32) -> f64 {
    let power_spectrum = make_power_spectrum(magnitude_spectrum);
    compute_spectral_slope_region(&power_spectrum, rfft_freqs, f_lower, f_upper, sample_rate)
}

/// Calculates the spectral variance from provided magnitude spectrum and real FFT frequency list.
/// (Eyben, pp. 23, 39-40)
///
/// # Example
///
/// ```
/// use aus::{spectrum, analysis};
/// let fft_size = 2048;
/// let audio = aus::read("myaudio.wav").unwrap();
/// let imaginary_spectrum = spectrum::rfft(&audio.samples[0][..fft_size], fft_size);
/// let (magnitude_spectrum, phase_spectrum) = spectrum::complex_to_polar_rfft(&imaginary_spectrum);
/// let freqs = spectrum::rfftfreq(fft_size, audio.sample_rate);
/// let variance = analysis::spectral_variance(&magnitude_spectrum, &freqs);
/// ```
pub fn spectral_variance(magnitude_spectrum: &Vec<f64>, rfft_freqs: &Vec<f64>) -> f64 {
    let power_spectrum = make_power_spectrum(magnitude_spectrum);
    let spectrum_pmf = make_spectrum_pmf(&power_spectrum, power_spectrum.iter().sum());
    let spectral_centroid = compute_spectral_centroid(magnitude_spectrum, rfft_freqs, magnitude_spectrum.iter().sum());
    compute_spectral_variance(&spectrum_pmf, rfft_freqs, spectral_centroid)
}

#[cfg(test)]
mod test {
    use super::*;
    use std::io;
    use std::io::Write;

    const AUDIO: &str = "myfile.wav";
    const FFT_SIZE: usize = 2048;
    
    #[test]
    fn test_autocorrelation() {
        let audio = crate::read(AUDIO).unwrap();
        let auto = autocorrelation(&audio.samples[0][44100..44100+FFT_SIZE], FFT_SIZE).unwrap();
        
        for val in auto.iter() {
            print!("{} ", val);
            io::stdout().flush().unwrap();
        }

        let mut max = auto[0];
        let mut max_idx: usize = 0;
        for i in 0..auto.len() {
            if auto[i] > max {
                max = auto[i];
                max_idx = i;
            }
        }

        println!("\nMax: {}, max_idx: {}", max, max_idx);
    }
}