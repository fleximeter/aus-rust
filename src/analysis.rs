/// File: analysis.rs
/// This file contains functionality for analyzing audio.
///
/// Note that the private "compute..." functions are for use with the "analyzer" function.
/// These private functions help to reduce the number of duplicate calculations needed 
/// (such as computing the sum of the magnitude spectrum, etc.) There are separate public
/// functions that allow these operations to be run individually.
///
/// Many of the spectral features extracted here are based on the formulas provided in
/// Florian Eyben, "Real-Time Speech and Music Classification by Large Audio Feature Space Extraction," Springer, 2016.

use crate::spectrum;

// The lowest f64 value for which dBFS can be computed.
// All lower values will result in f64::NEG_ININITY.
// Note that a value of 1e-20 corresponds to a dBFS of -400.0.
const DBFS_EPSILON: f64 = 1e-20;

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
pub fn analyzer(magnitude_spectrum: &Vec<f64>, fft_size: usize, sample_rate: u16) -> Analysis {
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

/// Calculates the spectral centroid from provided magnitude spectrum.
/// It requires the sum of the magnitude spectrum as a parameter, since
/// this is a value that might be reused.
/// Reference: Eyben, pp. 39-40
/// 
/// This function is for efficient batch calculation, if you want to 
/// calculate all spectral features at once with the analyzer function.
fn compute_spectral_centroid(magnitude_spectrum: &Vec<f64>, rfft_freqs: &Vec<f64>, magnitude_spectrum_sum: f64) -> f64 {
    let mut sum: f64 = 0.0;
    for i in 0..magnitude_spectrum.len() {
        sum += magnitude_spectrum[i] * rfft_freqs[i];
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
/// 
/// This function is for efficient batch calculation, if you want to 
/// calculate all spectral features at once with the analyzer function.
fn compute_spectral_entropy(spectrum_pmf: &Vec<f64>) -> f64 {
    let mut entropy: f64 = 0.0;
    for i in 0..spectrum_pmf.len() {
        entropy += spectrum_pmf[i] * spectrum_pmf[i].log2();
    }
    -entropy
}

/// Calculates the spectral flatness.
/// It requires the power spectrum mass function (PMF).
/// Reference: Eyben, p. 39, https://en.wikipedia.org/wiki/Spectral_flatness
/// 
/// This function is for efficient batch calculation, if you want to 
/// calculate all spectral features at once with the analyzer function.
fn compute_spectral_flatness(magnitude_spectrum: &Vec<f64>, magnitude_spectrum_sum: f64) -> f64 {
    let mut log_spectrum_sum: f64 = 0.0;
    for i in 0..magnitude_spectrum.len() {
        log_spectrum_sum += magnitude_spectrum[i].ln();
    }
    log_spectrum_sum /= magnitude_spectrum.len() as f64;
    f64::exp(log_spectrum_sum) * magnitude_spectrum.len() as f64 / magnitude_spectrum_sum
}

/// Calculates the spectral kurtosis
/// 
/// Requires the spectrum power mass function (PMF), RFFT magnitude frequencies, and spectral centroid
/// Reference: Eyben, pp. 23, 39-40
/// 
/// This function is for efficient batch calculation, if you want to 
/// calculate all spectral features at once with the analyzer function.
fn compute_spectral_kurtosis(spectrum_pmf: &Vec<f64>, rfft_freqs: &Vec<f64>, spectral_centroid: f64, spectral_variance: f64) -> f64 {
    let mut spectral_kurtosis: f64 = 0.0;
    for i in 0..spectrum_pmf.len() {
        spectral_kurtosis += f64::powf(rfft_freqs[i] - spectral_centroid, 4.0) * spectrum_pmf[i];
    }
    spectral_kurtosis /= spectral_variance.powf(2.0);
    spectral_kurtosis
}

/// Calculates the spectral roll off frequency from provided power spectrum
/// The parameter n (0.0 <= n <= 1.00) indicates the roll-off point we wish to calculate 
/// Reference: Eyben, p. 41
/// 
/// This function is for efficient batch calculation, if you want to 
/// calculate all spectral features at once with the analyzer function.
fn compute_spectral_roll_off_point(power_spectrum: &Vec<f64>, rfft_freqs: &Vec<f64>, power_spectrum_sum: f64, n: f64) -> f64 {
    let mut i: i64 = -1;
    let mut cumulative_energy = 0.0;
    while cumulative_energy < n && i < rfft_freqs.len() as i64 - 1 {
        i += 1;
        cumulative_energy += power_spectrum[i as usize] / power_spectrum_sum;
    }
    rfft_freqs[i as usize]
}

/// Calculates the spectral skewness
/// 
/// Requires the spectrum power mass function (PMF), RFFT magnitude frequencies, and spectral centroid
/// Reference: Eyben, pp. 23, 39-40
/// 
/// This function is for efficient batch calculation, if you want to 
/// calculate all spectral features at once with the analyzer function.
fn compute_spectral_skewness(spectrum_pmf: &Vec<f64>, rfft_freqs: &Vec<f64>, spectral_centroid: f64, spectral_variance: f64) -> f64 {
    let mut spectral_skewness: f64 = 0.0;
    for i in 0..spectrum_pmf.len() {
        spectral_skewness += f64::powf(rfft_freqs[i] - spectral_centroid, 3.0) * spectrum_pmf[i];
    }
    spectral_skewness /= spectral_variance.powf(1.5);
    spectral_skewness
}

/// Calculates the spectral slope from provided power spectrum.
/// Reference: Eyben, pp. 35-38
/// 
/// This function is for efficient batch calculation, if you want to 
/// calculate all spectral features at once with the analyzer function.
fn compute_spectral_slope(power_spectrum: &Vec<f64>, power_spectrum_sum: f64) -> f64 {
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
/// Reference: Eyben, pp. 35-38
/// 
/// This function is for efficient batch calculation, if you want to 
/// calculate all spectral features at once with the analyzer function.
fn compute_spectral_slope_region(power_spectrum: &Vec<f64>, rfft_freqs: &Vec<f64>, f_lower: f64, f_upper: f64, sample_rate: u16) -> f64 {
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

/// Calculates the spectral variance
/// Requires the spectrum power mass function (PMF), RFFT magnitude frequencies, and spectral centroid
/// Reference: Eyben, pp. 23, 39-40
/// 
/// This function is for efficient batch calculation, if you want to 
/// calculate all spectral features at once with the analyzer function.
fn compute_spectral_variance(spectrum_pmf: &Vec<f64>, rfft_freqs: &Vec<f64>, spectral_centroid: f64) -> f64 {
    let mut spectral_variance: f64 = 0.0;
    for i in 0..spectrum_pmf.len() {
        spectral_variance += f64::powf(rfft_freqs[i] - spectral_centroid, 2.0) * spectrum_pmf[i];
    }
    spectral_variance
}


/// Calculates the DC bias of the signal
#[inline(always)]
pub fn dc_bias(audio: &[f64]) -> f64 {
    let mut sum = 0.0;
    for i in 0..audio.len() {
        sum += audio[i];
    }
    sum / audio.len() as f64
}

/// Safely calculates dBFS, handling very small values accurately
#[inline(always)]
pub fn dbfs(val: f64) -> f64 {
    if val.abs() < DBFS_EPSILON {
        f64::NEG_INFINITY
    } else {
        20.0 * val.log10()
    }
}

/// Calculates the max dbfs in a list of audio samples
pub fn dbfs_max(audio: &[f64]) -> f64 {
    let mut maxval = 0.0;
    for i in 0..audio.len() {
        let sample_abs = audio[i].abs();
        if sample_abs > maxval {
            maxval = sample_abs;
        }
    }
    dbfs(maxval)
}

/// Simple dot product function, implemented for code readability rather than using zip(), etc.
#[inline]
fn dot_product(vec1: &[f64], vec2: &[f64]) -> f64 {
    let mut sum = 0.0;
    for i in 0..vec1.len() {
        sum += vec1[i] * vec2[i];
    }
    sum
}

/// Extracts the RMS energy of the signal
/// Reference: Eyben, pp. 21-22
pub fn energy(audio: &[f64]) -> f64 {
    let mut sumsquare: f64 = 0.0;
    for i in 0..audio.len() {
        sumsquare += audio[i].powf(2.0);
    }
    if audio.len() < 1 {
        return 0.0;
    } else {
        return f64::sqrt(1.0 / audio.len() as f64 * sumsquare);
    }
}

/// Creates a power spectrum based on a provided magnitude spectrum
fn make_power_spectrum(magnitude_spectrum: &Vec<f64>) -> Vec<f64> {
    let mut power_spec: Vec<f64> = vec![0.0; magnitude_spectrum.len()];
    for i in 0..magnitude_spectrum.len() {
        power_spec[i] = magnitude_spectrum[i].powf(2.0);
    }
    power_spec
}

/// Generates the spectrum power mass function (PMF) based on provided power spectrum 
/// and sum of power spectrum
fn make_spectrum_pmf(power_spectrum: &Vec<f64>, power_spectrum_sum: f64) -> Vec<f64> {
    let mut pmf_vector: Vec<f64> = vec![0.0; power_spectrum.len()];
    for i in 0..power_spectrum.len() {
        pmf_vector[i] = power_spectrum[i] / power_spectrum_sum;
    }
    pmf_vector
}

/// Calculates the spectral centroid from provided magnitude spectrum
/// Reference: Eyben, pp. 39-40
pub fn spectral_centroid(magnitude_spectrum: &Vec<f64>, rfft_freqs: &Vec<f64>) -> f64 {
    let mut sum: f64 = 0.0;
    let magnitude_spectrum_sum: f64 = magnitude_spectrum.iter().sum();
    compute_spectral_centroid(magnitude_spectrum, rfft_freqs, magnitude_spectrum_sum)
}

/// Calculates the spectral entropy.
/// It requires the power spectrum mass function (PMF).
/// Reference: Eyben, pp. 23, 40, 41
pub fn spectral_entropy(magnitude_spectrum: &Vec<f64>) -> f64 {
    let power_spectrum = make_power_spectrum(magnitude_spectrum);
    let spectrum_pmf = make_spectrum_pmf(&power_spectrum, power_spectrum.iter().sum());
    compute_spectral_entropy(&spectrum_pmf)
}

/// Calculates the spectral flatness.
/// It requires the power spectrum mass function (PMF).
/// Reference: Eyben, p. 39, https://en.wikipedia.org/wiki/Spectral_flatness
pub fn spectral_flatness(magnitude_spectrum: &Vec<f64>) -> f64 {
    let magnitude_spectrum_sum: f64 = magnitude_spectrum.iter().sum();
    compute_spectral_flatness(magnitude_spectrum, magnitude_spectrum_sum)
}

/// Calculates the spectral kurtosis
/// 
/// Requires the spectrum power mass function (PMF), RFFT magnitude frequencies, and spectral centroid
/// Reference: Eyben, pp. 23, 39-40
pub fn spectral_kurtosis(magnitude_spectrum: &Vec<f64>, rfft_freqs: &Vec<f64>) -> f64 {
    let power_spectrum = make_power_spectrum(magnitude_spectrum);
    let spectrum_pmf = make_spectrum_pmf(&power_spectrum, power_spectrum.iter().sum());
    let spectral_centroid = compute_spectral_centroid(magnitude_spectrum, rfft_freqs, magnitude_spectrum.iter().sum());
    let spectral_variance = compute_spectral_variance(&spectrum_pmf, rfft_freqs, spectral_centroid);
    compute_spectral_kurtosis(&spectrum_pmf, rfft_freqs, spectral_centroid, spectral_variance)
}

/// Calculates the spectral roll off frequency from provided power spectrum
/// The parameter n (0.0 <= n <= 1.00) indicates the roll-off point we wish to calculate 
/// Reference: Eyben, p. 41
pub fn spectral_roll_off_point(magnitude_spectrum: &Vec<f64>, rfft_freqs: &Vec<f64>, n: f64) -> f64 {
    let power_spectrum = make_power_spectrum(magnitude_spectrum);
    let power_spectrum_sum: f64 = power_spectrum.iter().sum();
    compute_spectral_roll_off_point(&power_spectrum, rfft_freqs, power_spectrum_sum, n)
}

/// Calculates the spectral skewness
/// 
/// Requires the spectrum power mass function (PMF), RFFT magnitude frequencies, and spectral centroid
/// Reference: Eyben, pp. 23, 39-40
pub fn spectral_skewness(magnitude_spectrum: &Vec<f64>, rfft_freqs: &Vec<f64>) -> f64 {
    let power_spectrum = make_power_spectrum(magnitude_spectrum);
    let spectrum_pmf = make_spectrum_pmf(&power_spectrum, power_spectrum.iter().sum());
    let spectral_centroid = compute_spectral_centroid(magnitude_spectrum, rfft_freqs, magnitude_spectrum.iter().sum());
    let spectral_variance = compute_spectral_variance(&spectrum_pmf, rfft_freqs, spectral_centroid);
    compute_spectral_skewness(&spectrum_pmf, rfft_freqs, spectral_centroid, spectral_variance)
}

/// Calculates the spectral slope from provided power spectrum.
/// Reference: Eyben, pp. 35-38
pub fn spectral_slope(magnitude_spectrum: &Vec<f64>) -> f64 {
    let power_spectrum = make_power_spectrum(magnitude_spectrum);
    let power_spectrum_sum: f64 = power_spectrum.iter().sum();
    compute_spectral_slope(&power_spectrum, power_spectrum_sum)
}

/// Calculates the spectral slope from provided power spectrum, between the frequencies
/// specified. The frequencies specified do not have to correspond to exact bin indices.
/// Reference: Eyben, pp. 35-38
pub fn spectral_slope_region(magnitude_spectrum: &Vec<f64>, rfft_freqs: &Vec<f64>, f_lower: f64, f_upper: f64, sample_rate: u16) -> f64 {
    let power_spectrum = make_power_spectrum(magnitude_spectrum);
    compute_spectral_slope_region(&power_spectrum, rfft_freqs, f_lower, f_upper, sample_rate)
}

/// Calculates the spectral variance
/// Reference: Eyben, pp. 23, 39-40
pub fn spectral_variance(magnitude_spectrum: &Vec<f64>, rfft_freqs: &Vec<f64>) -> f64 {
    let power_spectrum = make_power_spectrum(magnitude_spectrum);
    let spectrum_pmf = make_spectrum_pmf(&power_spectrum, power_spectrum.iter().sum());
    let spectral_centroid = compute_spectral_centroid(magnitude_spectrum, rfft_freqs, magnitude_spectrum.iter().sum());
    compute_spectral_variance(&spectrum_pmf, rfft_freqs, spectral_centroid)
}

/// Calculates the zero crossing rate.
/// Reference: Eyben, p. 20
pub fn zero_crossing_rate(audio: &[f64], sample_rate: u16) -> f64 {
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
