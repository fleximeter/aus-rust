/// File: audio_analysis_tools.rs
/// 
/// This file contains tools for computing audio features, such as dBFS.


// The lowest f64 value for which dBFS can be computed.
// All lower values will result in f64::NEG_ININITY.
// Note that a value of 1e-20 corresponds to a dBFS of -400.0.
const DBFS_EPSILON: f64 = 1e-20;

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
