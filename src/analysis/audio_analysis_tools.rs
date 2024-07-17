// File: audio_analysis_tools.rs
// 
// This file contains tools for computing audio features, such as dBFS.

use pyin;
use ndarray;
use std::borrow::Borrow;
use std::panic::{self, AssertUnwindSafe};
use pitch_detection::detector::{yin, PitchDetector};
use pitch_detection::Pitch;

// The lowest f64 value for which dBFS can be computed.
// All lower values will result in f64::NEG_ININITY.
// Note that a value of 1e-20 corresponds to a dBFS of -400.0.
const DBFS_EPSILON: f64 = 1e-20;

/// Calculates the DC bias of the signal.
#[inline(always)]
pub fn dc_bias(audio: &[f64]) -> f64 {
    let mut sum = 0.0;
    for i in 0..audio.len() {
        sum += audio[i];
    }
    sum / audio.len() as f64
}

/// Calculates dBFS. All dBFS values below 1e-20 will render as NEG_INFINITY.
#[inline(always)]
pub fn dbfs(val: f64) -> f64 {
    if val.abs() < DBFS_EPSILON {
        f64::NEG_INFINITY
    } else {
        20.0 * val.log10()
    }
}

/// Calculates the max dBFS in a list of audio samples.
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

/// Extracts the RMS energy of the signal.
/// (Eyben, pp. 21-22)
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
/// (Eyben, p. 20)
pub fn zero_crossing_rate(audio: &[f64], sample_rate: u32) -> f64 {
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

/// Performs pYIN pitch estimation.
/// This wrapper expects the pitch to be the same for the entire audio array,
/// so it will run the pYIN algorithm and choose the median output frequency.
pub fn pyin_pitch_estimator_single(audio: &[f64], sample_rate: u32, f_min: f64, f_max: f64) -> f64 {
    let audio_arr = ndarray::Array::<f64, ndarray::Ix1>::from_vec(audio.to_vec());
    let frame_length: usize = usize::min(audio.len(), 14000);
    let resolution = 0.1;
    let fill_unvoiced = f64::NAN;
    let framing = pyin::Framing::Center::<f64>(pyin::PadMode::<f64>::Constant(0.0));
    
    // this is necessary because the version of the pyin crate I'm using calls unwind() on a realfft operation. how annoying!!
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        let mut executor = pyin::PYINExecutor::<f64>::new(f_min, f_max, sample_rate, frame_length, None, None, Some(resolution));
        executor.pyin(ndarray::CowArray::from(audio_arr), fill_unvoiced, framing)
    }));
    let (output, voiced, probs) = match result {
        Ok(x) => (x.0.to_vec(), x.1.to_vec(), x.2.to_vec()),
        Err(err) => {
            (vec![], vec![], vec![])
        }
    };
    let mut output_vec: Vec<f64> = Vec::with_capacity(output.len());
    for i in 0..output.len() {
        if !output[i].is_nan() {
            output_vec.push(output[i]);
        }
    }
    if output_vec.len() > 0 {
        output_vec.sort_unstable_by(|a, b| {
            match a.partial_cmp(b) {
                Some(x) => x,
                None => std::cmp::Ordering::Equal
            }
        });
        let median = output_vec[output_vec.len() / 2];
        median
    } else {
        f64::NAN
    }
}

/// Performs pYIN pitch estimation.
/// Returns the pYIN output vectors (pitch estimation, voiced, and probability).
pub fn pyin_pitch_estimator(audio: &[f64], sample_rate: u32, f_min: f64, f_max: f64, frame_length: usize) -> (Vec<f64>, Vec<bool>, Vec<f64>) {
    let audio_arr = ndarray::Array::<f64, ndarray::Ix1>::from_vec(audio.to_vec());
    let resolution = 0.1;
    let fill_unvoiced = f64::NAN;
    let framing = pyin::Framing::Center::<f64>(pyin::PadMode::<f64>::Constant(0.0));
    let mut executor = pyin::PYINExecutor::<f64>::new(f_min, f_max, sample_rate, frame_length, None, None, Some(resolution));
    
    // this is necessary because the version of the pyin crate I'm using calls unwind() on a realfft operation. how annoying!!
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        executor.pyin(ndarray::CowArray::from(audio_arr), fill_unvoiced, framing)
    }));
    match result {
        Ok(x) => (x.0.to_vec(), x.1.to_vec(), x.2.to_vec()),
        Err(err) => {
            (vec![], vec![], vec![])
        }
    }
}

/// Performs YIN pitch estimation.
/// Returns (frequency, clarity).
pub fn yin_pitch_estimator(audio: &[f64], sample_rate: u32, frame_length: usize) -> (f64, f64) {
    let mut detector = yin::YINDetector::<f64>::new(frame_length, frame_length);
    let result = match detector.get_pitch(audio, sample_rate as usize, 0.01, 0.8) {
        Some(x) => x,
        None => Pitch::<f64>{frequency: 0.0, clarity: 0.0}
    };
    (result.frequency, result.clarity)
}
