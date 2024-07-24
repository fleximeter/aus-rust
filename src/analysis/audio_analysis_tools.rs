// File: audio_analysis_tools.rs
// 
// This file contains tools for computing audio features, such as dBFS.

use pyin;

// The lowest f64 value for which dBFS can be computed.
// All lower values will result in f64::NEG_ININITY.
// Note that a value of 1e-20 corresponds to a dBFS of -400.0.
const DBFS_EPSILON: f64 = 1e-20;

/// Calculates the DC bias of the signal.
/// 
/// # Example
/// 
/// ```
/// use aus::analysis::dc_bias;
/// let file = aus::read("myfile.wav").unwrap();
/// let bias = dc_bias(&file.samples[0]);
/// ```
#[inline(always)]
pub fn dc_bias(audio: &[f64]) -> f64 {
    let mut sum = 0.0;
    for i in 0..audio.len() {
        sum += audio[i];
    }
    sum / audio.len() as f64
}

/// Calculates dBFS. All dBFS values below `epsilon` will render as NEG_INFINITY. Suggested `epsilon` value is 1e-20, corresponding to -400.0 dBFS.
/// 
/// # Example
/// 
/// ```
/// use aus::analysis::dbfs;
/// let level = dbfs(0.5223, 1e-20);
/// ```
#[inline(always)]
pub fn dbfs(val: f64, epsilon: f64) -> f64 {
    if val.abs() < epsilon {
        f64::NEG_INFINITY
    } else {
        20.0 * val.log10()
    }
}

/// Calculates the max dBFS in a list of audio samples.
/// 
/// # Example
/// 
/// ```
/// use aus::analysis::dbfs_max;
/// let file = aus::read("myfile.wav").unwrap();
/// let max_dbfs = dbfs_max(&file.samples[0]);
/// ```
pub fn dbfs_max(audio: &[f64]) -> f64 {
    let mut maxval = 0.0;
    for i in 0..audio.len() {
        let sample_abs = audio[i].abs();
        if sample_abs > maxval {
            maxval = sample_abs;
        }
    }
    dbfs(maxval, DBFS_EPSILON)
}

/// Extracts the RMS energy of the signal.
/// (Eyben, pp. 21-22)
/// 
/// # Example
/// 
/// ```
/// use aus::analysis::energy;
/// let file = aus::read("myfile.wav").unwrap();
/// let rms_energy = energy(&file.samples[0]);
/// ```
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
/// 
/// # Example
/// 
/// ```
/// use aus::analysis::zero_crossing_rate;
/// let file = aus::read("myfile.wav").unwrap();
/// let zcr = zero_crossing_rate(&file.samples[0], file.sample_rate);
/// ```
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

/// Performs pYIN pitch estimation using the `pyin` crate.
/// This wrapper expects the pitch to be the same for the entire audio array,
/// so it will run the pYIN algorithm and choose the median output frequency.
pub fn pyin_pitch_estimator_single(audio: &[f64], sample_rate: u32, f_min: f64, f_max: f64) -> f64 {
    let frame_length: usize = usize::min(audio.len(), 14000);
    let resolution = 0.1;
    let fill_unvoiced = f64::NAN;
    let framing = pyin::Framing::Center::<f64>(pyin::PadMode::<f64>::Constant(0.0));
    
    // this is necessary because the version of the pyin crate I'm using calls unwind() on a realfft operation. how annoying!!
    let mut executor = pyin::PYINExecutor::<f64>::new(f_min, f_max, sample_rate, frame_length, None, None, Some(resolution));
    let result = executor.pyin(audio, fill_unvoiced, framing);
    
    // tuple index 1 is the frequency estimates (that is, in this version of pyin - 1.2.0)
    let mut output_vec: Vec<f64> = Vec::with_capacity(result.1.len());
    for i in 0..result.1.len() {
        if !result.1[i].is_nan() {
            output_vec.push(result.1[i]);
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

/// Performs pYIN pitch estimation using the `pyin` crate.
/// Returns the pYIN output vectors (timestamp, pitch estimation, probability, voiced).
/// See https://docs.rs/pyin/1.2.0/pyin/struct.PYINExecutor.html#method.pyin for details.
pub fn pyin_pitch_estimator(audio: &[f64], sample_rate: u32, f_min: f64, f_max: f64, frame_length: usize) -> (Vec<f64>, Vec<f64>, Vec<bool>, Vec<f64>) {
    let resolution = 0.1;
    let fill_unvoiced = f64::NAN;
    let framing = pyin::Framing::Center::<f64>(pyin::PadMode::<f64>::Constant(0.0));
    let mut executor = pyin::PYINExecutor::<f64>::new(f_min, f_max, sample_rate, frame_length, None, None, Some(resolution));
    executor.pyin(audio, fill_unvoiced, framing)
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::read;

    #[test]
    fn test_pyin() {
        let fft_size: usize = 2048;

        let audio_path = String::from("D:\\Recording\\Samples\\Iowa\\Bass.arco.mono.2444.1\\samples\\Bass.arco.sulD.ff.C3B3.mono.19.wav");
        let audio = match read(&audio_path) {
            Ok(x) => x,
            Err(_) => panic!("could not read audio")
        };
        
        let result = pyin_pitch_estimator(&audio.samples[0], audio.sample_rate, 50.0, 500.0, fft_size);
        println!("Timestamps: {:?}\nFrequencies: {:?}\nVoiced: {:?}\nProbabilities: {:?}", result.0, result.1, result.2, result.3);
        //println!("{}", analysis::pyin_pitch_estimator_single(&audio.samples[0], audio.sample_rate, 50.0, 500.0));
    }

    #[test]
    fn test_pyin_single() {
        let fft_size: usize = 2048;

        let audio_path = String::from("D:\\Recording\\Samples\\Iowa\\Bass.arco.mono.2444.1\\samples\\Bass.arco.sulD.ff.C3B3.mono.19.wav");
        let audio = match read(&audio_path) {
            Ok(x) => x,
            Err(_) => panic!("could not read audio")
        };
        
        let result = pyin_pitch_estimator_single(&audio.samples[0], audio.sample_rate, 50.0, 500.0);
        println!("Result: {}", result);
    }
}