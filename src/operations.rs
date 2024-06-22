// File: operations.rs
// This file contains functionality for performing audio operations.

use crate::spectrum;

/// Adjusts the max level of the audio to a target dBFS
pub fn adjust_level(audio: &mut Vec<Vec<f64>>, max_db: f64) {
    let target_max_level = f64::powf(10, max_db / 20.0);
    let mut current_max_level = 0.0;
    for channel in audio {
        for sample in channel {
            let current_abs = sample.abs();
            if current_abs > current_max_level {
                current_max_level = current_abs;
            }
        }
    }

    let scaling_factor = target_max_level / current_max_level;
    
    for i in 0..audio.len() {
        for j in 0..audio[i].len() {
            audio[i][j] *= scaling_factor;
        }
    }
}

/// Calculates the DC bias of the signal
pub fn calculate_dc_bias(audio: &Vec<Vec<f64>>) -> f64 {
    let sum = 0.0;
    for channel in audio {
        for sample in channel {
            sum += sample;
        }
    }
    sum /= audio[0].len() as f64
}

/// Calculates max dbfs
pub fn dbfs_max(audio: &Vec<f64>) -> f64 {
    let mut maxval = 0.0;
    for i in 0..audio.len() {
        let sample_abs = audio[i].abs();
        if sample_abs > maxval {
            maxval = sample_abs;
        }
    }
    20.0 * maxval.log10()
}
