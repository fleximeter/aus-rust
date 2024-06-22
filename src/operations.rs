// File: operations.rs
// This file contains functionality for performing audio operations.

use crate::spectrum;
use std::cmp::min;

/// Adjusts the max level of the audio to a target dBFS
pub fn adjust_level(audio: &mut Vec<Vec<f64>>, max_db: f64) {
    let target_max_level = f64::powf(10.0, max_db / 20.0);
    let mut current_max_level = 0.0;

    for i in 0..audio.len() {
        for j in 0..audio[i].len() {
            let current_abs = audio[i][j].abs();
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
    let mut sum = 0.0;
    for channel in audio {
        for sample in channel {
            sum += sample;
        }
    }
    sum /= audio[0].len() as f64;
    sum
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

/// Implements a fade-in on a vector of audio samples. The duration is in frames.
pub fn fade_in(audio: &mut Vec<f64>, envelope: spectrum::WindowType, duration: usize) {
    let duration = min(duration, audio.len());

    let envelope_samples = match &envelope {
        spectrum::WindowType::Bartlett => spectrum::bartlett(duration * 2),
        spectrum::WindowType::Blackman => spectrum::blackman(duration * 2),
        spectrum::WindowType::Hanning => spectrum::hanning(duration * 2),
        spectrum::WindowType::Hamming => spectrum::hamming(duration * 2)
    };

    for i in 0..duration {
        audio[i] *= envelope_samples[i];
    }
}

/// Implements a fade-out on a vector of audio samples. The duration is in frames.
pub fn fade_out(audio: &mut Vec<f64>, envelope: spectrum::WindowType, duration: usize) {
    let duration = min(duration, audio.len());

    let envelope_samples = match &envelope {
        spectrum::WindowType::Bartlett => spectrum::bartlett(duration * 2),
        spectrum::WindowType::Blackman => spectrum::blackman(duration * 2),
        spectrum::WindowType::Hanning => spectrum::hanning(duration * 2),
        spectrum::WindowType::Hamming => spectrum::hamming(duration * 2)
    };

    for i in audio.len() - duration..audio.len() {
        audio[i] *= envelope_samples[i + duration * 2 - audio.len()];
    }
}

/// Leaks DC bias of an audio signal by averaging
pub fn leak_dc_bias_averager(audio: &mut Vec<f64>) {
    let average = audio.iter().sum::<f64>() / audio.len() as f64;
    for i in 0..audio.len() {
        audio[i] -= average;
    }
}

/// Leaks DC bias of an audio signal by filtering
pub fn leak_dc_bias_filter(audio: &mut Vec<f64>) {
    const ALPHA: f64 = 0.95;
    let mut delay_register = 0.0;
    for i in 0..audio.len() {
        let combined_signal = audio[i] + ALPHA * delay_register;
        audio[i] = combined_signal - delay_register;
        delay_register = combined_signal;
    }
}
