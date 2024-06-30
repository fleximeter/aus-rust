/// File: operations.rs
/// This file contains functionality for performing audio operations.

use crate::spectrum;
use num::Complex;
use std::collections::HashMap;
use rand::Rng;

/// Calculates RMS for a list of audio samples
#[inline(always)]
fn rms(data: &[f64]) -> f64 {
    let mut sum = 0.0;
    for i in 0..data.len() {
        sum += f64::powf(data[i], 2.0);
    }
    f64::sqrt(sum / data.len() as f64)
}

/// Adjusts the max level of the audio to a target dBFS
pub fn adjust_level(audio: &mut Vec<f64>, max_db: f64) {
    let target_max_level = f64::powf(10.0, max_db / 20.0);
    let mut current_max_level = 0.0;

    for i in 0..audio.len() {
        let current_abs = audio[i].abs();
        if current_abs > current_max_level {
            current_max_level = current_abs;
        }
    }

    let scaling_factor = target_max_level / current_max_level;
    
    for i in 0..audio.len() {
        audio[i] *= scaling_factor;
    }
}

/// Implements a fade-in on a vector of audio samples. The duration is in frames.
pub fn fade_in(audio: &mut Vec<f64>, envelope: spectrum::WindowType, duration: usize) {
    let duration = usize::min(duration, audio.len());
    let envelope_samples = spectrum::generate_window(envelope, duration * 2);
    for i in 0..duration {
        audio[i] *= envelope_samples[i];
    }
}

/// Implements a fade-out on a vector of audio samples. The duration is in frames.
pub fn fade_out(audio: &mut Vec<f64>, envelope: spectrum::WindowType, duration: usize) {
    let duration = usize::min(duration, audio.len());
    let envelope_samples = spectrum::generate_window(envelope, duration * 2);
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

/// Forces equal energy on a mono signal over time using linear interpolation.
/// For example, if a signal initially has high energy, and gets less energetic, 
/// this will adjust the energy level so that it does not decrease.
/// Better results come with using a larger window size, so the energy changes more gradually.
pub fn force_equal_energy(audio: &mut Vec<f64>, dbfs: f64, window_size: usize) {
    let target_level = f64::powf(10.0, dbfs / 20.0);
    let num_level_frames = f64::ceil((audio.len() / window_size) as f64) as usize;
    let mut energy_levels: Vec<f64> = vec![0.0; num_level_frames + 2];

    // Compute the energy levels for each frame
    for i in 0..num_level_frames {
        let start_idx = i * window_size;
        let end_idx = usize::min(start_idx + window_size, audio.len());
        energy_levels[i+1] = rms(&audio[start_idx..end_idx]);
    }

    // The first and last frames need to be copied because we will be interpolating with combined adjacent half frames.
    energy_levels[0] = energy_levels[1];
    energy_levels[num_level_frames + 1] = energy_levels[num_level_frames];

    // Adjust level for the first half frame
    for i in 0..window_size / 2 {
        audio[i] = audio[i] * target_level / energy_levels[0];
    }

    // Adjust level for all other frames using linear interpolation.
    // To interpolate, we make combined adjacent half frames.
    for level_frame_idx in 1..num_level_frames + 1 {
        let slope = (energy_levels[level_frame_idx + 1] - energy_levels[level_frame_idx]) / window_size as f64;
        let y_int = energy_levels[level_frame_idx];
        let start_frame = level_frame_idx * window_size - window_size / 2;
        let end_frame = usize::min(start_frame + window_size, audio.len());
        for sample_idx in start_frame..end_frame {
            let scaler = 1.0 / (slope * (sample_idx - start_frame) as f64 + y_int);
            audio[sample_idx] *= scaler;
        }
    }

    // Find the current max level in the adjusted signal
    let mut max_level = 0.0;
    for sample_idx in 0..audio.len() {
        max_level = f64::max(audio[sample_idx].abs(), max_level);
    }

    // Scale the adjusted signal to the target max level
    for sample_idx in 0..audio.len() {
        audio[sample_idx] *= target_level / max_level;
    }
}

/// Exchanges samples in an audio file.
/// Each sample is swapped with the sample *hop* steps ahead or *hop* steps behind.
pub fn exchange_frames(data: &mut [f64], hop: usize) {
    let end_idx = data.len() - data.len() % (hop * 2);
    let step = hop * 2;
    for i in (0..end_idx).step_by(step) {
        for j in i..i+hop {
            let temp = data[j];
            data[j] = data[j + hop];
            data[j + hop] = temp;
        }
    }
}

/// Stochastically exchanges samples in an audio file.
/// Each sample is swapped with the sample up to *hop* steps ahead or *hop* steps behind. 
pub fn exchange_frames_stochastic(data: &mut [f64], max_hop: usize) {
    let mut future_indices: HashMap<usize, bool> = HashMap::with_capacity(data.len());
    let mut idx = 0;
    while idx < data.len() {
        // If *idx* is not in the list of future indices which have already been swapped,
        // we can try to swap it with something.
        if !future_indices.contains_key(&idx) {
            // Generate a vector of possible indices in the future with which we could swap this index
            let mut possible_indices: Vec<usize> = Vec::new();
            for i in idx..usize::min(idx + max_hop, data.len()) {
                if !future_indices.contains_key(&i) {
                    possible_indices.push(i);
                }
            }

            // Choose a random index to swap with, and perform the swap
            let swap_idx = rand::thread_rng().gen_range(0..possible_indices.len());
            let temp = data[idx];
            data[idx] = data[swap_idx];
            data[swap_idx] = temp;
            
            // Record that the swap index has been used
            future_indices.insert(swap_idx, true);
        }
        idx += 1;
    }
}
