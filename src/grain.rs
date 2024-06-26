// File: grain.rs
// This file contains functionality for audio granulation.

use crate::spectrum;

/// Extracts a grain from an audio file based on provided parameters. If you do not
/// specify a maximum window length, the window will be the entire size of the grain.
/// If the max window length is shorter than the grain, the window will be split in half
/// and applied to the beginning and end of the grain.
pub fn extract_grain(audio: &Vec<f64>, start_frame: usize, grain_length: usize, window: spectrum::WindowType, max_window_length: Option<usize>) -> Vec<f64> {
    let mut grain: Vec<f64> = vec![0.0; grain_length];

    if start_frame + grain_length < audio.len() {
        let window_length = match max_window_length {
            Some(x) => usize::min(grain_length, x),
            None => grain_length
        };

        let window = match window {
            spectrum::WindowType::Bartlett => spectrum::bartlett(window_length),
            spectrum::WindowType::Blackman => spectrum::blackman(window_length),
            spectrum::WindowType::Hamming => spectrum::hamming(window_length),
            spectrum::WindowType::Hanning => spectrum::hanning(window_length),
        };

        // extract the grain
        for i in start_frame..start_frame + grain_length {
            grain[i-start_frame] = audio[i];
        }

        // apply the first window half
        for i in 0..window_length / 2 {
            grain[i] *= window[i];
        }

        // apply the last window half
        let mut grain_idx = grain_length - (window_length - window_length / 2);
        for i in window_length / 2..window_length {
            grain[grain_idx] *= window[i];
            grain_idx += 1;
        }
    }

    grain
}

/// Finds the max dbfs in a list of grains
pub fn find_max_grain_dbfs(grains: &Vec<Vec<f64>>) -> f64 {
    let mut dbfs = 20.0 * grains[0][0].log10();
    for i in 0..grains.len() {
        for j in 0..grains[i].len() {
            let local_dbfs = 20.0 * grains[i][j].log10();
            if local_dbfs > dbfs {
                dbfs = local_dbfs;
            }
        }
    }
    dbfs
}

/// Merges a vector of grains with some overlap
pub fn merge_grains(grains: &Vec<Vec<f64>>, overlap_size: usize) -> Vec<f64> {
    let mut audio: Vec<f64> = Vec::new();
    let mut grain_start_and_end_indices: Vec<(usize, usize)> = Vec::with_capacity(grains.len());

    // Get the audio start and end indices for each grain, considering the 
    // amount of overlap that we want
    let mut current_start_idx = 0;
    for i in 0..grains.len() {
        let indices = (current_start_idx, current_start_idx + grains[i].len());
        grain_start_and_end_indices.push(indices);
        current_start_idx += grains[i].len() - overlap_size;
    }

    // Track the lowest and highest grain indices we are currently using
    let mut bottom_grain_idx = 0;
    let mut top_grain_idx = 0;

    // Track the index of the current sample that we are making
    let mut current_sample_idx = 0;
    while bottom_grain_idx < grains.len() {
        // check if we've moved beyond the end of the bottom grain
        if grain_start_and_end_indices[bottom_grain_idx].1 < current_sample_idx {
            bottom_grain_idx += 1;
        }
        // check if we've moved into the next grain beyond our top grain
        if top_grain_idx + 1 < grains.len() {
            if grain_start_and_end_indices[top_grain_idx + 1].0 <= current_sample_idx {
                top_grain_idx += 1;
            }
        }

        // If we haven't moved beyond the last grain, we can compute the sample
        if bottom_grain_idx < grains.len() {
            let mut sample = 0.0;
            for i in bottom_grain_idx..top_grain_idx + 1 {
                let local_idx = current_sample_idx - grain_start_and_end_indices[i].0;
                sample += grains[i][local_idx];
            }
            audio.push(sample);
        }
        current_sample_idx += 1;
    }
    
    audio
}

/// Scales the peaks of a vector of grains so that all grains have the same amplitude
pub fn scale_grain_peaks(grains: &mut Vec<Vec<f64>>) {
    let mut maxamp = 0.0;
    for i in 0..grains.len() {
        for j in 0..grains[i].len() {
            maxamp = f64::max(grains[i][j].abs(), maxamp);
        }
    }
    for i in 0..grains.len() {
        for j in 0..grains[i].len() {
            grains[i][j] /= maxamp;
        }
    }
}
