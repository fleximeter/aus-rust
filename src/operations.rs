/// File: operations.rs
/// This file contains functionality for performing audio operations.

use std::collections::HashMap;
use rand::Rng;
use std::f64::consts::PI;

/// Represents pan laws
pub enum PanLaw {
    Linear,
    ConstantPower,
    Neg4_5dB
}

/// Calculates RMS for a list of audio samples
/// 
/// # Example
/// ```
/// use audiorust::operations::rms;
/// let pseudo_audio = vec![0.0, 0.1, 0.3, -0.4, 0.1, -0.51];
/// let rms_energy = rms(&pseudo_audio);
/// ```
#[inline(always)]
pub fn rms(data: &[f64]) -> f64 {
    let mut sum = 0.0;
    for i in 0..data.len() {
        sum += f64::powf(data[i], 2.0);
    }
    f64::sqrt(sum / data.len() as f64)
}

/// Adjusts the max level of the audio to a target dBFS
/// 
/// # Example
/// ```
/// use audiorust::operations::adjust_level;
/// let mut pseudo_audio = vec![0.0, 0.1, 0.3, -0.4, 0.1, -0.51];
/// let max_db = -6.0;
/// adjust_level(&mut pseudo_audio, max_db);
/// ```
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
/// 
/// # Example
/// ```
/// use audiorust::operations::fade_in;
/// use audiorust::WindowType;
/// let mut pseudo_audio = vec![0.0, 0.1, 0.3, -0.4, 0.1, -0.51];
/// let num_frames = 1024;
/// fade_in(&mut pseudo_audio, WindowType::Hanning, num_frames);
/// ```
pub fn fade_in(audio: &mut Vec<f64>, envelope: crate::WindowType, duration: usize) {
    let duration = usize::min(duration, audio.len());
    let envelope_samples = crate::generate_window(envelope, duration * 2);
    for i in 0..duration {
        audio[i] *= envelope_samples[i];
    }
}

/// Implements a fade-out on a vector of audio samples. The duration is in frames.
/// 
/// # Example
/// ```
/// use audiorust::operations::fade_out;
/// use audiorust::WindowType;
/// let mut pseudo_audio = vec![0.0, 0.1, 0.3, -0.4, 0.1, -0.51];
/// let num_frames = 1024;
/// fade_out(&mut pseudo_audio, WindowType::Hanning, num_frames);
/// ```
pub fn fade_out(audio: &mut Vec<f64>, envelope: crate::WindowType, duration: usize) {
    let duration = usize::min(duration, audio.len());
    let envelope_samples = crate::generate_window(envelope, duration * 2);
    for i in audio.len() - duration..audio.len() {
        audio[i] *= envelope_samples[i + duration * 2 - audio.len()];
    }
}

/// Leaks DC bias of an audio signal by averaging
/// 
/// # Example
/// ```
/// use audiorust::operations::leak_dc_bias_averager;
/// let mut pseudo_audio = vec![1.0; 44100];
/// leak_dc_bias_averager(&mut pseudo_audio);
/// ```
pub fn leak_dc_bias_averager(audio: &mut Vec<f64>) {
    let average = audio.iter().sum::<f64>() / audio.len() as f64;
    for i in 0..audio.len() {
        audio[i] -= average;
    }
}

/// Leaks DC bias of an audio signal by filtering
/// 
/// # Example
/// ```
/// use audiorust::operations::leak_dc_bias_filter;
/// let mut pseudo_audio = vec![1.0; 44100];
/// leak_dc_bias_filter(&mut pseudo_audio);
/// ```
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
/// This algorithm divides the signal into adjacent windowed chunks, computes the energy level
/// for each chunk, and generates scaling coefficients to force the entire signal to have a similar energy level.
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
/// Each sample is swapped with the sample `hop` steps ahead or `hop` steps behind.
pub fn exchange_frames(audio: &mut [f64], hop: usize) {
    let end_idx = audio.len() - audio.len() % (hop * 2);
    let step = hop * 2;
    for i in (0..end_idx).step_by(step) {
        for j in i..i+hop {
            let temp = audio[j];
            audio[j] = audio[j + hop];
            audio[j + hop] = temp;
        }
    }
}

/// Stochastically exchanges samples in an audio file.
/// Each sample is swapped with the sample up to `max_hop` steps ahead or `max_hop` steps behind. 
pub fn exchange_frames_stochastic(audio: &mut [f64], max_hop: usize) {
    let mut future_indices: HashMap<usize, bool> = HashMap::with_capacity(audio.len());
    let mut idx = 0;
    while idx < audio.len() {
        // If *idx* is not in the list of future indices which have already been swapped,
        // we can try to swap it with something.
        if !future_indices.contains_key(&idx) {
            // Generate a vector of possible indices in the future with which we could swap this index
            let mut possible_indices: Vec<usize> = Vec::new();
            for i in idx..usize::min(idx + max_hop, audio.len()) {
                if !future_indices.contains_key(&i) {
                    possible_indices.push(i);
                }
            }

            // Choose a random index to swap with, and perform the swap
            let swap_idx = rand::thread_rng().gen_range(0..possible_indices.len());
            let temp = audio[idx];
            audio[idx] = audio[swap_idx];
            audio[swap_idx] = temp;
            
            // Record that the swap index has been used
            future_indices.insert(swap_idx, true);
        }
        idx += 1;
    }
}

/// A multichannel panner, moving from `start_pos` to `end_pos` over `num_iterations`. 
/// It generates a list of pan coefficients (each coefficient is the volume coefficient for the corresponding channel).
/// If you want to wrap around, you can set an end position beyond the number of channels, and the panner will
/// automatically wrap around back to channel 0 again. For example, if you have 8 channels, and want to wrap around twice,
/// you can start at 0.0 and end at 23.0.
/// 
/// This panner is set up for linear panning, constant power panning, or -4.5 dB panning.
/// (https://www.cs.cmu.edu/~music/icm-online/readings/panlaws/panlaws.pdf)
pub fn panner(num_channels: usize, start_pos: f64, end_pos: f64, num_iterations: usize, pan_law: PanLaw) -> Vec<Vec<f64>> {
    let mut pos_vec: Vec<f64> = vec![0.0; num_iterations];
    let step_val: f64 = (end_pos - start_pos) / num_iterations as f64;
    pos_vec[0] = start_pos % num_iterations as f64;
    for i in 1..num_iterations {
        pos_vec[i] = pos_vec[i-1] + step_val;
    }

    let mut pan_vec: Vec<Vec<f64>> = Vec::with_capacity(num_iterations);
    for i in 0..num_iterations {
        let mut coefficients: Vec<f64> = vec![0.0; num_channels];
        let int_part = pos_vec[i].trunc();
        let decimal_part = pos_vec[i].fract();
        let pos = int_part as usize % num_channels;
        let next_pos = (pos + 1) % num_channels;
        let theta = decimal_part * PI / 2.0;
        match pan_law {
            PanLaw::Linear => {
                coefficients[pos] = 1.0 - decimal_part;
                coefficients[next_pos] = decimal_part;
            },
            PanLaw::ConstantPower => {
                coefficients[pos] = f64::cos(theta);
                coefficients[next_pos] = f64::sin(theta);
            },
            PanLaw::Neg4_5dB => {
                coefficients[pos] = f64::sqrt((PI / 2.0 - theta) * 2.0 / PI * f64::cos(theta));
                coefficients[next_pos] = f64::sqrt(decimal_part * f64::sin(theta));
            }
        }
        pan_vec.push(coefficients);
    }

    pan_vec
}

/// Maps pan positions to actual speaker positions. You pass a mapping array 
/// that lists the speaker numbers in panning order.
///
/// This is useful if you want to use a different numbering system for your 
/// pan positions than the numbering system used for the actual output channels.
/// For example, you might want to pan in a circle for a quad-channel setup,
/// but the hardware is set up for stereo pairs.
///
/// Example: Suppose you have a quad setup. Your mapper would be [0, 1, 3, 2] 
/// if you are thinking clockwise, or [1, 0, 2, 3] if you are thinking counterclockwise. 
/// If you have an 8-channel setup, your mapper would be [0, 1, 3, 5, 7, 6, 4, 2] 
/// for clockwise and [1, 0, 2, 4, 6, 7, 5, 3] for counterclockwise.
pub fn pan_mapper(pan_coefficients: &mut [Vec<f64>], map: &[usize]) {
    let mut swap: Vec<f64> = vec![0.0; map.len()];
    for i in 0..pan_coefficients.len() {
        for j in 0..pan_coefficients[i].len() {
            swap[j] = pan_coefficients[i][j];
        }
        for j in 0..pan_coefficients[i].len() {
            pan_coefficients[i][map[j]] = swap[j];
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    const SAMPLE_RATE: u32 = 44100;
    const DIR: &str = "D:/Recording/tests";
    const AUDIO: &str = "D:/Recording/tests/grains.wav";

    /// Tests of level adjustment and fade in / fade out
    #[test]
    fn test_fade_and_level() {
        let path = String::from(AUDIO);
        let mut audio = match crate::read(&path) {
            Ok(x) => x,
            Err(_) => panic!("could not read audio")
        };
        adjust_level(&mut audio.samples[0], -12.0);
        fade_in(&mut audio.samples[0], crate::WindowType::Hanning, 44100 * 4);
        fade_out(&mut audio.samples[0], crate::WindowType::Hanning, 44100 * 4);
        let path: String = String::from(format!("{}/out1.wav", DIR));
        match crate::write(&path, &audio) {
            Ok(_) => (),
            Err(_) => panic!("could not write audio")
        }
    }

    /// Test force equal energy
    #[test]
    pub fn test_equal_energy() {
        let path = String::from(AUDIO);
        let mut audio = match crate::read(&path) {
            Ok(x) => x,
            Err(_) => panic!("could not read audio")
        };
        force_equal_energy(&mut audio.samples[0], -16.0, 16384);
        let path: String = String::from(format!("{}/out2.wav", DIR));
        match crate::write(&path, &audio) {
            Ok(_) => (),
            Err(_) => panic!("could not write audio")
        }    
    }
}