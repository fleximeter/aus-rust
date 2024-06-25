// File: spectrum.rs
// This file contains spectral resource functionality.

use realfft::RealFftPlanner;
use num::Complex;
use symphonia::core::sample;
use std::{cmp::max, collections::HashMap};

pub enum WindowType{
    Bartlett,
    Blackman,
    Hanning,
    Hamming
}

/// Creates a Bartlett window of size m
pub fn bartlett(m: usize) -> Vec<f64>{
    let mut window: Vec<f64> = vec![0.0; m];
    for i in 0..m {
        window[i] = (2.0 / (m as f64 - 1.0)) * ((m as f64 - 1.0) / 2.0 - f64::abs(i as f64 - (m as f64 - 1.0) / 2.0));
    }
    window
}

/// Creates a Blackman window of size m
pub fn blackman(m: usize) -> Vec<f64>{
    let mut window: Vec<f64> = vec![0.0; m];
    for i in 0..m {
        window[i] = 0.42 - 0.5 * f64::cos((2.0 * std::f64::consts::PI * i as f64) / (m as f64)) 
            + 0.08 * f64::cos((4.0 * std::f64::consts::PI * i as f64) / (m as f64));
    }
    window
}

/// Creates a Hanning window of size m
pub fn hanning(m: usize) -> Vec<f64>{
    let mut window: Vec<f64> = vec![0.0; m];
    for i in 0..m {
        window[i] = 0.5 - 0.5 * f64::cos((2.0 * std::f64::consts::PI * i as f64) / (m as f64 - 1.0));
    }
    window
}

/// Creates a Hamming window of size m
pub fn hamming(m: usize) -> Vec<f64>{
    let mut window: Vec<f64> = vec![0.0; m];
    for i in 0..m {
        window[i] = 0.54 - 0.46 * f64::cos((2.0 * std::f64::consts::PI * i as f64) / (m as f64 - 1.0));
    }
    window
}

/// Calculates the real FFT of a chunk of audio.
/// 
/// The input audio must be a 1D vector already of the appropriate size.
/// This function will return the complex spectrum.
pub fn rfft(audio: &mut [f64], fft_size: usize) -> Vec<Complex<f64>> {
    let mut real_planner = RealFftPlanner::<f64>::new();
    let r2c = real_planner.plan_fft_forward(fft_size);
    let mut input = audio;
    let mut spectrum: Vec<num::Complex<f64>> = r2c.make_output_vec();
    assert_eq!(input.len(), fft_size);
    assert_eq!(spectrum.len(), fft_size / 2 + 1);
    r2c.process(&mut input, &mut spectrum).unwrap();
    spectrum
}

/// Calculates the real IFFT of an audio spectrum.
/// 
/// The input audio must be a 1D vector already of the appropriate size.
/// This function will return the audio.
pub fn irfft(spectrum: &mut [Complex<f64>], fft_size: usize) -> Vec<f64> {
    let mut real_planner = RealFftPlanner::<f64>::new();
    let c2r = real_planner.plan_fft_inverse(fft_size);
    let mut input = spectrum;
    let mut audio: Vec<f64> = c2r.make_output_vec();
    assert_eq!(input.len(), fft_size / 2 + 1);
    assert_eq!(audio.len(), fft_size);
    c2r.process(&mut input, &mut audio).expect("Something went wrong in irfft");
    audio
}

/// This function creates the magnitude and phase spectra from provided
/// complex spectrum.
pub fn complex_to_polar_rfft(spectrum: Vec<Complex<f64>>) -> (Vec<f64>, Vec<f64>) {
    let mut magnitude_spectrum = vec![0.0 as f64; spectrum.len()];
    let mut phase_spectrum = vec![0.0 as f64; spectrum.len()];
    for i in 0..spectrum.len() {
        magnitude_spectrum[i] = f64::sqrt(f64::powf(spectrum[i].re, 2.0) + f64::powf(spectrum[i].im, 2.0));
        phase_spectrum[i] = f64::atan2(spectrum[i].im, spectrum[i].re);
    }
    (magnitude_spectrum, phase_spectrum)
}

/// This function creates the magnitude and phase spectra from provided
/// complex spectrum.
pub fn complex_to_polar_rstft(spectrum: Vec<Vec<Complex<f64>>>) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut magnitude_spectrum: Vec<Vec<f64>> = Vec::new();
    let mut phase_spectrum: Vec<Vec<f64>> = Vec::new();
    for i in 0..spectrum.len() {
        let mut frame_magnitude_spectrum = vec![0.0 as f64; spectrum[i].len()];
        let mut frame_phase_spectrum = vec![0.0 as f64; spectrum[i].len()];
        for j in 0..spectrum[i].len() {
            frame_magnitude_spectrum[j] = f64::sqrt(f64::powf(spectrum[i][j].re, 2.0) + f64::powf(spectrum[i][j].im, 2.0));
            frame_phase_spectrum[j] = f64::atan2(spectrum[i][j].im, spectrum[i][j].re);    
        }
        magnitude_spectrum.push(frame_magnitude_spectrum);
        phase_spectrum.push(frame_phase_spectrum);
    }    
    (magnitude_spectrum, phase_spectrum)
}

/// This function creates the complex spectrum from provided
/// magnitude and phase spectra
pub fn polar_to_complex_rfft(magnitude_spectrum: Vec<f64>, phase_spectrum: Vec<f64>) -> Vec<Complex<f64>> {
    let mut spectrum = vec![num::complex::Complex::new(0.0, 0.0); magnitude_spectrum.len()];    
    for i in 0..magnitude_spectrum.len() {
        let real = f64::cos(phase_spectrum[i]) * magnitude_spectrum[i];
        let imag = f64::sin(phase_spectrum[i]) * magnitude_spectrum[i];
        spectrum[i] = num::complex::Complex::new(real, imag);
    }
    spectrum
}

/// This function creates the complex spectrum from provided
/// magnitude and phase spectra
pub fn polar_to_complex_rstft(magnitude_spectrum: Vec<Vec<f64>>, phase_spectrum: Vec<Vec<f64>>) -> Vec<Vec<Complex<f64>>> {
    let mut spectrum: Vec<Vec<Complex<f64>>> = Vec::new();
    for i in 0..magnitude_spectrum.len() {
        let mut frame_spectrum = vec![num::complex::Complex::new(0.0, 0.0); magnitude_spectrum[i].len()];    
        for j in 0..magnitude_spectrum[i].len() {
            let real = f64::cos(phase_spectrum[i][j]) * magnitude_spectrum[i][j];
            let imag = f64::sin(phase_spectrum[i][j]) * magnitude_spectrum[i][j];
            frame_spectrum[i] = num::complex::Complex::new(real, imag);
        }
        spectrum.push(frame_spectrum);
    }
    spectrum
}

/// Gets the corresponding frequencies for rFFT data
pub fn rfftfreq(fft_size: usize, sample_rate: u16) -> Vec<f64> {
    let mut freqs = vec![0.0 as f64; fft_size / 2 + 1];
    let f_0 = sample_rate as f64 / fft_size as f64;
    for i in 1..freqs.len() {
        freqs[i] = f_0 * i as f64;
    }
    freqs
}

/// Calculates the real STFT of a chunk of audio.
/// 
/// The input audio must be a 1D vector already of the appropriate size.
/// This function will return the magnitude and phase spectrum.
pub fn rstft(audio: &mut Vec<f64>, fft_size: usize, hop_size: usize, window: WindowType) -> Vec<Vec<Complex<f64>>> {
    let mut real_planner = RealFftPlanner::<f64>::new();
    let r2c = real_planner.plan_fft_forward(fft_size);
    let mut spectrogram: Vec<Vec<Complex<f64>>> = Vec::new();
    
    // Get the window
    let window_samples = match &window {
        WindowType::Bartlett => bartlett(fft_size),
        WindowType::Blackman => blackman(fft_size),
        WindowType::Hanning => hanning(fft_size),
        WindowType::Hamming => hamming(fft_size),
    };
    
    let mut hop_idx = 0;
    let mut finished = false;
    while !finished {
        let start_idx = hop_idx * hop_size;
        let end_idx = if hop_idx * hop_size + fft_size < audio.len() {
            hop_idx * hop_size + fft_size
        } else {
            audio.len()
        };
        let num_zeros = if end_idx == audio.len() {
            finished = true;
            start_idx + fft_size - end_idx
        } else {
            0
        };

        // Apply the window to the frame of samples
        // Smaller window for the last frame if necessary
        let mut fft_input = if num_zeros > 0 {
            let window_samples = match &window {
                WindowType::Bartlett => bartlett(end_idx - start_idx),
                WindowType::Blackman => blackman(end_idx - start_idx),
                WindowType::Hanning => hanning(end_idx - start_idx),
                WindowType::Hamming => hamming(end_idx - start_idx),
            };
            let mut input = {
                let mut audio_chunk = audio[start_idx..end_idx].to_vec();
                for i in 0..audio_chunk.len() {
                    audio_chunk[i] *= window_samples[i];
                }
                audio_chunk
            };
            input.extend(vec![0.0; num_zeros]);
            input
        } else {
            let input = {
                let mut audio_chunk = audio[start_idx..end_idx].to_vec();
                for i in 0..audio_chunk.len() {
                    audio_chunk[i] *= window_samples[i];
                }
                audio_chunk
            };
            input
        };

        // prepare the output complex vector and check that the sizes are correct
        let mut spectrum = r2c.make_output_vec();
        assert_eq!(fft_input.len(), fft_size);
        assert_eq!(spectrum.len(), fft_size / 2 + 1);
        
        // process the FFT for this frame, and push it onto the output vector
        r2c.process(&mut fft_input, &mut spectrum).unwrap();
        spectrogram.push(spectrum);
        hop_idx += 1;
    }
    spectrogram
}

/// Calculates the inverse real STFT of a chunk of audio.
/// Note: For the STFT/ISTFT process to work correctly, you need to follow these guidelines:
///       a) Use the same window type for the STFT and ISTFT.
///       b) Choose an appropriate hop size for the window type to satisfy the constant overlap-add condition.
///          This is 50% of the FFT size for the Hanning and Hamming windows.
pub fn irstft(spectrogram: &mut Vec<Vec<Complex<f64>>>, fft_size: usize, hop_size: usize, window: WindowType) -> Vec<f64> {
    let mut real_planner = RealFftPlanner::<f64>::new();
    let c2r = real_planner.plan_fft_inverse(fft_size);
    let num_stft_frames = spectrogram.len();
    let num_output_frames = fft_size + (hop_size * (num_stft_frames - 1));
    let mut audio: Vec<f64> = Vec::with_capacity(num_output_frames);
    let mut audio_chunks: Vec<Vec<f64>> = vec![Vec::with_capacity(fft_size); num_stft_frames];
    let mut window_norm: Vec<f64> = vec![0.0; num_output_frames];

    // Get the window
    let window_samples = match &window {
        WindowType::Bartlett => bartlett(fft_size),
        WindowType::Blackman => blackman(fft_size),
        WindowType::Hanning => hanning(fft_size),
        WindowType::Hamming => hamming(fft_size)
    };

    // Perform IRFFT on each STFT frame
    for i in 0..num_stft_frames {
        let mut audio_frame = c2r.make_output_vec();
        assert_eq!(spectrogram[i].len(), fft_size / 2 + 1);
        assert_eq!(audio_frame.len(), fft_size);
        c2r.process(&mut spectrogram[i], &mut audio_frame).expect("Something went wrong in irfft");

        // window the samples
        for j in 0..audio_frame.len() {
            audio_chunks[i].push(audio_frame[j] * window_samples[j]);
        }

        // Compute the window norm for the current sample
        let start_idx = i * hop_size;
        let end_idx = start_idx + fft_size;
        for j in start_idx..end_idx {
            window_norm[j] += window_samples[j - start_idx].powf(2.0);
        }
    }

    // Overlap add the remaining chunks
    audio = overlap_add(&audio_chunks, fft_size, hop_size);

    // Apply the window norm to each sample
    for i in 0..audio.len() {
        audio[i] /= window_norm[i];
    }

    // Get the maximum level in the overlap-added audio
    let mut maxval = 0.0;
    for i in 0..audio.len() {
        maxval = f64::max(audio[i], maxval);
    }

    // Scale the level
    let max_dbfs = -6.0;
    let max_level_scaler = f64::powf(10.0, max_dbfs / 20.0) / maxval;
    for i in 0..audio.len() {
        audio[i] *= max_level_scaler;
    }
    
    audio
}

/// An efficient overlap add mechanism that doesn't require iterating through all 
/// frames for each sample
pub fn overlap_add(audio_chunks: &Vec<Vec<f64>>, fft_size: usize, hop_size: usize) -> Vec<f64> {
    let mut audio: Vec<f64> = Vec::new();

    // Get the global start and end index corresponding to each audio frame
    let mut frame_indices: Vec<(usize, usize)> = Vec::with_capacity(audio_chunks.len());
    let mut current_frame_start_idx: usize = 0;
    for i in 0..audio_chunks.len() {
        frame_indices.push((current_frame_start_idx, current_frame_start_idx + fft_size));
        current_frame_start_idx += hop_size;
    }

    let mut lower_frame_idx: usize = 0;  // The index of the lowest frame we are adding
    let mut upper_frame_idx: usize = 0;  // The index of the highest frame we are adding
    let mut current_sample_idx: usize = 0;  // The index of the current sample to compute

    // Overlap add
    while lower_frame_idx < audio_chunks.len() {
        // If we've moved beyond the range of the lower frame, we need to move the lower frame index up
        if current_sample_idx >= frame_indices[lower_frame_idx].1 {
            lower_frame_idx += 1;
        }

        // If we've moved into the range of a new upper frame, we need to adjust the upper frame index
        if upper_frame_idx + 1 < audio_chunks.len() {
            if current_sample_idx >= frame_indices[upper_frame_idx + 1].0 {
                upper_frame_idx += 1;
            }
        }

        // Check to make sure the lower frame index is still valid (i.e. we haven't gone beyond the end of the audio)
        if lower_frame_idx < audio_chunks.len() {
            // Build the sample using only the valid frames
            let mut sample: f64 = 0.0;
            for i in lower_frame_idx..upper_frame_idx + 1 {
                let local_frame_idx = current_sample_idx - frame_indices[i].0;
                sample += audio_chunks[i][local_frame_idx];
            }
            audio.push(sample);
            current_sample_idx += 1;
        }
    }

    audio
}
