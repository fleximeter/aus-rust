/// File: fft.rs
/// 
/// This file contains FFT abstraction functions based on the rustfft crate.
/// It has a rFFT/IrFFT pair and a rSTFT/IrSTFT pair.
/// 
/// Originally the code was based on the realfft crate, but ran into a problem
/// where spectral editing could result in errors when performing the inverse
/// transform because of the presence of imaginary residue, so instead the 
/// rustfft crate is used.

use rustfft::{FftPlanner, num_complex::Complex};
use super::fft_tools::overlap_add;
use super::window::{WindowType, generate_window};

/// Represents all possible errors that could happen in spectrum processing
#[derive(Debug, Clone)]
pub struct SpectrumError {
    pub error_msg: String
}

/// Calculates the real FFT of a chunk of audio.
/// 
/// The input audio must be a 1D vector of size fft_size.
/// The function will generate an error if the audio vector length is wrong.
/// If you want to zero-pad your audio, you will need to do it before running this function.
/// Returns the complex spectrum.
pub fn rfft(audio: &[f64], fft_size: usize) -> Vec<Complex<f64>> {
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);
    
    let mut spectrum: Vec<Complex<f64>> = Vec::with_capacity(audio.len());
    for i in 0..audio.len() {
        spectrum.push(Complex{re: audio[i], im: 0.0});
    }
    fft.process(&mut spectrum);
    spectrum[..fft_size / 2 + 1].to_vec()
}

/// Calculates the inverse real FFT of an audio spectrum.
/// 
/// The input spectrum must be a 1D vector of size fft_size / 2 + 1. The function will
/// generate an error if the spectrum vector length is wrong.
/// Returns the audio vector.
pub fn irfft(spectrum: &[Complex<f64>], fft_size: usize) -> Vec<f64> {
    let mut planner = FftPlanner::new();
    let ifft = planner.plan_fft_inverse(fft_size);
    
    // We'll use a separate vector for running the FFT.
    let mut spectrum_input: Vec<Complex<f64>> = Vec::with_capacity(fft_size);
    for i in 0..spectrum.len() {
        spectrum_input.push(spectrum[i]);
    }

    // Add the negative frequencies back
    for i in 0..fft_size / 2 - 1 {
        spectrum_input.push(spectrum[fft_size / 2 - 1 - i].conj());
    }

    assert_eq!(spectrum_input.len(), fft_size);
    ifft.process(&mut spectrum_input);

    // Copy to the audio buffer, discarding any imaginary component
    let mut audio: Vec<f64> = Vec::with_capacity(fft_size);
    for i in 0..spectrum_input.len() {
        audio.push(spectrum_input[i].re);
    }
    audio
}

/// Calculates the real STFT of a chunk of audio.
/// 
/// The last rFFT frame will be zero-padded if necessary.
/// This function will return a vector of complex rFFT spectrum frames.
/// 
/// If you plan to use the inverse STFT, you need to make sure that the parameters
/// are set correctly here for reconstruction.
/// a) Make sure you use a good window.
/// b) Choose a good hop size for your window to satisfy the constant overlap-add condition.
///    For the Hanning and Hamming windows, you should use a hop size of 50% of the FFT size.
pub fn rstft(audio: &mut Vec<f64>, fft_size: usize, hop_size: usize, window_type: WindowType) -> Vec<Vec<Complex<f64>>> {
    let mut planner: FftPlanner<f64> = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);
    let mut spectrogram: Vec<Vec<Complex<f64>>> = Vec::new();
    let window = generate_window(window_type, fft_size);

    // Track the current chunk index
    let mut hop_idx = 0;
    let mut finished = false;
    while !finished {
        // Compute the start and end indices for this FFT chunk
        let start_idx = hop_idx * hop_size;
        let end_idx = if hop_idx * hop_size + fft_size < audio.len() {
            hop_idx * hop_size + fft_size
        } else {
            audio.len()
        };

        // If we have reached the end of the audio, we need to flag that and check
        // to see if we need to zero-pad the last chunk
        let num_zeros = if end_idx == audio.len() {
            finished = true;
            start_idx + fft_size - end_idx
        } else {
            0
        };

        // Apply the window to the frame of samples
        // Use a smaller window for the last frame if necessary
        let mut fft_data = if num_zeros > 0 {
            let window = generate_window(window_type, end_idx - start_idx);
            let mut input = {
                let audio_chunk_len = end_idx - start_idx;
                let mut audio_chunk: Vec<Complex<f64>> = Vec::with_capacity(audio_chunk_len);
                for i in 0..audio_chunk_len {
                    audio_chunk.push(Complex{re: audio[start_idx..end_idx][i] * window[i], im: 0.0});
                }
                audio_chunk
            };
            input.extend(vec![Complex{re: 0.0, im: 0.0}; num_zeros]);
            input
        } else {
            let audio_chunk_len = end_idx - start_idx;
            let mut audio_chunk: Vec<Complex<f64>> = Vec::with_capacity(audio_chunk_len);
            for i in 0..audio_chunk_len {
                audio_chunk.push(Complex{re: audio[start_idx..end_idx][i] * window[i], im: 0.0});
            }
            audio_chunk
        };
        
        //println!("fft data len: {}", fft_data.len());
        // Process the FFT for this audio chunk, and push it onto the output vector.
        fft.process(&mut fft_data);
        spectrogram.push(fft_data[..fft_size / 2 + 1].to_vec());

        // Move to the next audio chunk
        hop_idx += 1;
    }
    spectrogram
}

/// Calculates the inverse real STFT of a chunk of audio.
/// 
/// This function requires all of the spectrogram frames to be of length fft_size / 2 + 1.
/// If any of the frames has the wrong length, the function will return a FftError.
///  
/// Note: For the STFT/ISTFT process to work correctly, you need to follow these guidelines:
///       a) Use the same window type for the STFT and ISTFT.
///       b) Choose an appropriate hop size for the window type to satisfy the constant overlap-add condition.
///          This is 50% of the FFT size for the Hanning and Hamming windows.
pub fn irstft(spectrogram: &Vec<Vec<Complex<f64>>>, fft_size: usize, hop_size: usize, window_type: WindowType) -> Vec<f64> {
    let mut planner: FftPlanner<f64> = FftPlanner::new();
    let fft = planner.plan_fft_inverse(fft_size);
    let num_stft_frames = spectrogram.len();
    let num_output_frames = fft_size + (hop_size * (num_stft_frames - 1));
    let mut audio_chunks: Vec<Vec<f64>> = vec![Vec::with_capacity(fft_size); num_stft_frames];
    let mut window_norm: Vec<f64> = vec![0.0; num_output_frames];

    // Get the window
    let window_samples = generate_window(window_type, fft_size);

    // Perform IRFFT on each STFT frame
    for i in 0..num_stft_frames {
        // We'll use a separate vector for running the FFT.
        let mut spectrum_input: Vec<Complex<f64>> = Vec::with_capacity(fft_size);
        for j in 0..spectrogram[i].len() {
            spectrum_input.push(spectrogram[i][j]);
        }

        // Add the negative frequencies back
        for j in 0..fft_size / 2 - 1 {
            spectrum_input.push(spectrogram[i][fft_size / 2 - 1 - j].conj());
        }
        
        fft.process(&mut spectrum_input);

        // window the samples
        for j in 0..spectrum_input.len() {
            audio_chunks[i].push(spectrum_input[j].re * window_samples[j]);
        }

        // Compute the window norm for the current sample
        let start_idx = i * hop_size;
        let end_idx = start_idx + fft_size;
        for j in start_idx..end_idx {
            window_norm[j] += window_samples[j - start_idx].powf(2.0);
        }
    }

    // Overlap add the remaining chunks
    let mut audio = overlap_add(&audio_chunks, fft_size, hop_size);

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
