// File: spectrum.rs
// This file contains spectral resource functionality.

use realfft::{FftError, RealFftPlanner};
use num::Complex;
use symphonia::core::dsp::fft::Fft;
use std::{collections::HashMap, f64::consts::PI};
use rand::Rng;

/// Represents a window type
#[derive(Copy, Clone)]
pub enum WindowType{
    Bartlett,
    Blackman,
    Hanning,
    Hamming
}

/// Creates a Bartlett window of size m
#[inline(always)]
pub fn generate_window_bartlett(window_length: usize) -> Vec<f64>{
    let mut window: Vec<f64> = vec![0.0; window_length];
    for i in 0..window_length {
        window[i] = (2.0 / (window_length as f64 - 1.0)) * ((window_length as f64 - 1.0) / 2.0 - f64::abs(i as f64 - (window_length as f64 - 1.0) / 2.0));
    }
    window
}

/// Creates a Blackman window of size m
#[inline(always)]
pub fn generate_window_blackman(window_length: usize) -> Vec<f64>{
    let mut window: Vec<f64> = vec![0.0; window_length];
    for i in 0..window_length {
        window[i] = 0.42 - 0.5 * f64::cos((2.0 * std::f64::consts::PI * i as f64) / (window_length as f64)) 
            + 0.08 * f64::cos((4.0 * std::f64::consts::PI * i as f64) / (window_length as f64));
    }
    window
}

/// Creates a Hanning window of size m
#[inline(always)]
pub fn generate_window_hanning(window_length: usize) -> Vec<f64>{
    let mut window: Vec<f64> = vec![0.0; window_length];
    for i in 0..window_length {
        window[i] = 0.5 - 0.5 * f64::cos((2.0 * std::f64::consts::PI * i as f64) / (window_length as f64 - 1.0));
    }
    window
}

/// Creates a Hamming window of size m
#[inline(always)]
pub fn generate_window_hamming(window_length: usize) -> Vec<f64>{
    let mut window: Vec<f64> = vec![0.0; window_length];
    for i in 0..window_length {
        window[i] = 0.54 - 0.46 * f64::cos((2.0 * std::f64::consts::PI * i as f64) / (window_length as f64 - 1.0));
    }
    window
}

/// Gets the corresponding window for a provided WindowType and window size
#[inline(always)]
pub fn generate_window(window_type: WindowType, window_length: usize) -> Vec<f64> {
    match &window_type {
        WindowType::Bartlett => generate_window_bartlett(window_length),
        WindowType::Blackman => generate_window_blackman(window_length),
        WindowType::Hanning => generate_window_hanning(window_length),
        WindowType::Hamming => generate_window_hamming(window_length),
    }
}

/// Checks to see if a number is a power of two. Used for verifying FFT size.
#[inline(always)]
fn is_power_of_two(val: usize) -> bool {
    let log = f64::log2(val as f64) as usize;
    if 2 << log == val {
        true
    } else {
        false
    }
}

/// Calculates the real FFT of a chunk of audio.
/// 
/// The input audio must be a 1D vector of size fft_size.
/// The function will generate an error if the audio vector length is wrong.
/// If you want to zero-pad your audio, you will need to do it before running this function.
/// Returns the complex spectrum.
pub fn rfft(audio: &mut [f64], fft_size: usize) -> Result<Vec<Complex<f64>>, realfft::FftError> {
    let mut real_planner = RealFftPlanner::<f64>::new();
    let r2c = real_planner.plan_fft_forward(fft_size);
    let mut spectrum: Vec<num::Complex<f64>> = r2c.make_output_vec();
    if audio.len() != fft_size {
        return Err(realfft::FftError::InputBuffer(fft_size, audio.len()));
    }
    match r2c.process(audio, &mut spectrum) {
        Ok(_) => (),
        Err(err) => return Err(err)
    }
    Ok(spectrum)
}

/// Calculates the inverse real FFT of an audio spectrum.
/// 
/// The input spectrum must be a 1D vector of size fft_size / 2 + 1. The function will
/// generate an error if the spectrum vector length is wrong.
/// Returns the audio vector.
pub fn irfft(spectrum: &mut [Complex<f64>], fft_size: usize) -> Result<Vec<f64>, realfft::FftError> {
    let mut real_planner = RealFftPlanner::<f64>::new();
    let c2r = real_planner.plan_fft_inverse(fft_size);
    let mut audio: Vec<f64> = c2r.make_output_vec();
    if spectrum.len() != fft_size / 2 + 1 {
        return Err(realfft::FftError::InputBuffer(fft_size, spectrum.len()));
    }
    match c2r.process(spectrum, &mut audio) {
        Ok(_) => (),
        Err(err) => return Err(err)
    }
    Ok(audio)
}

/// This function decomposes a complex spectrum into magnitude and phase spectra. 
pub fn complex_to_polar_rfft(spectrum: Vec<Complex<f64>>) -> (Vec<f64>, Vec<f64>) {
    let mut magnitude_spectrum = vec![0.0 as f64; spectrum.len()];
    let mut phase_spectrum = vec![0.0 as f64; spectrum.len()];
    for i in 0..spectrum.len() {
        magnitude_spectrum[i] = f64::sqrt(f64::powf(spectrum[i].re, 2.0) + f64::powf(spectrum[i].im, 2.0));
        phase_spectrum[i] = f64::atan2(spectrum[i].im, spectrum[i].re);
    }
    (magnitude_spectrum, phase_spectrum)
}

/// This function decomposes a complex spectrum into magnitude and phase spectra. 
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

/// This function combines magnitude and phase spectra into a complex spectrum for the inverse real FFT.
/// The magnitude and phase spectra must have the same length. If they do not have the same length,
/// this function will return a FftError.
pub fn polar_to_complex_rfft(magnitude_spectrum: Vec<f64>, phase_spectrum: Vec<f64>) -> Result<Vec<Complex<f64>>, FftError> {
    if magnitude_spectrum.len() != phase_spectrum.len() {
        return Err(FftError::InputBuffer(magnitude_spectrum.len(), phase_spectrum.len()));
    }
    let mut spectrum = vec![num::complex::Complex::new(0.0, 0.0); magnitude_spectrum.len()];    
    for i in 0..magnitude_spectrum.len() {
        let real = f64::cos(phase_spectrum[i]) * magnitude_spectrum[i];
        let imag = f64::sin(phase_spectrum[i]) * magnitude_spectrum[i];
        spectrum[i] = num::complex::Complex::new(real, imag);
    }
    Ok(spectrum)
}

/// This function combines magnitude and phase spectra into a complex spectrum for the inverse real STFT.
/// The magnitude and phase spectra must have the same length. If they do not have the same length,
/// this function will return a FftError.
pub fn polar_to_complex_rstft(magnitude_spectrum: Vec<Vec<f64>>, phase_spectrum: Vec<Vec<f64>>) -> Result<Vec<Vec<Complex<f64>>>, FftError> {
    if magnitude_spectrum.len() != phase_spectrum.len() {
        return Err(FftError::InputBuffer(magnitude_spectrum.len(), phase_spectrum.len()));
    }
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
    Ok(spectrum)
}

/// Gets the corresponding frequencies for rFFT data
#[inline(always)]
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
/// The last rFFT frame will be zero-padded if necessary.
/// This function will return a vector of complex rFFT spectrum frames.
/// 
/// If you plan to use the inverse STFT, you need to make sure that the parameters
/// are set correctly here for reconstruction.
/// a) Make sure you use a good window.
/// b) Choose a good hop size for your window to satisfy the constant overlap-add condition.
///    For the Hanning and Hamming windows, you should use a hop size of 50% of the FFT size.
pub fn rstft(audio: &mut Vec<f64>, fft_size: usize, hop_size: usize, window_type: WindowType) -> Result<Vec<Vec<Complex<f64>>>, realfft::FftError> {
    let mut real_planner = RealFftPlanner::<f64>::new();
    let r2c = real_planner.plan_fft_forward(fft_size);
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
        let mut fft_input = if num_zeros > 0 {
            let window = generate_window(window_type, end_idx - start_idx);
            let mut input = {
                let mut audio_chunk = audio[start_idx..end_idx].to_vec();
                for i in 0..audio_chunk.len() {
                    audio_chunk[i] *= window[i];
                }
                audio_chunk
            };
            input.extend(vec![0.0; num_zeros]);
            input
        } else {
            let input = {
                let mut audio_chunk = audio[start_idx..end_idx].to_vec();
                for i in 0..audio_chunk.len() {
                    audio_chunk[i] *= window[i];
                }
                audio_chunk
            };
            input
        };

        // prepare the output complex vector and check that the sizes are correct
        let mut spectrum = r2c.make_output_vec();
        if fft_input.len() != fft_size {
            return Err(realfft::FftError::InputBuffer(fft_size, fft_input.len()));
        }
        
        // Process the FFT for this audio chunk, and push it onto the output vector.
        // Track if an error occurs.
        match r2c.process(&mut fft_input, &mut spectrum) {
            Ok(_) => (),
            Err(err) => return Err(err)
        };
        
        spectrogram.push(spectrum);

        // Move to the next audio chunk
        hop_idx += 1;
    }
    Ok(spectrogram)
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
pub fn irstft(spectrogram: &mut Vec<Vec<Complex<f64>>>, fft_size: usize, hop_size: usize, window_type: WindowType) -> Result<Vec<f64>, realfft::FftError> {
    let mut real_planner = RealFftPlanner::<f64>::new();
    let c2r = real_planner.plan_fft_inverse(fft_size);
    let num_stft_frames = spectrogram.len();
    let num_output_frames = fft_size + (hop_size * (num_stft_frames - 1));
    let mut audio_chunks: Vec<Vec<f64>> = vec![Vec::with_capacity(fft_size); num_stft_frames];
    let mut window_norm: Vec<f64> = vec![0.0; num_output_frames];

    // Get the window
    let window_samples = generate_window(window_type, fft_size);

    // Perform IRFFT on each STFT frame
    for i in 0..num_stft_frames {
        let mut audio_frame = c2r.make_output_vec();
        if spectrogram[i].len() != fft_size / 2 + 1 {
            return Err(realfft::FftError::InputBuffer(fft_size / 2 + 1, spectrogram[i].len()));
        }
        match c2r.process(&mut spectrogram[i], &mut audio_frame) {
            Ok(_) => (),
            Err(err) => return Err(err)
        }

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
    
    Ok(audio)
}

/// An efficient overlap add mechanism.
/// 
/// The formula for overlap add in mathematical definitions of the ISTFT requires checking
/// each of the M audio chunks generated by the M IRFFT operations. For short audio, this
/// is not particularly costly, but as FFT sizes decrease and audio length increases, it
/// becomes wasteful.
/// 
/// This algorithm keeps track of which audio chunks are relevant to computing the current 
/// sample, so that only those chunks are consulted. It does this by using running indices
/// for the lowest and highest audio chunk indices that are currently relevant.
/// 
/// The algorithm assumes that the hop size is greater than 0.
fn overlap_add(audio_chunks: &Vec<Vec<f64>>, fft_size: usize, hop_size: usize) -> Vec<f64> {
    let mut audio: Vec<f64> = Vec::new();

    // Get the global start and end index corresponding to each audio frame
    let mut frame_indices: Vec<(usize, usize)> = Vec::with_capacity(audio_chunks.len());
    let mut current_frame_start_idx: usize = 0;
    for _ in 0..audio_chunks.len() {
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


////////////////////////////////////////////////////////////////////////////////////////////////
/// SPECTRAL TOOLS
/// 
/// The following functions are tools for processing spectral data.
////////////////////////////////////////////////////////////////////////////////////////////////

/// Exchanges frames in a STFT spectrum.
/// Each frame is swapped with the frame *hop* steps ahead or *hop* steps behind.
pub fn exchange_frames(data: &mut [Vec<Complex<f64>>], hop: usize) {
    let end_idx = data.len() - data.len() % (hop * 2);
    let step = hop * 2;
    for i in (0..end_idx).step_by(step) {
        for j in i..i+hop {
            // swap each FFT bin in the frame
            for k in 0..data[j].len() {
                let temp = data[j][k];
                data[j][k] = data[j + hop][k];
                data[j + hop][k] = temp;    
            }
        }
    }
}

/// Stochastically exchanges STFT frames in a complex spectrogram.
/// Each frame is swapped with the frame up to *hop* steps ahead or *hop* steps behind. 
pub fn exchange_frames_stochastic(data: &mut [Vec<Complex<f64>>], max_hop: usize) {
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

            // Choose a random index to swap with, and perform the swap for each FFT bin
            let swap_idx = rand::thread_rng().gen_range(0..possible_indices.len());
            for i in 0..data[idx].len() {
                let temp = data[idx][i];
                data[idx][i] = data[swap_idx][i];
                data[swap_idx][i] = temp;
            }
            
            // Record that the swap index has been used
            future_indices.insert(swap_idx, true);
        }
        idx += 1;
    }
}

/// Implements a spectral "freeze" where a single FFT spectrum is extended over time.
/// You will need to specify the hop size that you expect to use with the IrSTFT, as well
/// as the number of frames for which to freeze the spectrum.
/// The function will calculate phase angle differences as part of the freeze.
pub fn spectral_freeze(magnitude_spectrum: &Vec<f64>, phase_spectrum: &Vec<f64>, num_frames: usize, hop_size: usize) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut current_phases: Vec<f64> = vec![0.0; phase_spectrum.len()];
    let mut phase_differences: Vec<f64> = vec![0.0; phase_spectrum.len()];
    let mut stft_magnitudes: Vec<Vec<f64>> = Vec::with_capacity(num_frames);
    let mut stft_phases: Vec<Vec<f64>> = Vec::with_capacity(num_frames);

    // Compute the first frame in the output STFT spectrum, as well as computing phase differences
    let mut frame1_mag: Vec<f64> = Vec::with_capacity(magnitude_spectrum.len());
    let mut frame1_phase: Vec<f64> = Vec::with_capacity(magnitude_spectrum.len());
    for i in 0..magnitude_spectrum.len() {
        frame1_mag.push(magnitude_spectrum[i]);

        // We cannot have phase for 0.0 Hz
        if i == 0 {
            frame1_phase.push(0.0);
        } else {
            frame1_phase.push(phase_spectrum[i]);
            current_phases[i] = phase_spectrum[i];
            let num_periods: f64 = (i * hop_size) as f64 / magnitude_spectrum.len() as f64;
            phase_differences[i] = num_periods.fract() * 2.0 * PI;
        }
    }
    stft_magnitudes.push(frame1_mag);
    stft_phases.push(frame1_phase);
    
    // Compute all other frames
    for _ in 1..num_frames {
        let mut frame_mag: Vec<f64> = Vec::with_capacity(magnitude_spectrum.len());
        let mut frame_phase: Vec<f64> = Vec::with_capacity(magnitude_spectrum.len());
        for i in 0..magnitude_spectrum.len() {
            frame_mag.push(magnitude_spectrum[i]);
            
            if i == 0 {
                frame_phase.push(0.0);
            } else {
                // Compute phase for current frame and FFT bin. The phase will be scaled to between -pi and +pi.
                let mut phase = current_phases[i] + phase_differences[i];
                while phase > PI {
                    phase -= 2.0 * PI;
                }
                frame_phase.push(phase);
                current_phases[i] = phase;
            }
            // Compute phase for current frame and FFT bin. The phase will be scaled to between -pi and +pi.
            let mut phase = current_phases[i] + phase_differences[i];
            while phase > PI {
                phase -= 2.0 * PI;
            }
            frame_phase.push(phase);
        }
        stft_magnitudes.push(frame_mag);
        stft_phases.push(frame_phase);
    }

    (stft_magnitudes, stft_phases)
}
