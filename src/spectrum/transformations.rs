// File: transformations.rs
// 
// This file contains spectral transformations such as a convolution abstraction
// and spectral frame swapping.

use std::collections::HashMap;
use rand::Rng;
use fft_convolver::FFTConvolver;
use super::fft::SpectrumError;
use std::f64::consts::PI;


/// Performs partitioned FFT convolution using the `fft_convolver` crate.
/// 
/// A good block size is 16. Note that the output will be truncated to the
/// length of the audio1 vector, so if you want the reverb tail to last longer,
/// you will need to zero-pad the audio.
/// 
/// This is a convenience function - if you want to perform multiple convolutions
/// with the same impulse response, it is better to use the code directly rather
/// than calling this function, which has to recreate the convolver and load
/// its impulse response every time you call it.
pub fn partitioned_convolution(audio1: &mut Vec<f64>, audio2: &mut Vec<f64>, block_size: usize) -> Result<Vec<f64>, SpectrumError> {
    let mut convolver = FFTConvolver::default();
    let mut output: Vec<f64> = vec![0.0; audio1.len()];
    match convolver.init(block_size, &audio2) {
        Ok(()) => (),
        Err(err) => {
            return Err(SpectrumError{error_msg: err.to_string()});
        }
    }
    match convolver.process(&audio1, &mut output) {
        Ok(()) => (),
        Err(err) => {
            return Err(SpectrumError{error_msg: err.to_string()});
        }
    }
    Ok(output)
}

////////////////////////////////////////////////////////////////////////////////////////////////
// SPECTRAL TOOLS
// 
// The following functions are tools for processing spectral data.
////////////////////////////////////////////////////////////////////////////////////////////////

/// Exchanges frames in a STFT spectrum.
/// Each frame is swapped with the frame *hop* steps ahead or *hop* steps behind.
/// 
/// # Example
/// 
/// ```
/// use aus::spectrum;
/// let file = aus::read("myfile.wav").unwrap();
/// let mut imaginary_spectrogram = spectrum::rstft(&file.samples[0], 2048, 1024, aus::WindowType::Hanning);
/// let (mut magnitude_spectrogram, mut phase_spectrogram) = spectrum::complex_to_polar_rstft(&imaginary_spectrogram);
/// spectrum::stft_exchange_frames(&mut magnitude_spectrogram, &mut phase_spectrogram, 10);
/// let new_imaginary_spectrogram = spectrum::polar_to_complex_rstft(&magnitude_spectrogram, &phase_spectrogram).unwrap();
/// let new_audio = spectrum::irstft(&new_imaginary_spectrogram, 2048, 1024, aus::WindowType::Hanning).unwrap();
/// let output_file = aus::AudioFile::new_mono(aus::AudioFormat::S24, file.sample_rate, new_audio);
/// aus::write("myfile2.wav", &output_file);
/// ```
pub fn stft_exchange_frames(magnitude_spectrogram: &mut [Vec<f64>], phase_spectrogram: &mut [Vec<f64>], hop: usize) {
    let end_idx = magnitude_spectrogram.len() - magnitude_spectrogram.len() % (hop * 2);
    let step = hop * 2;
    for i in (0..end_idx).step_by(step) {
        for j in i..i+hop {
            // swap each FFT bin in the frame
            for k in 0..magnitude_spectrogram[j].len() {
                let temp_magnitude = magnitude_spectrogram[j][k];
                let temp_phase = phase_spectrogram[j][k];
                magnitude_spectrogram[j][k] = magnitude_spectrogram[j + hop][k];
                magnitude_spectrogram[j + hop][k] = temp_magnitude;
                phase_spectrogram[j][k] = phase_spectrogram[j + hop][k];
                phase_spectrogram[j + hop][k] = temp_phase;
            }
        }
    }
}

/// Stochastically exchanges STFT frames in a complex spectrogram.
/// Each frame is swapped with the frame up to *hop* steps ahead or *hop* steps behind. 
/// 
/// # Example
/// 
/// ```
/// use aus::spectrum;
/// let file = aus::read("myfile.wav").unwrap();
/// let mut imaginary_spectrogram = spectrum::rstft(&file.samples[0], 2048, 1024, aus::WindowType::Hanning);
/// let (mut magnitude_spectrogram, mut phase_spectrogram) = spectrum::complex_to_polar_rstft(&imaginary_spectrogram);
/// spectrum::stft_exchange_frames_stochastic(&mut magnitude_spectrogram, &mut phase_spectrogram, 20);
/// let new_imaginary_spectrogram = spectrum::polar_to_complex_rstft(&magnitude_spectrogram, &phase_spectrogram).unwrap();
/// let new_audio = spectrum::irstft(&new_imaginary_spectrogram, 2048, 1024, aus::WindowType::Hanning).unwrap();
/// let output_file = aus::AudioFile::new_mono(aus::AudioFormat::S24, file.sample_rate, new_audio);
/// aus::write("myfile2.wav", &output_file);
/// ```
pub fn stft_exchange_frames_stochastic(magnitude_spectrogram: &mut [Vec<f64>], phase_spectrogram: &mut [Vec<f64>], max_hop: usize) {
    let mut future_indices: HashMap<usize, bool> = HashMap::with_capacity(magnitude_spectrogram.len());
    let mut idx = 0;
    while idx < magnitude_spectrogram.len() {
        // If *idx* is not in the list of future indices which have already been swapped,
        // we can try to swap it with something.
        if !future_indices.contains_key(&idx) {
            // Generate a vector of possible indices in the future with which we could swap this index
            let mut possible_indices: Vec<usize> = Vec::new();
            for i in idx..usize::min(idx + max_hop, magnitude_spectrogram.len()) {
                if !future_indices.contains_key(&i) {
                    possible_indices.push(i);
                }
            }

            // Choose a random index to swap with, and perform the swap for each FFT bin
            let swap_idx = rand::thread_rng().gen_range(0..possible_indices.len());
            for i in 0..magnitude_spectrogram[idx].len() {
                let temp_magnitude = magnitude_spectrogram[idx][i];
                let temp_phase = phase_spectrogram[idx][i];
                magnitude_spectrogram[idx][i] = magnitude_spectrogram[swap_idx][i];
                magnitude_spectrogram[swap_idx][i] = temp_magnitude;
                phase_spectrogram[idx][i] = phase_spectrogram[swap_idx][i];
                phase_spectrogram[swap_idx][i] = temp_phase;
            }
            
            // Record that the swap index has been used
            future_indices.insert(swap_idx, true);
        }
        idx += 1;
    }
}

/// Exchanges bins in a FFT spectrum.
/// Each bin is swapped with the bin `hop` steps above or `hop` steps below.
/// 
/// # Example
/// 
/// ```
/// use aus::spectrum;
/// let fft_size: usize = 2048;
/// let audio = aus::read("myfile.wav").unwrap();
/// let window = aus::generate_window_hanning(fft_size);
/// // Just choose the first 2048 samples in the audio file. This might be a problem, because those samples might be zeros.
/// let audio_chunk: Vec<f64> = audio.samples[0][..fft_size].iter().zip(window.iter()).map(|(a, b)| a * b).collect();
/// let (mut magnitude_spectrum, mut phase_spectrum) = spectrum::complex_to_polar_rfft(&spectrum::rfft(&audio_chunk, fft_size));
/// spectrum::fft_exchange_bins(&mut magnitude_spectrum, &mut phase_spectrum, 20);
/// ```
pub fn fft_exchange_bins(magnitude_spectrum: &mut [f64], phase_spectrum: &mut [f64], hop: usize) {
    let end_idx = magnitude_spectrum.len() - magnitude_spectrum.len() % (hop * 2);
    let step = hop * 2;
    for i in (0..end_idx).step_by(step) {
        for j in i..i+hop {
            let temp_magnitude = magnitude_spectrum[j];
            let temp_phase = phase_spectrum[j];
            magnitude_spectrum[j] = magnitude_spectrum[j + hop];
            magnitude_spectrum[j + hop] = temp_magnitude;    
            phase_spectrum[j] = phase_spectrum[j + hop];
            phase_spectrum[j + hop] = temp_phase;
        }
    }
}

/// Stochastically exchanges bins in a FFT spectrum.
/// Each bin is swapped with the bin up to `max_hop` steps above or `max_hop` steps below.
/// 
/// # Example
/// 
/// ```
/// use aus::spectrum;
/// let fft_size: usize = 2048;
/// let audio = aus::read("myfile.wav").unwrap();
/// let window = aus::generate_window_hanning(fft_size);
/// // Just choose the first 2048 samples in the audio file. This might be a problem, because those samples might be zeros.
/// let audio_chunk: Vec<f64> = audio.samples[0][..fft_size].iter().zip(window.iter()).map(|(a, b)| a * b).collect();
/// let (mut magnitude_spectrum, mut phase_spectrum) = spectrum::complex_to_polar_rfft(&spectrum::rfft(&audio_chunk, fft_size));
/// spectrum::fft_exchange_bins_stochastic(&mut magnitude_spectrum, &mut phase_spectrum, 20);
/// ```
pub fn fft_exchange_bins_stochastic(magnitude_spectrum: &mut [f64], phase_spectrum: &mut [f64], max_hop: usize) {
    let mut future_indices: HashMap<usize, bool> = HashMap::with_capacity(magnitude_spectrum.len());
    let mut idx = 0;
    while idx < magnitude_spectrum.len() {
        // If *idx* is not in the list of future indices which have already been swapped,
        // we can try to swap it with something.
        if !future_indices.contains_key(&idx) {
            // Generate a vector of possible indices in the future with which we could swap this index
            let mut possible_indices: Vec<usize> = Vec::new();
            for i in idx..usize::min(idx + max_hop, magnitude_spectrum.len()) {
                if !future_indices.contains_key(&i) {
                    possible_indices.push(i);
                }
            }

            // Choose a random index to swap with, and perform the swap for each FFT bin
            let swap_idx = rand::thread_rng().gen_range(0..possible_indices.len());
            let temp_magnitude = magnitude_spectrum[idx];
            let temp_phase = phase_spectrum[idx];
            magnitude_spectrum[idx] = magnitude_spectrum[swap_idx];
            magnitude_spectrum[swap_idx] = temp_magnitude;
            phase_spectrum[idx] = phase_spectrum[swap_idx];
            phase_spectrum[swap_idx] = temp_phase;
            
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
/// 
/// # Example
/// 
/// ```
/// use aus::spectrum;
/// let fft_size: usize = 2048;
/// let audio = aus::read("myfile.wav").unwrap();
/// let window = aus::generate_window_hanning(fft_size);
/// // Just choose the first 2048 samples in the audio file. This might be a problem, because those samples might be zeros.
/// let audio_chunk: Vec<f64> = audio.samples[0][..fft_size].iter().zip(window.iter()).map(|(a, b)| a * b).collect();
/// let (magnitude_spectrum, phase_spectrum) = spectrum::complex_to_polar_rfft(&spectrum::rfft(&audio_chunk, fft_size));
/// let (magnitude_spectrogram, phase_spectrogram) = spectrum::fft_freeze(&magnitude_spectrum, &phase_spectrum, 20, fft_size / 2);
/// let new_audio = spectrum::irstft(&spectrum::polar_to_complex_rstft(&magnitude_spectrogram, &phase_spectrogram).unwrap(), fft_size, fft_size / 2, aus::WindowType::Hanning);
/// ```
pub fn fft_freeze(magnitude_spectrum: &Vec<f64>, phase_spectrum: &Vec<f64>, num_frames: usize, hop_size: usize) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut current_phases: Vec<f64> = vec![0.0; phase_spectrum.len()];
    let mut phase_differences: Vec<f64> = vec![0.0; phase_spectrum.len()];
    let mut stft_magnitudes: Vec<Vec<f64>> = Vec::with_capacity(num_frames);
    let mut stft_phases: Vec<Vec<f64>> = Vec::with_capacity(num_frames);

    // Compute the first frame in the output STFT spectrum, as well as computing phase differences
    let mut frame1_mag: Vec<f64> = Vec::with_capacity(magnitude_spectrum.len());
    let mut frame1_phase: Vec<f64> = Vec::with_capacity(magnitude_spectrum.len());
    for i in 0..magnitude_spectrum.len() {
        frame1_mag.push(magnitude_spectrum[i]);
        frame1_phase.push(phase_spectrum[i]);
        current_phases[i] = phase_spectrum[i];
        let num_periods: f64 = (i * hop_size) as f64 / magnitude_spectrum.len() as f64;
        phase_differences[i] = num_periods.fract() * 2.0 * PI;
    }
    stft_magnitudes.push(frame1_mag);
    stft_phases.push(frame1_phase);
    
    // Compute all other frames
    for _ in 1..num_frames {
        let mut frame_mag: Vec<f64> = Vec::with_capacity(magnitude_spectrum.len());
        let mut frame_phase: Vec<f64> = Vec::with_capacity(magnitude_spectrum.len());
        for i in 0..magnitude_spectrum.len() {
            frame_mag.push(magnitude_spectrum[i]);
            // Compute phase for current frame and FFT bin. The phase will be scaled to between -pi and +pi.
            let mut phase = current_phases[i] + phase_differences[i];
            while phase > PI {
                phase -= 2.0 * PI;
            }
            frame_phase.push(phase);
            current_phases[i] = phase;
        }
        stft_magnitudes.push(frame_mag);
        stft_phases.push(frame_phase);
    }

    (stft_magnitudes, stft_phases)
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{operations, spectrum};

    #[test]
    /// Test convolution
    fn test_convolution() {
        let fft_size: usize = 2048;

        let audio_path = String::from("D:\\Recording\\Samples\\Iowa\\Cello.arco.mono.2444.1\\samples_ff\\sample_Cello.arco.ff.sulC.C2B2.wav_0.wav");
        let ir_path = String::from("D:\\Recording\\Samples\\Impulse_Responses\\560377__manysounds__1000-liter-wine-tank-plate-spring-impulse.wav");
        let mut audio = match crate::read(&audio_path) {
            Ok(x) => x,
            Err(_) => panic!("could not read audio")
        };
        let mut ir = match crate::read(&ir_path) {
            Ok(x) => x,
            Err(_) => panic!("could not read audio")
        };

        let zeros: Vec<f64> = vec![0.0; ir.samples[0].len()];
        audio.samples[0].extend(&zeros);
        
        let mut output_audio = partitioned_convolution(&mut audio.samples[0], &mut ir.samples[0], fft_size).unwrap();

        // Fade in and out at beginning and end
        operations::adjust_level(&mut output_audio, -6.0);
        operations::fade_in(&mut output_audio, crate::WindowType::Hanning, 10000);
        operations::fade_out(&mut output_audio, crate::WindowType::Hanning, 10000);

        // Make output audio file
        let output_audiofile = crate::AudioFile::new_mono(crate::AudioFormat::S24, 44100, output_audio);
        let path: String = String::from("D:\\Recording\\out8.wav");
        match crate::write(&path, &output_audiofile) {
            Ok(_) => (),
            Err(_) => ()
        }
    }

    /// Test spectral freeze
    #[test]
    fn test_spectral_freeze() {
        let fft_size: usize = 4096;
        let hop_size: usize = fft_size / 2;
        let window_type = crate::WindowType::Hamming;
        let path = String::from("D:\\Recording\\Samples\\Iowa\\Cello.arco.mono.2444.1\\samples_ff\\sample_Cello.arco.ff.sulC.C2B2.wav_0.wav");
        let mut audio = match crate::read(&path) {
            Ok(x) => x,
            Err(_) => panic!("could not read audio")
        };
        let spectrogram = spectrum::rstft(&mut audio.samples[0], fft_size, hop_size, window_type);
        let (mag, phase) = spectrum::complex_to_polar_rstft(&spectrogram);
        let (freeze_mag, freeze_phase) = spectrum::fft_freeze(&mag[8], &phase[8], 50, fft_size / 2);
        let mut freeze_spectrogram = spectrum::polar_to_complex_rstft(&freeze_mag, &freeze_phase).unwrap();
        let mut output_audio = spectrum::irstft(&mut freeze_spectrogram, fft_size, fft_size / 2, crate::WindowType::Hamming).unwrap();

        // Fade in and out at beginning and end
        operations::fade_in(&mut output_audio, crate::WindowType::Hanning, 10000);
        operations::fade_out(&mut output_audio, crate::WindowType::Hanning, 10000);

        // Make output audio file
        let output_audiofile = crate::AudioFile::new_mono(crate::AudioFormat::S24, 44100, output_audio);
        let path: String = String::from("D:\\Recording\\out7.wav");
        match crate::write(&path, &output_audiofile) {
            Ok(_) => (),
            Err(_) => panic!("could not write audio")
        }
    }

    /// Test stochastic exchange with STFT/ISTFT
    #[test]
    fn test_stochastic_exchange() {
        let fft_size: usize = 4096;
        let hop_size: usize = fft_size / 2;
        let window_type = crate::WindowType::Hamming;
        let path = String::from("D:\\Recording\\Samples\\freesound\\creative_commons_0\\wind_chimes\\eq\\217800__minian89__wind_chimes_eq.wav");
        let mut audio = match crate::read(&path) {
            Ok(x) => x,
            Err(_) => panic!("could not read audio")
        };
        crate::mixdown(&mut audio);
        let spectrogram: Vec<Vec<num::Complex<f64>>> = spectrum::rstft(&mut audio.samples[0], fft_size, hop_size, window_type);
        let (mut magnitude_spectrogram, mut phase_spectrogram) = spectrum::complex_to_polar_rstft(&spectrogram);

        // Perform spectral operations here
        spectrum::stft_exchange_frames_stochastic(&mut magnitude_spectrogram, &mut phase_spectrogram, 20);
        
        // Perform ISTFT and add fade in/out
        let output_spectrogram = spectrum::polar_to_complex_rstft(&magnitude_spectrogram, &phase_spectrogram).unwrap();
        let mut output_audio: Vec<f64> = spectrum::irstft(&output_spectrogram, fft_size, hop_size, window_type).unwrap();
        operations::fade_in(&mut output_audio, crate::WindowType::Hanning, 1000);
        operations::fade_out(&mut output_audio, crate::WindowType::Hanning, 1000);

        // Generate the output audio file
        let output_audiofile = crate::AudioFile::new_mono(crate::AudioFormat::S24, 44100, output_audio);
        let path: String = String::from("D:\\Recording\\out5.wav");
        match crate::write(&path, &output_audiofile) {
            Ok(_) => (),
            Err(_) => panic!("could not write audio")
        }
    }



}