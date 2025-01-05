//! # Multithreaded tools
//! The `mp` module contains multithreaded tools.

use crate::analysis::{Analysis, analyzer};
use crate::spectrum;
use std::thread;
use threadpool::ThreadPool;
use std::sync::mpsc;
use num::Complex;
use crate::WindowType;


/// A thread-pool STFT analyzer using the tools in the analysis crate.
/// 
/// If None or 0 is provided for the `max_num_threads`, the maximum available number of threads will be used. 
/// This might slow your computer down for other tasks while the analysis is running. If you provide
/// a higher number of threads than your computer supports, the number of threads will be truncated to
/// match what the computer can handle.
/// 
/// # Example:
/// 
/// ```
/// use aus::mp::stft_analysis;
/// let mut audio = aus::read("myfile.wav").unwrap();
/// let analysis = stft_analysis(&mut audio.samples[0], 2048, audio.sample_rate, Some(8));
/// ```
pub fn stft_analysis(audio: &mut Vec<f64>, fft_size: usize, sample_rate: u32, max_num_threads: Option<usize>) -> Vec<Analysis> {
    let max_available_threads = match std::thread::available_parallelism() {
        Ok(x) => x.get(),
        Err(_) => 1
    };
    let pool_size = match max_num_threads {
        Some(x) => {
            if x > max_available_threads || x == 0 {
                max_available_threads
            } else {
                x
            }
        },
        None => max_available_threads
    };

    let rfft_freqs = spectrum::rfftfreq(fft_size, sample_rate);
    let stft_imaginary_spectrum: Vec<Vec<Complex<f64>>> = spectrum::rstft(audio, fft_size, fft_size / 2, crate::WindowType::Hamming);
    let (stft_magnitude_spectrum, _) = spectrum::complex_to_polar_rstft(&stft_imaginary_spectrum);
    
    // Set up the multithreading
    let (tx, rx) = mpsc::channel();  // the message passing channel
    let num_threads: usize = match thread::available_parallelism() {
        Ok(x) => x.get(),
        Err(_) => 1
    };

    // Get the starting STFT frame index for each thread
    let mut thread_start_indices: Vec<usize> = vec![0; num_threads];
    let num_frames_per_thread: usize = f64::ceil(stft_magnitude_spectrum.len() as f64 / num_threads as f64) as usize;
    for i in 0..num_threads {
        thread_start_indices[i] = num_frames_per_thread * i;
    }

    // Run the threads
    let pool = ThreadPool::new(pool_size);
    for i in 0..num_threads {
        let tx_clone = tx.clone();
        let thread_idx = i;
        
        // Copy the fragment of the magnitude spectrum for this thread
        let mut local_magnitude_spectrum: Vec<Vec<f64>> = Vec::with_capacity(num_frames_per_thread);
        let start_idx = i * num_frames_per_thread;
        let end_idx = usize::min(start_idx + num_frames_per_thread, stft_magnitude_spectrum.len());
        for j in start_idx..end_idx {
            let mut rfft_frame: Vec<f64> = Vec::with_capacity(stft_magnitude_spectrum[j].len());
            for k in 0..stft_magnitude_spectrum[j].len() {
                rfft_frame.push(stft_magnitude_spectrum[j][k]);
            }
            local_magnitude_spectrum.push(rfft_frame);
        }

        // Copy other important variables
        let local_sample_rate = sample_rate;
        let local_rfft_freqs = rfft_freqs.clone();

        // Start the thread
        pool.execute(move || {
            let mut analyses: Vec<Analysis> = Vec::with_capacity(local_magnitude_spectrum.len());
            
            // Perform the analyses
            for j in 0..local_magnitude_spectrum.len() {
                analyses.push(analyzer(&local_magnitude_spectrum[j], None, local_sample_rate, &local_rfft_freqs))
            }

            let _ = match tx_clone.send((thread_idx, analyses)) {
                Ok(x) => x,
                Err(_) => ()
            };
        });
    }

    // Drop the original sender. Once all senders are dropped, receiving will end automatically.
    drop(tx);

    // Collect the analysis vectors and sort them by thread id
    let mut results = vec![];
    for received_data in rx {
        results.push(received_data);
    }
    results.sort_by_key(|&(index, _)| index);
    
    // let all threads wrap up
    pool.join();

    // Combine the analysis vectors into one big vector
    let mut analyses: Vec<Analysis> = Vec::new();
    for i in 0..results.len() {
        for j in 0..results[i].1.len() {
            analyses.push(results[i].1[j]);
        }
    }

    analyses
}

/// A multithreaded real STFT.
/// 
/// The last rFFT frame will be zero-padded if necessary.
/// This function will return a vector of complex rFFT spectrum frames.
/// 
/// If you plan to use the inverse STFT, you need to make sure that the parameters
/// are set correctly here for reconstruction.
/// a) Make sure you use a good window.
/// b) Choose a good hop size for your window to satisfy the constant overlap-add condition.
///    For the Hanning and Hamming windows, you should use a hop size of 50% of the FFT size.
/// 
/// # Example
/// 
/// ```
/// use aus::{WindowType, mp};
/// use rand::Rng;
/// let mut rng = rand::thread_rng();
/// let fft_size: usize = 2048;
/// let hop_size: usize = fft_size / 2;
/// let window_type = WindowType::Hanning;
/// // 60 seconds of noise
/// let mut pseudo_audio: Vec<f64> = (0..44100 * 60).map(|_| rng.gen_range(-1.0..1.0)).collect();
/// let spectrum = mp::rstft(&pseudo_audio, fft_size, hop_size, window_type, Some(4));
/// ```
pub fn rstft(audio: &[f64], fft_size: usize, hop_size: usize, window_type: WindowType, max_num_threads: Option<usize>) -> Vec<Vec<Complex<f64>>> {
    let max_available_threads = match std::thread::available_parallelism() {
        Ok(x) => x.get(),
        Err(_) => 1
    };
    let pool_size = match max_num_threads {
        Some(x) => {
            if x > max_available_threads || x == 0 {
                max_available_threads
            } else {
                x
            }
        },
        None => max_available_threads
    };

    // Get the start and end points for each thread
    let mut thread_start: Vec<usize> = vec![0; pool_size];
    let mut thread_end: Vec<usize> = vec![0; pool_size];
    let mut samples_per_thread = audio.len() / pool_size;
    let mut i = 1;
    loop {
        if i > samples_per_thread {
            samples_per_thread = i / 2;
            break;
        }
        i *= 2;
    }
    for i in 0..pool_size {
        thread_start[i] = samples_per_thread * i;
        thread_end[i] = samples_per_thread * (i + 1);
    }
    thread_end[pool_size-1] = audio.len();

    let mut spectrogram: Vec<Vec<Complex<f64>>> = Vec::new();
    
    // Set up the multithreading
    let (tx, rx) = mpsc::channel();  // the message passing channel
    let pool = ThreadPool::new(pool_size);
    for i in 0..pool_size {
        let tx_clone = tx.clone();
        let thread_idx = i;
        let mut local_audio: Vec<f64> = Vec::with_capacity(samples_per_thread);
        for j in thread_start[i]..thread_end[i] {
            local_audio.push(audio[j]);
        }
        let local_fft_size = fft_size;
        let local_hop_size = hop_size;
        let local_window_type = window_type;

        // Start the thread
        pool.execute(move || {
            let local_spectrogram = spectrum::rstft(&local_audio, local_fft_size, local_hop_size, local_window_type);
            let _ = match tx_clone.send((thread_idx, local_spectrogram)) {
                Ok(x) => x,
                Err(_) => ()
            };
        });
    }

    // Drop the original sender. Once all senders are dropped, receiving will end automatically.
    drop(tx);

    // Collect the analysis vectors and sort them by thread id
    let mut results = vec![];
    for received_data in rx {
        results.push(received_data);
    }
    results.sort_by_key(|&(index, _)| index);
    
    // let all threads wrap up
    pool.join();

    // Combine the analysis vectors into one big vector
    for i in 0..results.len() {
        for j in 0..results[i].1.len() {
            spectrogram.push(results[i].1[j].clone());
        }
    }
    
    // let all threads wrap up
    pool.join();

    spectrogram
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::spectrum::irstft;
    const AUDIO: &str = "D:/Recording/tests/grains.wav";
    const FFT_SIZE: usize = 2048;
    
    /// Test multithreaded STFT
    #[test]
    fn mp_basic_tests1() {
        let path = String::from(AUDIO);
        let audio = match crate::read(&path) {
            Ok(x) => x,
            Err(_) => panic!("could not read audio")
        };
        let imaginary_spectrogram = rstft(&audio.samples[0], FFT_SIZE, FFT_SIZE / 2, WindowType::Hamming, Some(8));
        let new_audio = irstft(&imaginary_spectrogram, FFT_SIZE, FFT_SIZE / 2, WindowType::Hamming).unwrap();
        let new_audio_file = crate::AudioFile::new_mono(audio.audio_format, audio.sample_rate, new_audio);
        crate::write("D:/Recording/tests/graintest.wav", &new_audio_file).unwrap();
    }

    /// Test multithreaded spectral analyzer
    #[test]
    fn mp_basic_tests2() {
        let path = String::from(AUDIO);
        let mut audio = match crate::read(&path) {
            Ok(x) => x,
            Err(_) => panic!("could not read audio")
        };
        let _ = stft_analysis(&mut audio.samples[0], FFT_SIZE, audio.sample_rate, Some(8));
    }
    
}
