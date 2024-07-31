//! # MP
//! The `mp` module contains multithreaded tools.

use crate::analysis::{Analysis, analyzer};
use crate::spectrum;
use std::thread;
use threadpool::ThreadPool;
use std::sync::mpsc;
use num::Complex;


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
        let local_fft_size = fft_size;
        let local_sample_rate = sample_rate;

        // Start the thread
        pool.execute(move || {
            let mut analyses: Vec<Analysis> = Vec::with_capacity(local_magnitude_spectrum.len());
            
            // Perform the analyses
            for j in 0..local_magnitude_spectrum.len() {
                analyses.push(analyzer(&local_magnitude_spectrum[j], local_fft_size, local_sample_rate))
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

#[cfg(test)]
mod tests {
    use super::*;
    const AUDIO: &str = "D:/Recording/tests/grains.wav";
    
    /// Test multithreaded spectral analyzer
    #[test]
    fn basic_tests6() {
        let fft_size: usize = 4096;
        let path = String::from(AUDIO);
        let mut audio = match crate::read(&path) {
            Ok(x) => x,
            Err(_) => panic!("could not read audio")
        };
        let _ = stft_analysis(&mut audio.samples[0], fft_size, audio.sample_rate, Some(8));
    }
}
