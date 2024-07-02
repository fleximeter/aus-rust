/// File: mp.rs
/// This file contains functionality for multithreaded operations.

use crate::analysis::Analysis;
use crate::spectrum;
use crate::analysis;
use std::thread;
use std::sync::mpsc;
use num::Complex;


/// A multithreaded STFT analyzer using the tools in the analysis crate
pub fn stft_analysis(audio: &mut Vec<f64>, fft_size: usize, sample_rate: u32) -> Vec<Analysis> {
    let stft_imaginary_spectrum: Vec<Vec<Complex<f64>>> = spectrum::rstft(audio, fft_size, fft_size / 2, spectrum::WindowType::Hamming);
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
        thread::spawn(move || {
            let mut analyses: Vec<Analysis> = Vec::with_capacity(local_magnitude_spectrum.len());
            
            // Perform the analyses
            for j in 0..local_magnitude_spectrum.len() {
                analyses.push(analysis::analyzer(&local_magnitude_spectrum[j], local_fft_size, local_sample_rate))
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

    // Combine the analysis vectors into one big vector
    let mut analyses: Vec<Analysis> = Vec::new();
    for i in 0..results.len() {
        for j in 0..results[i].1.len() {
            analyses.push(results[i].1[j]);
        }
    }

    analyses
}
