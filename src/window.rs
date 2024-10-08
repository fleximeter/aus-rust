// File: window.rs
// 
// This file contains window definitions.

use std::f64::consts::PI;

/// Represents a window type.
#[derive(Copy, Clone)]
pub enum WindowType{
    Bartlett,
    Blackman,
    Hanning,
    Hamming,
    Rectangular
}

/// Creates a Bartlett window of size `window_length`.
/// 
/// # Example
/// ```
/// use aus::generate_window_bartlett;
/// let window_length = 22050;
/// let window = generate_window_bartlett(window_length);
/// ```
#[inline(always)]
pub fn generate_window_bartlett(window_length: usize) -> Vec<f64>{
    let mut window: Vec<f64> = vec![0.0; window_length];
    for i in 0..window_length {
        window[i] = (2.0 / (window_length as f64 - 1.0)) * ((window_length as f64 - 1.0) / 2.0 - f64::abs(i as f64 - (window_length as f64 - 1.0) / 2.0));
    }
    window
}

/// Creates a Blackman window of size `window_length`.
/// 
/// # Example
/// ```
/// use aus::generate_window_blackman;
/// let window_length = 22050;
/// let window = generate_window_blackman(window_length);
/// ```
#[inline(always)]
pub fn generate_window_blackman(window_length: usize) -> Vec<f64>{
    let mut window: Vec<f64> = vec![0.0; window_length];
    for i in 0..window_length {
        window[i] = 0.42 - 0.5 * f64::cos((2.0 * PI * i as f64) / (window_length as f64)) 
            + 0.08 * f64::cos((4.0 * PI * i as f64) / (window_length as f64));
    }
    window
}

/// Creates a Hanning window of size `window_length`.
/// 
/// # Example
/// ```
/// use aus::generate_window_hanning;
/// let window_length = 22050;
/// let window = generate_window_hanning(window_length);
/// ```
#[inline(always)]
pub fn generate_window_hanning(window_length: usize) -> Vec<f64>{
    let mut window: Vec<f64> = vec![0.0; window_length];
    for i in 0..window_length {
        window[i] = 0.5 - 0.5 * f64::cos((2.0 * PI * i as f64) / (window_length as f64 - 1.0));
    }
    window
}

/// Creates a Hamming window of size `window_length`.
/// 
/// # Example
/// ```
/// use aus::generate_window_hamming;
/// let window_length = 22050;
/// let window = generate_window_hamming(window_length);
/// ```
#[inline(always)]
pub fn generate_window_hamming(window_length: usize) -> Vec<f64>{
    let mut window: Vec<f64> = vec![0.0; window_length];
    for i in 0..window_length {
        window[i] = 0.54 - 0.46 * f64::cos((2.0 * PI * i as f64) / (window_length as f64 - 1.0));
    }
    window
}

/// Creates a rectangular window of size `window_length`.
/// 
/// # Example
/// ```
/// use aus::generate_window_rectangular;
/// let window_length = 22050;
/// let window = generate_window_rectangular(window_length);
/// ```
#[inline(always)]
pub fn generate_window_rectangular(window_length: usize) -> Vec<f64>{
    let window: Vec<f64> = vec![1.0; window_length];
    window
}

/// Gets the corresponding window for a provided `WindowType` and window size.
/// 
/// # Example
/// ```
/// use aus::{generate_window, WindowType};
/// let window_type = WindowType::Hanning;
/// let window_length = 22050;
/// let window = generate_window(window_type, window_length);
/// ```
#[inline(always)]
pub fn generate_window(window_type: WindowType, window_length: usize) -> Vec<f64> {
    match &window_type {
        WindowType::Bartlett => generate_window_bartlett(window_length),
        WindowType::Blackman => generate_window_blackman(window_length),
        WindowType::Hanning => generate_window_hanning(window_length),
        WindowType::Hamming => generate_window_hamming(window_length),
        WindowType::Rectangular => generate_window_rectangular(window_length),
    }
}
