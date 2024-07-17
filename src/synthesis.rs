//! # Synthesis
//! The `synthesis` module contains synthesis functionality.

use std::f64::consts::PI;

/// Creates a sawtooth waveform of length `len`, frequency `frequency`, and initial phase `initial_phase`.
/// The `max_harmonic_index` controls how many partials are present. Naturally, for sufficiently high frequency
/// and sufficiently high max harmonic, this will lead to aliasing.
/// 
/// # Example
/// 
/// ```
/// use aus::synthesis::saw;
/// use std::f64::consts::PI;
/// let freq = 440.0;
/// let iphase = PI / 2.0;
/// let max_harmonic_index = 20;
/// let sample_rate: u32 = 44100;
/// let length = sample_rate as usize * 5;
/// let waveform = saw(freq, iphase, max_harmonic_index, length, sample_rate);
/// ```
pub fn saw(frequency: f64, initial_phase: f64, max_harmonic_index: usize, len: usize, sample_rate: u32) -> Vec<f64> {
    let mut samples: Vec<f64> = vec![0.0; len];
    let sine_coef = 2.0 * PI * frequency / sample_rate as f64;
    let adjusted_phase = initial_phase / sine_coef;
    for n in 0..len {
        let mut y_val: f64 = 0.0;
        for p in 1..max_harmonic_index {
            let x_position = sine_coef * p as f64 * (n as f64 + adjusted_phase);
            y_val += f64::sin(x_position) / (2 * p) as f64;
        }
        samples[n] = y_val;
    }
    samples
}

/// Creates a sine waveform of length `len`, frequency `frequency`, and initial phase `initial_phase`.
/// 
/// # Example
/// 
/// ```
/// use aus::synthesis::sine;
/// use std::f64::consts::PI;
/// let freq = 440.0;
/// let iphase = PI / 2.0;
/// let sample_rate: u32 = 44100;
/// let length = sample_rate as usize * 5;
/// let waveform = sine(freq, iphase, length, sample_rate);
/// ```
pub fn sine(frequency: f64, initial_phase: f64, len: usize, sample_rate: u32) -> Vec<f64> {
    let mut samples: Vec<f64> = vec![0.0; len];
    let sine_coef = 2.0 * PI * frequency / sample_rate as f64;
    let adjusted_phase = initial_phase / sine_coef;
    for n in 0..len {
        let x_position = sine_coef * (n as f64 + adjusted_phase);
        samples[n] = f64::sin(x_position)
    }
    samples
}

/// Creates a square waveform of length `len`, frequency `frequency`, and initial phase `initial_phase`.
/// The `max_harmonic_index` controls how many partials are present. Naturally, for sufficiently high frequency
/// and sufficiently high max harmonic, this will lead to aliasing.
/// 
/// # Example
/// 
/// ```
/// use aus::synthesis::square;
/// use std::f64::consts::PI;
/// let freq = 440.0;
/// let iphase = PI / 2.0;
/// let max_harmonic_index = 20;
/// let sample_rate: u32 = 44100;
/// let length = sample_rate as usize * 5;
/// let waveform = square(freq, iphase, max_harmonic_index, length, sample_rate);
/// ```
pub fn square(frequency: f64, initial_phase: f64, max_harmonic_index: usize, len: usize, sample_rate: u32) -> Vec<f64> {
    let mut samples: Vec<f64> = vec![0.0; len];
    let sine_coef = 2.0 * PI * frequency / sample_rate as f64;
    let adjusted_phase = initial_phase / sine_coef;
    let adjusted_index = (max_harmonic_index - 1) / 2;
    for n in 0..len {
        let mut y_val: f64 = 0.0;
        for p in 0..adjusted_index {
            let x_position = (2 * p + 1) as f64 * sine_coef * (n as f64 + adjusted_phase);
            y_val += f64::sin(x_position) / (2 * p + 1) as f64;
        }
        samples[n] = y_val;
    }
    samples
}


/// Creates a triangle waveform of length `len`, frequency `frequency`, and initial phase `initial_phase`.
/// The `max_harmonic_index` controls how many partials are present. Naturally, for sufficiently high frequency
/// and sufficiently high max harmonic, this will lead to aliasing.
/// 
/// # Example
/// 
/// ```
/// use aus::synthesis::triangle;
/// use std::f64::consts::PI;
/// let freq = 440.0;
/// let iphase = PI / 2.0;
/// let max_harmonic_index = 20;
/// let sample_rate: u32 = 44100;
/// let length = sample_rate as usize * 5;
/// let waveform = triangle(freq, iphase, max_harmonic_index, length, sample_rate);
/// ```
pub fn triangle(frequency: f64, initial_phase: f64, max_harmonic_index: usize, len: usize, sample_rate: u32) -> Vec<f64> {
    let mut samples: Vec<f64> = vec![0.0; len];
    let sine_coef = 2.0 * PI * frequency / sample_rate as f64;
    let global_coef = 8.0 / (PI * PI);
    let adjusted_phase = initial_phase / sine_coef;
    let adjusted_index = (max_harmonic_index - 1) / 2;
    for n in 0..len {
        let mut y_val: f64 = 0.0;
        let mut flipper = 1;
        for p in 0..adjusted_index {
            let local_coef = (2 * p + 1) as f64;
            let x_position = local_coef * sine_coef * (n as f64 + adjusted_phase);
            y_val += f64::sin(x_position) * flipper as f64 / (local_coef * local_coef);
            flipper *= -1;
        }
        samples[n] = y_val * global_coef;
    }
    samples
}

#[cfg(test)]
mod test {
    use super::*;
    use std::f64::consts::PI;
    const DIR: &str = "D:/Recording/tests";
    
    #[test]
    fn test_saw() {
        let waveform = saw(440.0, PI / 2.0, 20, 44100 * 5, 44100);
        let audio = crate::AudioFile::new_mono(crate::AudioFormat::S24, 44100, waveform);
        crate::write(&format!("{}/saw.wav", DIR), &audio).unwrap();
    }
    
    #[test]
    fn test_sine() {
        let waveform = sine(440.0, PI / 2.0, 44100 * 5, 44100);
        let audio = crate::AudioFile::new_mono(crate::AudioFormat::S24, 44100, waveform);
        crate::write(&format!("{}/sine.wav", DIR), &audio).unwrap();
    }
    
    #[test]
    fn test_square() {
        let waveform = square(440.0, PI / 2.0, 20, 44100 * 5, 44100);
        let audio = crate::AudioFile::new_mono(crate::AudioFormat::S24, 44100, waveform);
        crate::write(&format!("{}/square.wav", DIR), &audio).unwrap();
    }
    
    #[test]
    fn test_triangle() {
        let waveform = triangle(440.0, PI / 2.0, 20, 44100 * 5, 44100);
        let audio = crate::AudioFile::new_mono(crate::AudioFormat::S24, 44100, waveform);
        crate::write(&format!("{}/triangle.wav", DIR), &audio).unwrap();
    }
}