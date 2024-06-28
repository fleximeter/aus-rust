/// File: tuning.rs
/// This file contains tuning functionality.


/// Calculates the MIDI note of a provided frequency
#[inline(always)]
pub fn freq_to_midi(frequency: f64) -> f64 {
    f64::log2(frequency / 440.0) * 12.0 + 69.0
}

/// Calculates the frequency of a provided MIDI note
#[inline(always)]
pub fn midi_to_freq(midi: f64) -> f64 {
    440.0 * f64::powf(2.0, (midi - 69.0) / 12.0)
}

/// Calculates the ratio between two MIDI notes
#[inline(always)]
pub fn midi_ratio(interval: f64) -> f64 {
    f64::powf(2.0, interval / 12.0)
}
