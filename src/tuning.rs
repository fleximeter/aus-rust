// File: tuning.rs
// This file contains tuning functionality.


/// Calculates the MIDI note of a provided frequency.
///  
/// # Examples
/// 
/// ```
/// use audiorust::tuning::freq_to_midi;
/// let midi_note = freq_to_midi(440.0);
/// ```
#[inline(always)]
pub fn freq_to_midi(frequency: f64) -> f64 {
    f64::log2(frequency / 440.0) * 12.0 + 69.0
}

/// Calculates the frequency of a provided MIDI note.
/// 
/// # Examples
/// 
/// ```
/// use audiorust::tuning::midi_to_freq;
/// let freq = midi_to_freq(69.0);
/// ```
#[inline(always)]
pub fn midi_to_freq(midi: f64) -> f64 {
    440.0 * f64::powf(2.0, (midi - 69.0) / 12.0)
}

/// Calculates the ratio between two MIDI notes. You provide an interval in semitones.
/// 
/// # Examples
/// 
/// ```
/// use audiorust::tuning::midi_ratio;
/// let ratio = midi_ratio(4.0);
/// ```
#[inline(always)]
pub fn midi_ratio(interval: f64) -> f64 {
    f64::powf(2.0, interval / 12.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tuning() {
        let freq = 100.0;
        let midi = 65.0;
        let interval = 4.0;
        assert_eq!(freq_to_midi(freq), f64::log2(freq / 440.0) * 12.0 + 69.0);
        assert_eq!(midi_to_freq(midi), 440.0 * f64::powf(2.0, (midi - 69.0) / 12.0));
        assert_eq!(midi_ratio(interval), f64::powf(2.0, interval / 12.0));
    }
}
