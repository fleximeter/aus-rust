//! # Mel spectrum
//! The `analysis::mel` module contains functionality for Mel spectrum and MFCC analysis.
//! Mel spectrum and MFCC computation is designed to mimic the `librosa` implementation of `librosa.feature.melspectrogram` and `librosa.feature.mfcc`.
//! 
//! McFee, Brian, Colin Raffel, Dawen Liang, Daniel PW Ellis, Matt McVicar, Eric Battenberg, and Oriol Nieto. “librosa: Audio and music signal analysis in python.” In Proceedings of the 14th python in science conference, pp. 18-25. 2015.

use crate::util;
use rustdct::DctPlanner;

use super::make_log_spectrum;

/// Represents a computation function for generating a triangular filter
/// as part of a Mel filterbank
struct Triangle {
    pub x1: f64,
    pub x2: f64,
    pub x3: f64,
    pub y1: f64,
    pub y2: f64,
    slope_ascending: f64,
    slope_descending: f64
}

impl Triangle {
    /// Creates a new Triangle struct with endpoints x1 and x3 and midpoint x2, with low point y1 and high point y2
    pub fn new(x1: f64, x2: f64, x3: f64, y1: f64, y2: f64) -> Triangle {
        Triangle {
            x1: x1,
            x2: x2,
            x3: x3,
            y1: y1,
            y2: y2,
            slope_ascending: (y2 - y1) / (x2 - x1),
            slope_descending: (y1 - y2) / (x3 - x2)
        }
    }

    /// Computes the value at position `pos`. If `pos` is before `x1` or after `x3`,
    /// the output value will be `y1` (which is 0 for Mel triangular filters)
    pub fn compute(&self, pos: f64) -> f64 {
        if pos <= self.x1 || pos >= self.x3 {
            return self.y1;
        } else if pos <= self.x2 {
            return self.slope_ascending * (pos - self.x1) + self.y1;
        } else {
            return self.slope_descending * (pos - self.x2) + self.y2;
        }
    }
}

/// Represents a triangular filter for use in a Mel filterbank
struct TriangleFilter {
    start_idx: usize,
    end_idx: usize,
    triangle_filter: Vec<f64>
}

impl TriangleFilter {
    /// Creates a new triangular filter for use in a Mel filterbank.
    /// The filter is applied to a magnitude or power spectrum.
    /// However, the area of the spectrum for filtering lies between `start_idx` and `end_idx`,
    /// so it is not necessary to generate a filter of the same length as the spectral frame.
    /// The `triangle` struct handles the generation of the filter.
    pub fn new(start_idx: usize, end_idx: usize, triangle: Triangle, fft_freqs: &[f64], normalize: bool) -> TriangleFilter {
        let length = end_idx - start_idx + 1;
        let mut filter: Vec<f64> = Vec::with_capacity(length);
        if normalize {
            let coef = 2.0 / (fft_freqs[end_idx] - fft_freqs[start_idx]);
            for i in 0..length {
                filter.push(triangle.compute(fft_freqs[i + start_idx]) * coef);
            }
        } else {
            for i in 0..length {
                filter.push(triangle.compute(fft_freqs[i + start_idx]));
            }
        }
        //dbg!(start_idx, end_idx, triangle.x1, triangle.x2, triangle.x3, triangle.y1, triangle.y2);
        //dbg!(&filter);
        TriangleFilter {
            start_idx: start_idx,
            end_idx: end_idx,
            triangle_filter: filter
        }
    }

    /// Filters the input spectral vector by the triangle filter
    pub fn filter(&self, vec: &[f64]) -> f64 {
        let mut result: f64 = 0.0;
        let mut i: usize = 0;
        let mut j: usize = self.start_idx;
        while j <= self.end_idx {
            result += self.triangle_filter[i] * vec[j];
            i += 1;
            j += 1;
        }
        result
    }
}

/// Represents a Mel filterbank of triangular filters
pub struct MelFilterbank {
    freq_low: f64,
    freq_high: f64,
    filters: Vec<TriangleFilter>,
    num_filters: usize
}

impl MelFilterbank {
    /// Constructs a triangular filterbank within the frequency range `freq_low` to `freq_high`,
    /// with `num_filters` filters. If `quantize` is true, then the filter points are quantized
    /// to the nearest values in the provided array of `fft_freqs`.
    /// If `normalize` is true, then librosa-style filter scaling is applied.
    pub fn new(freq_low: f64, freq_high: f64, num_filters: usize, fft_freqs: &[f64], quantize: bool, normalize: bool) -> MelFilterbank {
        // Determine the filter points, including a start point for the first filter
        // and an end point for the last filter
        let arr_len = num_filters + 2;
        let mel_low = freq_to_mel(freq_low, true);
        let mel_high = freq_to_mel(freq_high, true);
        let mel_center_freqs: Vec<f64> = util::linspace(mel_low, mel_high, arr_len, true);
        let freq_center_freqs: Vec<f64> = mel_center_freqs.iter().map(|x| mel_to_freq(*x, true)).collect();
        
        // Compute each filter in the filterbank
        let mut filterbank: Vec<TriangleFilter> = Vec::with_capacity(num_filters);
        for i in 1..num_filters+1 {
            // The triangle points
            let low_freq: f64;
            let mid_freq: f64;
            let high_freq: f64;

            // We may want to quantize the filterbank X points to actual values in `fft_freqs`
            if quantize {
                low_freq = fft_freqs[util::ordered_search(&fft_freqs, freq_center_freqs[i-1]).unwrap()];
                mid_freq = fft_freqs[util::ordered_search(&fft_freqs, freq_center_freqs[i]).unwrap()];
                high_freq = fft_freqs[util::ordered_search(&fft_freqs, freq_center_freqs[i+1]).unwrap()];
            } else {
                low_freq = freq_center_freqs[i-1];
                mid_freq = freq_center_freqs[i];
                high_freq = freq_center_freqs[i+1];
            }

            // Create the triangle computation function and the triangular filter
            let tri = Triangle::new(low_freq, mid_freq, high_freq, 0.0, 1.0);
            let start_idx = util::ordered_search_le(&fft_freqs, freq_center_freqs[i-1]).unwrap();
            let mut end_idx = util::ordered_search_le(&fft_freqs, freq_center_freqs[i+1]).unwrap() + 2;
            if end_idx >= fft_freqs.len() {
                end_idx = fft_freqs.len() - 1;
            }
            let tri_filter = TriangleFilter::new(start_idx, end_idx, tri, fft_freqs, normalize);
            filterbank.push(tri_filter);
        }

        // A triangular filterbank for Mel filtering
        MelFilterbank {
            freq_low: freq_low,
            freq_high: freq_high,
            filters: filterbank,
            num_filters: num_filters
        }
    }

    /// Computes the Mel spectrum for a given magnitude or power spectrum.
    /// 
    /// # Example
    /// ```
    /// use aus::{spectrum, analysis};
    /// let fft_size = 2048;
    /// let audio = aus::read("myfile.wav").unwrap();
    /// let audio_chunk = &audio.samples[0][..fft_size];
    /// let rfft_freqs = spectrum::rfftfreq(fft_size, audio.sample_rate);
    /// let mel_filterbank = analysis::mel::MelFilterbank::new(20.0, 8000.0, 40, &rfft_freqs, true, false);
    /// let imaginary_spectrum = spectrum::rfft(&audio_chunk, fft_size);
    /// let (magnitude_spectrum, phase_spectrum) = spectrum::complex_to_polar_rfft(&imaginary_spectrum);
    /// let power_spectrum = analysis::make_power_spectrum(&magnitude_spectrum);
    /// let mel_spectrum = mel_filterbank.filter(&power_spectrum);
    /// ```
    pub fn filter(&self, vec: &[f64]) -> Vec<f64> {
        let mut filtered: Vec<f64> = Vec::with_capacity(self.num_filters);
        for i in 0..self.num_filters {
            filtered.push(self.filters[i].filter(&vec))
        }
        filtered
    }

    /// Gets the lowest frequency of the filterbank
    pub fn get_freq_low(&self) -> f64 {
        self.freq_low
    }

    /// Gets the highest frequency of the filterbank
    pub fn get_freq_high(&self) -> f64 {
        self.freq_high
    }

    /// Gets the number of filters in the filterbank
    pub fn len(&self) -> usize {
        self.num_filters
    }
}

/// Computes the Mel equivalent of a frequency in Hz.
/// If `slaney` is `true` (recommended behavior), the piecewise Slaney formula will be used:
/// $$
/// f^{(mel)} = \begin{cases}
/// \frac{3f}{200} & \text{if } f \leq 1000 \\\\
/// 15 + 27\log\_{6.4}{\left(\frac{f}{1000}\right)} & \text{if } f \geq 1000
/// \end{cases}
/// $$
/// Otherwise, the O'Shaughnessy formula is used:
/// $$
/// f^{(mel)}=2595 \log_{10}{\left(1+\frac{f}{700}\right)}
/// $$
/// The default behavior in `librosa` is to use the Slaney formula.
#[inline]
pub fn freq_to_mel(freq: f64, slaney: bool) -> f64 {
    if slaney {
        if freq < 1000.0 {
            3.0 * freq / 200.0
        } else {
            15.0 + 27.0 * (f64::log10(freq / 1000.0) / f64::log10(6.4))
        }
    } else {
        2595.0 * f64::log10(1.0 + freq / 700.0)
    }
}

/// Computes the frequency equivalent in Hz of a Mel.
/// If `slaney` is `true` (recommended behavior), the piecewise Slaney formula will be used:
/// $$
/// f = \begin{cases}
/// \frac{200f^{(mel})}{3} & \text{if } f^{(mel)} < 15 \\\\
/// 1000 \cdot 10^{\frac{\log{(6.4)}(f^{(mel)}-15)}{27}} & \text{if } f^{(mel)} \geq 15
/// \end{cases}
/// $$
/// Otherwise, the O'Shaughnessy formula is used:
/// $$
/// f = 700 \left(10^{\frac{f^{(mel)}}{2595}} - 1\right)
/// $$
/// The default behavior in `librosa` is to use the Slaney formula.
#[inline]
pub fn mel_to_freq(mel: f64, slaney: bool) -> f64 {
    if slaney {
        if mel < 15.0 {
            200.0 * mel / 3.0
        } else {
            1000.0 * f64::powf(10.0, f64::log10(6.4) * (mel - 15.0) / 27.0)
        }
    } else {
        700.0 * (f64::powf(10.0, mel / 2595.0) - 1.0)
    }
}

/// Generates a Mel scale from an array of frequencies
pub fn melscale(freqs: &[f64]) -> Vec<f64> {
    let mut scale: Vec<f64> = Vec::with_capacity(freqs.len());
    for freq in freqs {
        scale.push(freq_to_mel(*freq, true));
    }
    scale
}

/// Computes the Mel spectrogram from a given magnitude or power spectrogram.
/// You need to specify the lower and upper Mel bounds. 
/// You also need to specify the number of filters (this determines the size of the Mel spectrum).
/// 
/// This function also returns the associated Mel scale.
/// 
/// # Example
/// ```
/// use aus::{spectrum, analysis};
/// let fft_size = 2048;
/// let audio = aus::read("myfile.wav").unwrap();
/// let rfft_freqs = spectrum::rfftfreq(fft_size, audio.sample_rate);
/// let mel_filterbank = analysis::mel::MelFilterbank::new(20.0, 8000.0, 40, &rfft_freqs, true, false);
/// let imaginary_spectrogram = spectrum::rstft(&audio.samples[0], fft_size, fft_size / 2, aus::WindowType::Hanning);
/// let (magnitude_spectrogram, _) = spectrum::complex_to_polar_rstft(&imaginary_spectrogram);
/// let power_spectrogram = analysis::make_power_spectrogram(&magnitude_spectrogram);
/// let mel_spectrogram = analysis::mel::make_mel_spectrogram(&power_spectrogram, &mel_filterbank);
/// ```
pub fn make_mel_spectrogram(spectrogram: &[Vec<f64>], filterbank: &MelFilterbank) -> Vec<Vec<f64>> {
    let mut mel_spectrogram: Vec<Vec<f64>> = Vec::with_capacity(spectrogram.len());
    for i in 0..spectrogram.len() {
        mel_spectrogram.push(filterbank.filter(&spectrogram[i]));
    }
    mel_spectrogram
}

/// Derives the Mel frequency cepstral coefficients (MFCCs) given a Mel spectrum.
/// Eyben's advice is to use a 20-8000Hz filterbank, a 26-band spectrum, and discard all MFCCs except 12-16. (Eyben, 60-61)
/// 
/// If you provide a `lifter` value greater than 0.0, liftering will be applied to the MFCCs
/// (this approach is borrowed from `librosa`: <https://librosa.org/doc/main/generated/librosa.feature.mfcc.html>).
/// 
/// The MFCCs are derived by converting the Mel spectrum to a log Mel spectrum, then applying the Discrete Cosine Transform Type II.
/// Liftering is optional.
/// 
/// # Example
/// This example covers the entire process for calculating the MFCCs from a FFT frame.
/// ```
/// use aus::{spectrum, analysis};
/// let fft_size = 2048;
/// let audio = aus::read("myfile.wav").unwrap();
/// let rfft_freqs = spectrum::rfftfreq(fft_size, audio.sample_rate);
/// let mel_filterbank = analysis::mel::MelFilterbank::new(20.0, 8000.0, 40, &rfft_freqs, true, false);
/// let audio_chunk = &audio.samples[0][..fft_size];
/// let imaginary_spectrum = spectrum::rfft(&audio_chunk, fft_size);
/// let (magnitude_spectrum, phase_spectrum) = spectrum::complex_to_polar_rfft(&imaginary_spectrum);
/// let power_spectrum = analysis::make_power_spectrum(&magnitude_spectrum);
/// let mel_spectrum = mel_filterbank.filter(&power_spectrum);
/// let log_spectrum: Vec<f64> = analysis::make_log_spectrum(&mel_spectrum, 10e-8);
/// let mfccs = analysis::mel::mfcc(&log_spectrum, 2.0); // then use indices 11-15
/// ```
pub fn mfcc_spectrum(mel_spectrum: &[f64], lifter: f64) -> Vec<f64> {
    let mut planner = DctPlanner::new();
    let dct3 = planner.plan_dct2(mel_spectrum.len());
    let mut mfccs: Vec<f64> = make_log_spectrum(mel_spectrum, 10e-8);
    dct3.process_dct3(&mut mfccs);
    
    // Perform "liftering"
    if lifter > 0.0 {
        for k in 0..mfccs.len() {
            mfccs[k] *= 1.0 + (lifter / 2.0) * f64::sin(std::f64::consts::PI * (k + 1) as f64 / lifter); 
        }
    }
    mfccs
}

/// Derives the Mel frequency cepstral coefficients (MFCCs) given a Mel spectrum.
/// Eyben's advice is to use a 20-8000Hz filterbank, a 26-band spectrum, and discard all MFCCs except 12-16. (Eyben, 60-61)
/// 
/// If you provide a `lifter` value greater than 0.0, liftering will be applied to the MFCCs
/// (this approach is borrowed from `librosa`: <https://librosa.org/doc/main/generated/librosa.feature.mfcc.html>).
/// 
/// The MFCCs are derived by converting the Mel spectrum to a log Mel spectrum, then applying the Discrete Cosine Transform Type II.
/// Liftering is optional.
/// 
/// # Example
/// This example covers the entire process for calculating the MFCCs from a FFT frame.
/// ```
/// use aus::{spectrum, analysis};
/// let fft_size = 2048;
/// let audio = aus::read("myfile.wav").unwrap();
/// let rfft_freqs = spectrum::rfftfreq(fft_size, audio.sample_rate);
/// let mel_filterbank = analysis::mel::MelFilterbank::new(20.0, 8000.0, 40, &rfft_freqs, true, false);
/// let imaginary_spectrogram = spectrum::rstft(&audio.samples[0], fft_size, fft_size / 2, aus::WindowType::Hanning);
/// let (magnitude_spectrogram, _) = spectrum::complex_to_polar_rstft(&imaginary_spectrogram);
/// let power_spectrogram = analysis::make_power_spectrogram(&magnitude_spectrogram);
/// let mel_spectrogram = analysis::mel::make_mel_spectrogram(&power_spectrogram, &mel_filterbank);
/// let mfccs = analysis::mel::mfcc_spectrogram(&mel_spectrogram, 2.0); // then use indices 11-15
/// ```
pub fn mfcc_spectrogram(mel_spectrogram: &[Vec<f64>], lifter: f64) -> Vec<Vec<f64>> {
    let mut mfccs: Vec<Vec<f64>> = Vec::new();
    if mel_spectrogram.len() > 0 {
        let mut planner = DctPlanner::new();
        let dct3 = planner.plan_dct2(mel_spectrogram[0].len());
        
        for spec in mel_spectrogram {
            let mut mfccs_vec: Vec<f64> = make_log_spectrum(spec, 10e-8);
            dct3.process_dct3(&mut mfccs_vec);
            
            // Perform "liftering"
            if lifter > 0.0 {
                for k in 0..mfccs_vec.len() {
                    mfccs_vec[k] *= 1.0 + (lifter / 2.0) * f64::sin(std::f64::consts::PI * (k + 1) as f64 / lifter); 
                }
            }

            mfccs.push(mfccs_vec);
        }
    }
   
    mfccs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{spectrum, analysis};
    
    // Tests frequency to Mel conversion
    #[test]
    fn test_freq_to_mel() {
        const EPSILON: f64 = 1e-6;
        // Test Slaney formula
        assert!(f64::abs(freq_to_mel(-1.004, true) - -0.015059999999999999) < EPSILON);
        assert!(f64::abs(freq_to_mel(0.0, true) - 0.0) < EPSILON);
        assert!(f64::abs(freq_to_mel(1.0, true) - 0.015) < EPSILON);
        assert!(f64::abs(freq_to_mel(50.0, true) - 0.75) < EPSILON);
        assert!(f64::abs(freq_to_mel(142.429, true) - 2.136435) < EPSILON);
        assert!(f64::abs(freq_to_mel(451.987, true) - 6.779805) < EPSILON);
        assert!(f64::abs(freq_to_mel(1000.0, true) - 14.999999999999998) < EPSILON);
        assert!(f64::abs(freq_to_mel(1002.428, true) - 15.035272646925904) < EPSILON);
        assert!(f64::abs(freq_to_mel(5304.53, true) - 39.26935324537849) < EPSILON);
        assert!(f64::abs(freq_to_mel(12042.233, true) - 51.194262623415284) < EPSILON);
        // Test O'Shaughnessy formula
        assert!(f64::abs(freq_to_mel(-1.004, false) - -1.617591975265908) < EPSILON);
        assert!(f64::abs(freq_to_mel(0.0, false) - 0.0) < EPSILON);
        assert!(f64::abs(freq_to_mel(1.0, false) - 1.6088427864826338) < EPSILON);
        assert!(f64::abs(freq_to_mel(50.0, false) - 77.75456466446511) < EPSILON);
        assert!(f64::abs(freq_to_mel(142.429, false) - 208.72952224868627) < EPSILON);
        assert!(f64::abs(freq_to_mel(451.987, false) - 561.4270515062376) < EPSILON);
        assert!(f64::abs(freq_to_mel(586.1, false) - 685.5385318706192) < EPSILON);
        assert!(f64::abs(freq_to_mel(1002.428, false) - 1001.5940016448719) < EPSILON);
        assert!(f64::abs(freq_to_mel(5304.53, false) - 2422.1236404690194) < EPSILON);
        assert!(f64::abs(freq_to_mel(12042.233, false) - 3270.0827681073483) < EPSILON);
    }

    /// Tests Mel to frequency conversion
    #[test]
    fn test_mel_to_freq() {
        const EPSILON: f64 = 1e-6;
        // Test Slaney formula
        assert!(f64::abs(mel_to_freq(-1.01, true) - -67.33333333333334) < EPSILON);
        assert!(f64::abs(mel_to_freq(0.0, true) - 0.0) < EPSILON);
        assert!(f64::abs(mel_to_freq(0.32, true) - 21.333333333333336) < EPSILON);
        assert!(f64::abs(mel_to_freq(1.45, true) - 96.66666666666667) < EPSILON);
        assert!(f64::abs(mel_to_freq(5.923, true) - 394.8666666666667) < EPSILON);
        assert!(f64::abs(mel_to_freq(11.483, true) - 765.5333333333334) < EPSILON);
        assert!(f64::abs(mel_to_freq(15.0, true) - 1000.0000000000002) < EPSILON);
        assert!(f64::abs(mel_to_freq(15.03, true) - 1002.0646818488809) < EPSILON);
        assert!(f64::abs(mel_to_freq(18.4922, true) - 1271.3698701976102) < EPSILON);
        assert!(f64::abs(mel_to_freq(23.5809, true) - 1803.9020548956844) < EPSILON);
        // Test O'Shaughnessy formula
        assert!(f64::abs(mel_to_freq(-43.0, false) - -26.205110846993996) < EPSILON);
        assert!(f64::abs(mel_to_freq(0.0, false) - 0.0) < EPSILON);
        assert!(f64::abs(mel_to_freq(1.23, false) - 0.7643961548333467) < EPSILON);
        assert!(f64::abs(mel_to_freq(43.45, false) - 27.51470832169547) < EPSILON);
        assert!(f64::abs(mel_to_freq(120.4335, false) - 78.94692390028159) < EPSILON);
        assert!(f64::abs(mel_to_freq(435.239, false) - 329.9596192156131) < EPSILON);
        assert!(f64::abs(mel_to_freq(801.43, false) - 725.3918244975519) < EPSILON);
        assert!(f64::abs(mel_to_freq(1009.87, false) - 1014.975669072667) < EPSILON);
        assert!(f64::abs(mel_to_freq(2003.49, false) - 3441.482623133689) < EPSILON);
        assert!(f64::abs(mel_to_freq(3210.49, false) - 11385.958101160506) < EPSILON);
    }

    /// Verifies that the `Triangle` struct computes triangle values correctly
    #[test]
    fn test_triangle_computation() {
        const EPSILON: f64 = 1e-6;
        let tri = Triangle::new(5.0, 20.0, 82.0, 1.1, 9.44);
        assert!({f64::abs(tri.compute(2.0) - 1.1) < EPSILON});
        assert!({f64::abs(tri.compute(4.0) - 1.1) < EPSILON});
        assert!({f64::abs(tri.compute(5.0) - 1.1) < EPSILON});
        assert!({f64::abs(tri.compute(11.0) - 4.436) < EPSILON});
        assert!({f64::abs(tri.compute(20.0) - 9.44) < EPSILON});
        assert!({f64::abs(tri.compute(21.0) - 9.305483870967741) < EPSILON});
        assert!({f64::abs(tri.compute(43.0) - 6.346129032258064) < EPSILON});
        assert!({f64::abs(tri.compute(82.0) - 1.1) < EPSILON});
        assert!({f64::abs(tri.compute(98.0) - 1.1) < EPSILON});
    }

    /// Verifies that the `Triangle` struct computes triangle values correctly
    #[test]
    fn test_mel_filterbank() {
        const EPSILON: f64 = 1e-6;
        const FFT_SIZE: usize = 256;
        const SAMPLE_RATE: u32 = 22050;
        const NUM_MELS: usize = 20;
        const LOW_FREQ: f64 = 0.0;
        const HIGH_FREQ: f64 = 11025.0;
        let rfreqs = crate::spectrum::rfftfreq(FFT_SIZE, SAMPLE_RATE);
        let fb = MelFilterbank::new(LOW_FREQ, HIGH_FREQ, NUM_MELS, &rfreqs, false, true);
        
        // A fake FFT power spectrum to test on
        let fake_fft_power_spectrum: Vec<f64> = vec![
            1e-2, 2e-5, 15e-4, 21e-5, 1e-4, 2e-5, 11e-5, 31e-4, 132e-9, 22e-3, 152e-3, 201e-5, 123e-5, 221e-6, 11e-5, 32e-4,
            12e-5, 2e-5, 15e-4, 21e-5, 1e-4, 2e-5, 11e-5, 31e-4, 132e-9, 22e-3, 152e-3, 201e-5, 123e-5, 221e-6, 11e-5, 32e-4,
            3e-4, 2e-5, 15e-4, 21e-5, 1e-4, 2e-5, 11e-5, 31e-4, 132e-9, 22e-3, 152e-3, 201e-5, 123e-5, 221e-6, 11e-5, 32e-4,
            2e-3, 2e-5, 15e-4, 21e-5, 1e-4, 2e-5, 11e-5, 31e-4, 132e-9, 22e-3, 152e-3, 201e-5, 123e-5, 221e-6, 11e-5, 32e-4,
            43e-6, 2e-5, 15e-4, 21e-5, 1e-4, 2e-5, 11e-5, 31e-4, 132e-9, 22e-3, 152e-3, 201e-5, 123e-5, 221e-6, 11e-5, 32e-4,
            2e-3, 2e-5, 15e-4, 21e-5, 1e-4, 2e-5, 11e-5, 31e-4, 132e-9, 22e-3, 152e-3, 201e-5, 123e-5, 221e-6, 11e-5, 32e-4,
            6e-5, 2e-5, 15e-4, 21e-5, 1e-4, 2e-5, 11e-5, 31e-4, 132e-9, 22e-3, 152e-3, 201e-5, 123e-5, 221e-6, 11e-5, 32e-4,
            4e-3, 2e-5, 15e-4, 21e-5, 1e-4, 2e-5, 11e-5, 31e-4, 132e-9, 22e-3, 152e-3, 201e-5, 123e-5, 221e-6, 11e-5, 32e-4,
            2e-5
        ];
        let mel_spec = fb.filter(&fake_fft_power_spectrum);

        // This literal output vector was generated from a Python mockup, and the
        // Mel spectrum produced by this test should match it. 
        let out_vec: Vec<f64> = vec![6.76895304e-06, 1.36028451e-06, 2.77367273e-06, 1.89267587e-05,
            4.07858842e-04, 3.19873457e-04, 5.04253261e-06, 8.52845397e-06,
            4.99160938e-06, 5.40207921e-06, 3.29421796e-04, 7.69145971e-05,
            6.61126227e-06, 2.21605018e-04, 4.42958155e-05, 1.83195179e-04,
            9.42967421e-05, 1.23944028e-04, 1.34227360e-04, 1.39157067e-04];

        for i in 0..out_vec.len() {
            assert!({f64::abs(out_vec[i] - mel_spec[i]) < EPSILON})
        }
    }

    /// Verifies that it is possible to compute a Mel spectrogram without crashing
    #[test]
    fn test_mel_spec() {
        let fft_size = 2048;
        let audio = crate::read("myfile.wav").unwrap();
        let rfft_freqs = spectrum::rfftfreq(fft_size, audio.sample_rate);
        let mel_filterbank = analysis::mel::MelFilterbank::new(20.0, 8000.0, 40, &rfft_freqs, true, false);
        let imaginary_spectrogram = spectrum::rstft(&audio.samples[0], fft_size, fft_size / 2, crate::WindowType::Hanning);
        let (magnitude_spectrogram, _) = spectrum::complex_to_polar_rstft(&imaginary_spectrogram);
        let power_spectrogram = analysis::make_power_spectrogram(&magnitude_spectrogram);
        let _ = analysis::mel::make_mel_spectrogram(&power_spectrogram, &mel_filterbank);
    }

    /// Verifies that it is possible to compute MFCCs for a Mel spectrogram without crashing
    #[test]
    fn test_mfcc_spectrum() {
        let fft_size = 2048;
        let audio = crate::read("myfile.wav").unwrap();
        let rfft_freqs = spectrum::rfftfreq(fft_size, audio.sample_rate);
        let mel_filterbank = analysis::mel::MelFilterbank::new(20.0, 8000.0, 40, &rfft_freqs, true, false);
        let imaginary_spectrogram = spectrum::rstft(&audio.samples[0], fft_size, fft_size / 2, crate::WindowType::Hanning);
        let (magnitude_spectrogram, _) = spectrum::complex_to_polar_rstft(&imaginary_spectrogram);
        let power_spectrogram = analysis::make_power_spectrogram(&magnitude_spectrogram);
        let mel_spectrogram = analysis::mel::make_mel_spectrogram(&power_spectrogram, &mel_filterbank);
        let _ = analysis::mel::mfcc_spectrogram(&mel_spectrogram, 2.0); // then use indices 11-15
    }
}
