# aus
This is a crate for audio processing and analysis in Rust, combining new functionality with aggregated functionality from other existing crates. For example, this crate provides wrappers for `rustfft`, allowing a FFT to be performed with a single function call. It also has a STFT/ISTFT function pair. It also has built-in window generation in the style of `numpy`. And there are implementations of spectral feature extraction, such as calculating spectral centroid, entropy, slope, etc.

## Primary Goals
- Abstraction of existing crates (`rustfft`, `symphonia`, `hound`, `fft-convolver`), allowing their functionality to be used with a simple function call or so.
- FFT functionality designed not just for analysis, but also for FFT modification and resynthesis, including STFT.
- Analysis tools that compute spectral and audio features for analysis and synthesis projects.
- Multithreaded tools for more efficient processing. At present, there is a multithreaded analyzer that allows spectral analysis data to be computed much more quickly for an entire audio file.

## Features
- Audio read/write using `symphonia` and `hound`. Reads multiple formats, but only writes to WAV.
- FFT processing courtesy of `rustfft`. Includes real FFT, inverse real FFT, real STFT, inverse real STFT, spectrum decomposition and recomposition.
- Spectral transformations (scrambling FFT bins and STFT frames, as well as spectral freeze). Includes a convenience wrapper for `fft-convolver`.
- Granular synthesis tools
- Tuning computation
- Spectral analysis tools:
    - Spectral centroid
    - Spectral entropy
    - Spectral flatness
    - Spectral kurtosis
    - Spectral roll-off-points
    - Spectral skewness
    - Spectral slope (including slope of sub-bands)
    - Spectral variance
