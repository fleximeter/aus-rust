// File: audiofile.rs
// This file contains functionality for reading from and writing to audio files.
// It can handle reading multiple audio formats, but only writes to WAV.

use symphonia::core::codecs::{CODEC_TYPE_NULL, DecoderOptions};
use symphonia::core::formats::FormatOptions;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::probe::Hint;
use symphonia::core::audio::AudioBufferRef;
use hound;

const INTMAX8: f64 = (i64::pow(2, 7) - 1) as f64;
const INTMAX16: f64 = (i64::pow(2, 15) - 1) as f64;
const INTMAX24: f64 = (i64::pow(2, 23) - 1) as f64;
const INTMAX32: f64 = (i64::pow(2, 31) - 1) as f64;

/// Represents an audio format (fixed or float)
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum AudioFormat {
    F32,
    F64,
    S8,
    S16,
    S24,
    S32
}

/// Represents an error for audio files
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum AudioError {
    FileInaccessible(String),
    FileCorrupt,
    SampleValueOutOfRange(String),
    NumChannels(String),
    NumFrames(String),
    WrongFormat(String)
}

/// Represents an audio file. Samples are always stored in f64 format,
/// regardless of their original format.
pub struct AudioFile {
    pub audio_format: AudioFormat,
    pub bits_per_sample: u32,
    pub duration: f64,
    pub num_channels: usize,
    pub num_frames: usize,
    pub sample_rate: u32,
    pub samples: Vec<Vec<f64>>,
}

impl AudioFile {
    /// Copies the header of an AudioFile and initializes the struct with an empty sample vector.
    /// Warning - the empty sample vector has no channels and no frames.
    pub fn copy_header(&self) -> AudioFile {
        let samples: Vec<Vec<f64>> = Vec::new();
        AudioFile {
            audio_format: self.audio_format,
            bits_per_sample: self.bits_per_sample,
            duration: self.duration,
            num_channels: self.num_channels,
            num_frames: self.num_frames,
            sample_rate: self.sample_rate,
            samples: samples
        }
    }

    /// Creates a new AudioFile from provided audio format, sample rate, and 2D sample vector.
    /// 
    /// # Example
    /// 
    /// ```
    /// use aus::{AudioFile, AudioFormat};
    /// use aus::synthesis::sine;
    /// let waveform = sine(440.0, 0.0, 44100 * 2, 44100);
    /// let mut channels: Vec<Vec<f64>> = Vec::with_capacity(2);
    /// channels.push(waveform.clone());
    /// channels.push(waveform.clone());
    /// let file = AudioFile::new(AudioFormat::S24, 44100, channels);
    /// aus::write("file_name.wav", &file);
    /// ```
    pub fn new(audio_format: AudioFormat, sample_rate: u32, samples: Vec<Vec<f64>>) -> AudioFile {
        let bits_per_sample: u32 = match audio_format {
            AudioFormat::F32 => 32,
            AudioFormat::F64 => 64,
            AudioFormat::S16 => 16,
            AudioFormat::S24 => 24,
            AudioFormat::S32 => 32,
            AudioFormat::S8 => 8
        };

        let num_frames = if samples.len() > 0 {
            samples[0].len()
        } else {
            0
        };
        let duration = num_frames as f64 / sample_rate as f64;
        AudioFile {
            audio_format,
            bits_per_sample,
            duration,
            num_channels: samples.len(),
            num_frames,
            sample_rate,
            samples
        }
    }

    /// Creates a new mono AudioFile from provided audio format, sample rate, and 1D sample vector.
    /// 
    /// # Example
    /// 
    /// ```
    /// use aus::{AudioFile, AudioFormat};
    /// use aus::synthesis::sine;
    /// let waveform = sine(440.0, 0.0, 44100 * 2, 44100);
    /// let file = AudioFile::new_mono(AudioFormat::S24, 44100, waveform);
    /// aus::write("file_name.wav", &file);
    /// ```
    pub fn new_mono(audio_format: AudioFormat, sample_rate: u32, samples: Vec<f64>) -> AudioFile {
        let mut multi_channel_samples: Vec<Vec<f64>> = Vec::with_capacity(1);
        multi_channel_samples.push(samples);
        let bits_per_sample: u32 = match audio_format {
            AudioFormat::F32 => 32,
            AudioFormat::F64 => 64,
            AudioFormat::S16 => 16,
            AudioFormat::S24 => 24,
            AudioFormat::S32 => 32,
            AudioFormat::S8 => 8
        };
        let num_frames = multi_channel_samples[0].len();
        let duration = num_frames as f64 / sample_rate as f64;
        AudioFile {
            audio_format,
            bits_per_sample,
            duration,
            num_channels: 1,
            num_frames,
            sample_rate,
            samples: multi_channel_samples
        }
    }
}

/// Mixes an audio file down to mono
/// 
/// This will mix all channels down to the first one, and delete
/// the remaining channels. It is performed in-place, so you will
/// lose data!
/// 
/// # Example
/// 
/// ```
/// use aus::{AudioFile, AudioFormat};
/// use aus::synthesis::sine;
/// let waveform = sine(440.0, 0.0, 44100 * 2, 44100);
/// let mut channels: Vec<Vec<f64>> = Vec::with_capacity(2);
/// channels.push(waveform.clone());
/// channels.push(waveform.clone());
/// let mut file = AudioFile::new(AudioFormat::S24, 44100, channels);
/// aus::mixdown(&mut file);
/// ```
pub fn mixdown(audiofile: &mut AudioFile) {
    if audiofile.samples.len() > 1 {
        for frame_idx in 0..audiofile.samples[0].len() {
            for channel_idx in 0..audiofile.samples.len() {
                audiofile.samples[0][frame_idx] += audiofile.samples[channel_idx][frame_idx];
            }
            audiofile.samples[0][frame_idx] /= audiofile.samples.len() as f64;
        }
        audiofile.samples.truncate(1);
    }
}

/// Reads an audio file. Courtesy of symphonia.
/// 
/// # Example
/// 
/// ```
/// use aus::read;
/// let file = read("myfile.wav").unwrap();
/// ```
pub fn read(path: &str) -> Result<AudioFile, AudioError> {
    let src = match std::fs::File::open(&path) {
        Ok(x) => x,
        Err(err) => return Err(AudioError::FileInaccessible(err.to_string()))
    };
    
    let mut audio = AudioFile {
        audio_format: AudioFormat::F32,
        bits_per_sample: 0,
        duration: 0.0,
        num_channels: 0,
        num_frames: 0,
        sample_rate: 0,
        samples: Vec::<Vec<f64>>::new()
    };

    // We need to make a media source stream before opening the file. Symphonia will automatically detect
    // the file format and codec used.
    let mss = MediaSourceStream::new(Box::new(src), Default::default());
    let hint = Hint::new();
    let meta_opts: MetadataOptions = Default::default();
    let fmt_opts: FormatOptions = Default::default();
    let probed = match symphonia::default::get_probe().format(&hint, mss, &fmt_opts, &meta_opts) {
        Ok(x) => x,
        Err(_) => return Err(AudioError::FileCorrupt)
    };
    let mut format = probed.format;

    // We'll retrieve the first track in the file.
    let track = match format.tracks().iter().find(|t| t.codec_params.codec != CODEC_TYPE_NULL) {
        Some(x) => x,
        None => return Err(AudioError::FileCorrupt)
    };
    let decoder_options: DecoderOptions = Default::default();
    let mut decoder = match symphonia::default::get_codecs().make(&track.codec_params, &decoder_options) {
        Ok(x) => x,
        Err(err) => return Err(AudioError::WrongFormat(err.to_string()))
    };
    let track_id = track.id;
    
    // Get metadata information (number of channels, bit depth, and sample rate)
    // Resize the samples vector to handle the appropriate number of channels
    if let Some(track) = format.default_track() {
        let codec_params = &track.codec_params;
        if let Some(channels) = codec_params.channels {
            audio.num_channels = channels.count();
            audio.samples.resize_with(audio.num_channels as usize, Default::default);
        }
        if let Some(sample_rate) = codec_params.sample_rate {
            audio.sample_rate = sample_rate;
        }
        if let Some(bit_depth) = codec_params.bits_per_sample {
            audio.bits_per_sample = bit_depth;
        }
    }

    // Next we'll start a decode loop for the track
    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            // quit at the end of the file
            Err(_) => {
                break;
            }
        };

        while !format.metadata().is_latest() {
            format.metadata().pop();
        }

        if packet.track_id() != track_id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(decoded) => {
                // handle samples of different formats
                match decoded {
                    AudioBufferRef::F32(buf) => {
                        audio.audio_format = AudioFormat::F32;
                        let planes = buf.planes();
                        let mut channel_idx = 0;
                        for plane in planes.planes() {
                            for &sample in plane.iter() {
                                audio.samples[channel_idx].push(sample as f64);
                            }
                            channel_idx += 1;
                        }
                    }
                    AudioBufferRef::F64(buf) => {
                        audio.audio_format = AudioFormat::F64;
                        let planes = buf.planes();
                        let mut channel_idx = 0;
                        for plane in planes.planes() {
                            for &sample in plane.iter() {
                                audio.samples[channel_idx].push(sample);
                            }
                            channel_idx += 1;
                        }
                    }
                    AudioBufferRef::S8(buf) => {
                        audio.audio_format = AudioFormat::S8;
                        let planes = buf.planes();
                        let mut channel_idx = 0;
                        for plane in planes.planes() {
                            for &sample in plane.iter() {
                                audio.samples[channel_idx].push(sample as f64 / INTMAX8 as f64);
                            }
                            channel_idx += 1;
                        }
                    }
                    AudioBufferRef::S16(buf) => {
                        audio.audio_format = AudioFormat::S16;
                        let planes = buf.planes();
                        let mut channel_idx = 0;
                        for plane in planes.planes() {
                            for &sample in plane.iter() {
                                audio.samples[channel_idx].push(sample as f64 / INTMAX16 as f64);
                            }
                            channel_idx += 1;
                        }
                    }
                    AudioBufferRef::S24(buf) => {
                        audio.audio_format = AudioFormat::S24;
                        let planes = buf.planes();
                        let mut channel_idx = 0;
                        for plane in planes.planes() {
                            for &sample in plane.iter() {
                                audio.samples[channel_idx].push(sample.inner() as f64 / INTMAX24 as f64);
                            }
                            channel_idx += 1;
                        }
                    }
                    AudioBufferRef::S32(buf) => {
                        audio.audio_format = AudioFormat::S32;
                        let planes = buf.planes();
                        let mut channel_idx: usize = 0;
                        for plane in planes.planes() {
                            for &sample in plane.iter() {
                                audio.samples[channel_idx].push(sample as f64 / INTMAX32 as f64);
                            }
                            channel_idx += 1;
                        }
                    }
                    // We don't support other formats, such as unsigned.
                    _ => {
                        return Err(AudioError::WrongFormat(String::from("This audio reader does not support unsigned sample types.")));
                    }
                }
            }
            Err(err) => return Err(AudioError::FileCorrupt)
        }
    }
    audio.num_frames = audio.samples[0].len();
    Ok(audio)
}

/// Writes a WAV audio file to disk. Courtesy of hound.
/// 
/// # Example
/// 
/// ```
/// use aus::{read, write};
/// let file = read("myfile.wav").unwrap();
/// write("myfile2.wav", &file);
/// ```
pub fn write(path: &str, audio: &AudioFile) -> Result<(), AudioError> {
    // Verify that the number of channels and frames in the audio sample vector are correct
    if audio.samples.len() != audio.num_channels {
        return Err(AudioError::NumChannels(String::from(format!("The AudioFile claims to have {} channels, but it actually has {} channels.", 
            audio.num_channels, audio.samples.len()))));
    }
    for i in 0..audio.samples.len() {
        if audio.samples[i].len() != audio.num_frames {
            return Err(AudioError::NumChannels(String::from(format!("The AudioFile claims to have {} frames, but it actually has {} frames in channel {}.", 
                audio.num_frames, audio.samples[i].len(), i))));
        }
    }
    
    // The file spec for the hound crate
    let spec = if audio.audio_format == AudioFormat::F32 || audio.audio_format == AudioFormat::F64 {
        hound::WavSpec {
            channels: audio.num_channels as u16,
            sample_rate: audio.sample_rate,
            bits_per_sample: audio.bits_per_sample as u16,
            sample_format: hound::SampleFormat::Float
        }
    } else {
        hound::WavSpec {
            channels: audio.num_channels as u16,
            sample_rate: audio.sample_rate,
            bits_per_sample: audio.bits_per_sample as u16,
            sample_format: hound::SampleFormat::Int
        }
    };
    
    let mut writer = match hound::WavWriter::create(path, spec) {
        Ok(x) => x,
        Err(err) => return Err(AudioError::FileInaccessible(err.to_string()))
    };

    // Write the samples
    if audio.audio_format == AudioFormat::F32 || audio.audio_format == AudioFormat::F64 {
        for j in 0..audio.num_frames {
            for i in 0..audio.num_channels {
                if audio.samples[i][j].abs() > 1.0 {
                    return Err(AudioError::SampleValueOutOfRange(String::from(format!(
                        "Sample index {} in channel {} had value {}, which is out of range. Sample values must be between -1.0 and 1.0.", 
                        j, i, audio.samples[i][j]))));
                } else {
                    match writer.write_sample(audio.samples[i][j] as f32) {
                        Ok(()) => (),
                        Err(err) => return Err(AudioError::FileInaccessible(err.to_string()))
                    }
                }
            }
        }
    } else {
        // The maximum sample value
        let maxval: f64 = match audio.audio_format {
            AudioFormat::S8 => INTMAX8,
            AudioFormat::S16 => INTMAX16,
            AudioFormat::S24 => INTMAX24,
            _ => INTMAX32,
        };
        for j in 0..audio.num_frames {
            for i in 0..audio.num_channels {
                if audio.samples[i][j].abs() > 1.0 {
                    return Err(AudioError::SampleValueOutOfRange(String::from(format!(
                        "Sample index {} in channel {} had value {}, which is out of range. Sample values must be between -1.0 and 1.0.", 
                        j, i, audio.samples[i][j]))));
                } else {
                    match writer.write_sample((audio.samples[i][j] * maxval).round() as i32) {
                        Ok(()) => (),
                        Err(err) => return Err(AudioError::FileInaccessible(err.to_string()))
                    }
                }
            }
        }
    }

    match writer.finalize() {
        Ok(()) => Ok(()),
        Err(err) => return Err(AudioError::FileInaccessible(err.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const SAMPLE_RATE: u32 = 44100;

    /// Tests the methods of the AudioFile struct
    #[test]
    fn test_audiofile_methods() {
        const NUM_SAMPLES: usize = 100;
        let channel1_vec: Vec<f64> = vec![0.0; NUM_SAMPLES];
        let mut all_channels_vec: Vec<Vec<f64>> = Vec::with_capacity(1);
        all_channels_vec.push(channel1_vec.clone());
        
        // Test creating an AudioFile from scratch and copying the header
        let af1 = AudioFile{
            audio_format: AudioFormat::S24,
            bits_per_sample: 24,
            duration: NUM_SAMPLES as f64 / SAMPLE_RATE as f64,
            num_channels: 1,
            num_frames: NUM_SAMPLES,
            sample_rate: SAMPLE_RATE,
            samples: all_channels_vec.clone()
        };

        let copied_af = af1.copy_header();
        assert_eq!(af1.audio_format, copied_af.audio_format);
        assert_eq!(af1.bits_per_sample, copied_af.bits_per_sample);
        assert_eq!(af1.duration, copied_af.duration);
        assert_eq!(af1.num_channels, copied_af.num_channels);
        assert_eq!(af1.num_frames, copied_af.num_frames);
        assert_eq!(af1.sample_rate, copied_af.sample_rate);
        assert_eq!(af1.samples.len(), 1);
        
        // Test the new methods
        let af2 = AudioFile::new(AudioFormat::S16, SAMPLE_RATE, all_channels_vec);
        let af3 = AudioFile::new_mono(AudioFormat::S16, SAMPLE_RATE, channel1_vec);

        assert_eq!(af2.audio_format, AudioFormat::S16);
        assert_eq!(af2.bits_per_sample, 16);
        assert_eq!(af2.duration, NUM_SAMPLES as f64 / SAMPLE_RATE as f64);
        assert_eq!(af2.num_channels, 1);
        assert_eq!(af2.num_frames, NUM_SAMPLES);
        assert_eq!(af2.sample_rate, SAMPLE_RATE);
        assert_eq!(af2.samples.len(), 1);
        assert_eq!(af2.samples[0].len(), NUM_SAMPLES);

        assert_eq!(af3.audio_format, AudioFormat::S16);
        assert_eq!(af3.bits_per_sample, 16);
        assert_eq!(af3.duration, NUM_SAMPLES as f64 / SAMPLE_RATE as f64);
        assert_eq!(af3.num_channels, 1);
        assert_eq!(af3.num_frames, NUM_SAMPLES);
        assert_eq!(af3.sample_rate, SAMPLE_RATE);
        assert_eq!(af3.samples.len(), 1);
        assert_eq!(af3.samples[0].len(), NUM_SAMPLES);
    }

    /// Tests reading audio
    #[test]
    fn test_read() {
        let path = String::from("blah");
        let _ = match read(&path) {
            Ok(_) => panic!("The audio file shouldn't be able to be located."),
            Err(_) => ()
        };
    }

    /// Tests writing audio
    #[test]
    fn test_write() {
        const NUM_SAMPLES: usize = 100;
        let channel1_vec: Vec<f64> = vec![0.0; NUM_SAMPLES];
        
        let mut af1 = AudioFile::new_mono(AudioFormat::S16, SAMPLE_RATE, channel1_vec);
        let path = String::from("audio/temp.wav");
        
        af1.samples[0][0] = 2.1;
        let _ = match write(&path, &af1) {
            Ok(_) => panic!("The writer should have flagged the sample value that was out of bounds."),
            Err(_) => ()
        };
        af1.num_channels = 2;
        let _ = match write(&path, &af1) {
            Ok(_) => panic!("The writer should have flagged the wrong number of channels."),
            Err(_) => ()
        };
        af1.num_channels = 1;
        af1.num_frames = 1;
        let _ = match write(&path, &af1) {
            Ok(_) => panic!("The writer should have flagged the wrong number of frames."),
            Err(_) => ()
        };
    }
}
