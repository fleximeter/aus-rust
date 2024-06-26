// File: audiofile.rs
// This file contains functionality for reading from and writing to audio files.

use symphonia::core::codecs::{CODEC_TYPE_NULL, DecoderOptions};
use symphonia::core::errors::Error;
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
///
#[derive(Copy, Clone)]
pub enum AudioFormat {
    F32,
    F64,
    S8,
    S16,
    S24,
    S32
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
}

/// Converts a float sample to fixed
#[inline(always)]
fn convert_to_fixed(sample: f64, format: &AudioFormat) -> i32 {
    let maxval: f64 = match format {
        AudioFormat::S8 => INTMAX8,
        AudioFormat::S16 => INTMAX16,
        AudioFormat::S24 => INTMAX24,
        _ => INTMAX32,
    };
    (sample * maxval) as i32
}

/// Mixes an audio file down to mono
/// 
/// This will mix all channels down to the first one, and delete
/// the remaining channels. It is performed in-place, so you will
/// lose data!
/// 
/// # Examples
/// ```
/// mod audiofile;
/// let audioFile = audiofile::read_wav("SomeAudio.wav");
/// audiofile::mixdown(audioFile);
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

/// Reads an audio file. It can take WAV or AIFF files, as well as other formats.
/// Courtesy of the documentation for symphonia.
pub fn read(path: &String) -> Result<AudioFile, std::io::Error> {
    let src = match std::fs::File::open(&path) {
        Ok(x) => x,
        Err(err) => return Err(std::io::Error::from(err))
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
        Err(err) => return Err(std::io::Error::new(std::io::ErrorKind::Other, err.to_string()))
    };
    let mut format = probed.format;

    // We'll retrieve the first track in the file.
    let track = match format.tracks().iter().find(|t| t.codec_params.codec != CODEC_TYPE_NULL) {
        Some(x) => x,
        None => return Err(std::io::Error::new(std::io::ErrorKind::Other, "No tracks in the audio file"))
    };
    let decoder_options: DecoderOptions = Default::default();
    let mut decoder = match symphonia::default::get_codecs().make(&track.codec_params, &decoder_options) {
        Ok(x) => x,
        Err(err) => return Err(std::io::Error::new(std::io::ErrorKind::Other, err.to_string()))
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
            Ok(_decoded) => {
                // handle samples of different formats
                match _decoded {
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
                        return Err(std::io::Error::new(std::io::ErrorKind::Other, "This audio reader does not support unsigned sample types."));
                    }
                }
            }
            Err(err) => return Err(std::io::Error::new(std::io::ErrorKind::Other, err.to_string()))
        }
    }
    audio.num_frames = audio.samples[0].len();
    Ok(audio)
}

/// Writes an audio file to disk
pub fn write(path: String, audio: &AudioFile) -> Result<(), std::io::Error> {
    // Verify that the number of channels and frames in the audio sample vector are correct
    if audio.samples.len() != audio.num_channels {
        return Err(std::io::Error::new(std::io::ErrorKind::Other, "The number of channels specified in the AudioFile does not match the number of channels present."));
    }
    for i in 0..audio.samples.len() {
        if audio.samples[i].len() != audio.num_frames {
            return Err(std::io::Error::new(std::io::ErrorKind::Other, "The number of frames specified in the AudioFile does not match the number of frames present."));
        }
    }

    // Set the specifications and format for the audio file
    let spec = hound::WavSpec {
        channels: audio.num_channels as u16,
        sample_rate: audio.sample_rate,
        bits_per_sample: audio.bits_per_sample as u16,
        sample_format: hound::SampleFormat::Int,
    };
    let format = match audio.bits_per_sample {
        8 => AudioFormat::S8,
        16 => AudioFormat::S16,
        24 => AudioFormat::S24,
        _ => AudioFormat::S32
    };
    
    let mut writer = match hound::WavWriter::create(path, spec) {
        Ok(x) => x,
        Err(err) => return Err(std::io::Error::new(std::io::ErrorKind::Other, err.to_string()))
    };

    // Write the samples
    for j in 0..audio.num_frames {
        for i in 0..audio.num_channels {
            match writer.write_sample(convert_to_fixed(audio.samples[i][j], &format)) {
                Ok(()) => (),
                Err(err) => return Err(std::io::Error::new(std::io::ErrorKind::Other, err.to_string()))
            }
        }
    }
    match writer.finalize() {
        Ok(()) => Ok(()),
        Err(err) => return Err(std::io::Error::new(std::io::ErrorKind::Other, err.to_string()))
    }
}
