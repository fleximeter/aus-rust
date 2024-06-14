// File: audiofile.rs
// This file contains functionality for reading from and writing to audio files.

use symphonia::core::codecs::{CODEC_TYPE_NULL, DecoderOptions};
use symphonia::core::errors::Error;
use symphonia::core::formats::FormatOptions;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::probe::Hint;
use symphonia::core::audio::AudioBufferRef;

const INTMAX8: i64 = i64::pow(2, 7) - 1;
const INTMAX16: i64 = i64::pow(2, 15) - 1;
const INTMAX24: i64 = i64::pow(2, 23) - 1;
const INTMAX32: i64 = i64::pow(2, 31) - 1;

/// Represents an audio format (fixed or float)
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

/// Reads an audio file. It can take WAV or AIFF files, as well as other formats.
/// Courtesy of the documentation for symphonia.
pub fn read(path: &String) -> AudioFile {
    let src = std::fs::File::open(&path).expect("Failed to open audio");
    
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
    let probed = symphonia::default::get_probe().format(&hint, mss, &fmt_opts, &meta_opts).expect("Unsupported format");
    let mut format = probed.format;

    // We'll retrieve the first track in the file.
    let track = format.tracks().iter().find(|t| t.codec_params.codec != CODEC_TYPE_NULL).expect("No supported audio tracks found in the file");
    let decoder_options: DecoderOptions = Default::default();
    let mut decoder = symphonia::default::get_codecs().make(&track.codec_params, &decoder_options).expect("Unsupported codec");
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
            Err(Error::ResetRequired) => {
                unimplemented!();
            }
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
                        unimplemented!();
                    }
                }
            }
            Err(Error::IoError(_)) => {
                continue;
            }
            Err(Error::DecodeError(_)) => {
                continue;
            }
            Err(err) => {
                panic!("{}", err);
            }
        }
    }
    audio
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
