// File: audiofile.rs
// This file contains functionality for reading from and writing to audio files.

use std::fs;

pub struct AudioFile {
    audio_format: u16,
    bits_per_sample: u16,
    block_align: u16,
    byte_rate: u32,
    bytes_per_sample: u16,
    duration: f64,
    file_name: String,
    path: String,
    num_channels: u16,
    frames: u32,
    sample_rate: u32,
    samples: Vec<Vec<f64>>,
}

impl AudioFile {
    pub fn get_audio_format(&self) -> u16 {
        self.audio_format
    }
    pub fn get_bits_per_sample(&self) -> u16 {
        self.bits_per_sample
    }
    pub fn get_byte_rate(&self) -> u32 {
        self.byte_rate
    }
    pub fn get_bytes_per_sample(&self) -> u16 {
        self.bytes_per_sample
    }
    pub fn get_duration(&self) -> f64 {
        self.duration
    }
    pub fn get_file_name(&self) -> String {
        let file_name = self.file_name.clone();
        file_name
    }
    pub fn get_frames(&self) -> u32 {
        self.frames
    }
    pub fn get_num_channels(&self) -> u16 {
        self.num_channels
    }
    pub fn get_path(&self) -> String {
        let path = self.path.clone();
        path
    }
    pub fn get_sample_rate(&self) -> u32 {
        self.sample_rate
    }
    pub fn set_audio_format(&mut self, format: u16) {
        self.audio_format = format;
    }
    pub fn set_bits_per_sample(&mut self, bits: u16) {
        self.bits_per_sample = bits;
        self.bytes_per_sample = bits / 8;
        self.byte_rate = self.bytes_per_sample as u32 * self.num_channels as u32 * self.sample_rate;
        self.block_align = self.num_channels * self.bytes_per_sample;
    }
}

pub fn read_wav(path: &String) -> AudioFile {
    // constants for chunk identification
    const _RIFF: [u8; 4] = [82, 73, 70, 70];
    const _WAVE: [u8; 4] = [87, 65, 86, 69];
    const _FMT: [u8; 4] = [102, 109, 116, 32];
    const _JUNK: [u8; 4] = [74, 85, 78, 75];
    const _DATA: [u8; 4] = [100, 97, 116, 97];
    const _RIFF_CHUNK_SIZE: usize = 12;
    const _FMT_CHUNK_SIZE: usize = 16;
    const _CHUNK_HEADER_SIZE: usize = 8;
    
    let mut audio = AudioFile {
        audio_format: 1,
        bits_per_sample: 16,
        block_align: 1,
        byte_rate: 1,
        bytes_per_sample: 2,
        duration: 0.0,
        file_name: String::new(),
        path: String::from(path),
        num_channels: 1,
        frames: 1,
        sample_rate: 0,
        samples: Vec::<Vec<f64>>::new(),
    };

    match fs::read(&path) {
        Ok(contents) => {
            let mut bytes_remaining = contents.len() as i64;
            let mut i = 0;
            // verify riff chunk before proceeding
            if contents.len() >= _RIFF_CHUNK_SIZE {
                if contents[i..i+4] == _RIFF && contents[i+8..i+12] == _WAVE {
                    i += _RIFF_CHUNK_SIZE;
                    bytes_remaining -= _RIFF_CHUNK_SIZE as i64;

                    // We need to be able to read both the header name and the chunk size, so
                    // if there are fewer than 8 bytes remaining, we just have to stop.
                    while bytes_remaining >= 8 {
                        // get the remaining chunk size
                        let header: &[u8] = &contents[i..i+4];
                        let chunk_size: [u8; 4] = contents[i+4..i+8].try_into().expect("Chunk size extraction failed");
                        let chunk_size = u32::from_le_bytes(chunk_size) as usize;
                        i += _CHUNK_HEADER_SIZE;
                        bytes_remaining -= _CHUNK_HEADER_SIZE as i64;

                        // FMT chunk
                        if header == _FMT && chunk_size == _FMT_CHUNK_SIZE {
                            audio.audio_format = u16::from_le_bytes(contents[i..i+2].try_into().expect("Audio format extraction failed"));
                            audio.num_channels = u16::from_le_bytes(contents[i+2..i+4].try_into().expect("Number of channels extraction failed"));
                            audio.sample_rate = u32::from_le_bytes(contents[i+4..i+8].try_into().expect("Sample rate extraction failed"));
                            audio.byte_rate = u32::from_le_bytes(contents[i+8..i+12].try_into().expect("Byte rate extraction failed"));
                            audio.block_align = u16::from_le_bytes(contents[i+12..i+14].try_into().expect("Block align extraction failed"));
                            audio.bits_per_sample = u16::from_le_bytes(contents[i+14..i+16].try_into().expect("Bits per sample extraction failed"));
                            audio.bytes_per_sample = audio.bits_per_sample / 8;
                        }

                        // data chunk
                        else if header == _DATA {
                            audio.frames = chunk_size as u32 / (audio.num_channels * audio.bytes_per_sample) as u32;
                            audio.duration = (audio.frames / audio.sample_rate) as f64;

                            // resize the samples vector in preparation for importing audio
                            audio.samples.resize_with(audio.num_channels as usize, Default::default);
                            for channel in audio.samples.iter_mut() {
                                channel.resize_with(audio.frames as usize, Default::default);
                            }

                            // all forms of fixed up to int 64
                            if audio.audio_format == 1 {
                                let intmax = f64::powf(2.0, (audio.bits_per_sample - 1) as f64);
                                let mut sample: [u8; 8] = [0, 0, 0, 0, 0, 0, 0, 0];
                                for j in 0..audio.frames as usize {
                                    let start_idx_frame = i + j * audio.block_align as usize;
                                    for k in 0..audio.num_channels as usize {
                                        let start_idx_channel = start_idx_frame + k * audio.bytes_per_sample as usize;
                                        for l in start_idx_channel..start_idx_channel+(audio.bytes_per_sample as usize) {
                                            sample[l-start_idx_channel] = contents[l];
                                        }
                                        audio.samples[k][j] = i64::from_le_bytes(sample) as f64 / intmax;
                                    }
                                }
                            }

                            // float 32
                            else if audio.audio_format == 3 && audio.bits_per_sample == 32 {
                                println!("Audio format 3.1");
                                let mut sample: [u8; 4] = [0, 0, 0, 0];
                                for j in 0..audio.frames as usize {
                                    let start_idx_frame = i + j * audio.block_align as usize;
                                    for k in 0..audio.num_channels as usize {
                                        let start_idx_channel = start_idx_frame + k * audio.bytes_per_sample as usize;
                                        for l in start_idx_channel..start_idx_channel+(audio.bytes_per_sample as usize) {
                                            sample[l-start_idx_channel] = contents[l];
                                        }
                                        audio.samples[k][j] = f32::from_le_bytes(sample) as f64;
                                    }
                                }
                            }

                            // float 64
                            else if audio.audio_format == 3 && audio.bits_per_sample == 64 {
                                println!("Audio format 3.2");
                                let mut sample: [u8; 8] = [0, 0, 0, 0, 0, 0, 0, 0];
                                for j in 0..audio.frames as usize {
                                    let start_idx_frame = i + j * audio.block_align as usize;
                                    for k in 0..audio.num_channels as usize {
                                        let start_idx_channel = start_idx_frame + k * audio.bytes_per_sample as usize;
                                        for l in start_idx_channel..start_idx_channel+(audio.bytes_per_sample as usize) {
                                            sample[l-start_idx_channel] = contents[l];
                                        }
                                        audio.samples[k][j] = f64::from_le_bytes(sample);
                                    }
                                }
                            }
                        }

                        i += chunk_size;
                        bytes_remaining -= chunk_size as i64;                            
                    }
                }
            }
        }
        Err(err) => {
            eprintln!("Error reading file: {}", err);
        }
    }

    audio
}
