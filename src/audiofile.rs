use std::fs;
use std::mem;

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
    fn get_audio_format(&self) -> u16 {
        self.audio_format
    }
    fn get_bits_per_sample(&self) -> u16 {
        self.bits_per_sample
    }
    fn get_byte_rate(&self) -> u32 {
        self.byte_rate
    }
    fn get_bytes_per_sample(&self) -> u16 {
        self.bytes_per_sample
    }
    fn get_duration(&self) -> f64 {
        self.duration
    }
    fn get_file_name(&self) -> String {
        let file_name = self.file_name.clone();
        file_name
    }
    fn get_frames(&self) -> u32 {
        self.frames
    }
    fn get_num_channels(&self) -> u16 {
        self.num_channels
    }
    fn get_path(&self) -> String {
        let path = self.path.clone();
        path
    }
    fn get_sample_rate(&self) -> u32 {
        self.sample_rate
    }
    fn set_audio_format(&mut self, format: u16) {
        self.audio_format = format;
    }
    fn set_bits_per_sample(&mut self, bits: u16) {
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
            let mut bytes_remaining = contents.len();
            let mut i = 0;
            // verify riff chunk before proceeding
            if contents.len() >= _RIFF_CHUNK_SIZE {
                if contents[i..i+4] == _RIFF && contents[i+8..i+12] == _WAVE {
                    i += _RIFF_CHUNK_SIZE;
                    bytes_remaining -= _RIFF_CHUNK_SIZE;
                    while bytes_remaining > 0 {
                        if bytes_remaining < 8 {
                            // invalid chunk
                        }

                        // get the remaining chunk size
                        let header: &[u8] = &contents[i..i+4];
                        let (chunk_size, _) = contents[i+4..i+8].split_at(mem::size_of::<u32>());
                        let chunk_size = usize::from_le_bytes(chunk_size.try_into().unwrap());
                        i += _CHUNK_HEADER_SIZE;

                        // FMT chunk
                        if header == _FMT && chunk_size == _FMT_CHUNK_SIZE {
                            let (audio_format, _) = contents[i+8..i+10].split_at(mem::size_of::<u16>());
                            let (num_channels, _) = contents[i+10..i+12].split_at(mem::size_of::<u16>());
                            let (sample_rate, _) = contents[i+12..i+16].split_at(mem::size_of::<u32>());
                            let (byte_rate, _) = contents[i+16..i+20].split_at(mem::size_of::<u32>());
                            let (block_align, _) = contents[i+20..i+22].split_at(mem::size_of::<u16>());
                            let (bits_per_sample, _) = contents[i+22..i+24].split_at(mem::size_of::<u16>());
                            audio.audio_format = u16::from_le_bytes(audio_format.try_into().unwrap());
                            audio.num_channels = u16::from_le_bytes(num_channels.try_into().unwrap());
                            audio.sample_rate = u32::from_le_bytes(sample_rate.try_into().unwrap());
                            audio.byte_rate = u32::from_le_bytes(byte_rate.try_into().unwrap());
                            audio.block_align = u16::from_le_bytes(block_align.try_into().unwrap());
                            audio.bits_per_sample = u16::from_le_bytes(bits_per_sample.try_into().unwrap());
                            audio.bytes_per_sample = audio.bits_per_sample / 8;
                            i += _CHUNK_HEADER_SIZE + _FMT_CHUNK_SIZE;
                            bytes_remaining -= _CHUNK_HEADER_SIZE + _FMT_CHUNK_SIZE;
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
                                    for k in 0..audio.num_channels as usize {
                                        for l in i..i+(audio.bytes_per_sample as usize) {
                                            sample[l-i] = contents[l];
                                        }
                                        audio.samples[k][j] = i64::from_le_bytes(sample) as f64 / intmax;
                                        i += audio.bytes_per_sample as usize;
                                        bytes_remaining -= audio.bytes_per_sample as usize;                        
                                    }
                                }
                            }

                            // float 32
                            else if audio.audio_format == 3 && audio.bits_per_sample == 32 {
                                let mut sample: [u8; 4] = [0, 0, 0, 0];
                                for j in 0..audio.frames as usize {
                                    for k in 0..audio.num_channels as usize {
                                        for l in i..i+(audio.bytes_per_sample as usize) {
                                            sample[l-i] = contents[l];
                                        }
                                        audio.samples[k][j] = f32::from_le_bytes(sample) as f64;
                                        i += audio.bytes_per_sample as usize;
                                        bytes_remaining -= audio.bytes_per_sample as usize;                  
                                    }
                                }
                            }

                            // float 64
                            else if audio.audio_format == 3 && audio.bits_per_sample == 64 {
                                let mut sample: [u8; 8] = [0, 0, 0, 0, 0, 0, 0, 0];
                                for j in 0..audio.frames as usize {
                                    for k in 0..audio.num_channels as usize {
                                        for l in i..i+(audio.bytes_per_sample as usize) {
                                            sample[l-i] = contents[l];
                                        }
                                        audio.samples[k][j] = f64::from_le_bytes(sample);
                                        i += audio.bytes_per_sample as usize;
                                        bytes_remaining -= audio.bytes_per_sample as usize;               
                                    }
                                }
                            }
                        }

                        // anything else, including JUNK
                        else {
                            i += _CHUNK_HEADER_SIZE + chunk_size;
                            bytes_remaining -= _CHUNK_HEADER_SIZE + chunk_size;                            
                        }
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
