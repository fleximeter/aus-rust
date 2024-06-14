// File: main.rs
// This file contains functionality for testing the package.

mod audiofile;
mod analysis;

fn main() {
    let path = String::from("D:\\Recording\\Samples\\Iowa\\Cello.arco.mono.2444.1\\samples_ff\\sample_Cello.arco.ff.sulA.A3Ab4.wav_7.wav");
    let audio = audiofile::read_wav(&path);
    println!("Num channels: {0}", audio.get_num_channels());
    println!("Sample rate: {0}", audio.get_sample_rate());
    println!("Byte rate: {0}", audio.get_byte_rate());
    println!("Bits per sample: {0}", audio.get_bits_per_sample());
    println!("Bytes per sample: {0}", audio.get_bytes_per_sample());
    println!("Frames: {0}", audio.get_frames());
    println!("Sample: {0}", audio.samples[0][34902])
}
