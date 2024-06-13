// File: main.rs
// This file contains functionality for testing the package.

mod audiofile;
mod analysis;

fn main() {
    let path = String::from("C:\\Users\\jeffr\\Recording\\grains.wav");
    let audio = audiofile::read_wav(&path);
    println!("Num channels: {0}", audio.get_num_channels());
    println!("Sample rate: {0}", audio.get_sample_rate());
    println!("Byte rate: {0}", audio.get_byte_rate());
    println!("Bits per sample: {0}", audio.get_bits_per_sample());
    println!("Bytes per sample: {0}", audio.get_bytes_per_sample());
}
