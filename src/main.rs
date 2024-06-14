// File: main.rs
// This file contains functionality for testing the package.

mod audiofile;
mod analysis;
mod spectrum;

fn main() {
    let path = String::from("D:\\Recording\\Samples\\Iowa\\Cello.arco.mono.2444.1\\samples_ff\\sample_Cello.arco.ff.sulA.A3Ab4.wav_7.wav");
    let audio = audiofile::read(&path);
}
