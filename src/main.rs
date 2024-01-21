mod audiofile;

fn main() {
    let path = String::from("D:\\temp.wav");
    audiofile::read_wav(&path);
    println!("Hello, world!");
}
