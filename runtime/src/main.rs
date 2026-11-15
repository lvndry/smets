use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use smets::audio::{read_stereo, write_stereo};
use smets::{DEFAULT_CHUNK_SAMPLES, STEM_NAMES, Separator};

#[derive(Parser)]
#[command(about = "Separate a 44.1 kHz stereo WAV file into drums, bass, other and vocals")]
struct Arguments {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    output_directory: PathBuf,
    #[arg(long, default_value_t = 8)]
    threads: usize,
    #[arg(long, default_value_t = 2)]
    overlap: usize,
    #[arg(long, default_value_t = DEFAULT_CHUNK_SAMPLES)]
    chunk_samples: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments = Arguments::parse();
    std::fs::create_dir_all(&arguments.output_directory)?;
    let [left, right] = read_stereo(&arguments.input)?;
    let mut separator = Separator::load(&arguments.model, arguments.threads, arguments.chunk_samples)?;
    let started = Instant::now();
    let stems = separator.separate(&left, &right, arguments.overlap)?;
    let seconds = left.len() as f64 / 44_100.0;
    let elapsed = started.elapsed().as_secs_f64();
    eprintln!("separated {seconds:.1} s of audio in {elapsed:.1} s ({:.1} s per minute)", elapsed / seconds * 60.0);
    for (name, channels) in STEM_NAMES.iter().zip(stems.iter()) {
        write_stereo(&arguments.output_directory.join(format!("{name}.wav")), channels)?;
    }
    Ok(())
}
