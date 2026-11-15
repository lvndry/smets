use std::error::Error;
use std::path::Path;

use hound::{SampleFormat, WavReader, WavSpec, WavWriter};

use crate::pipeline::SAMPLE_RATE;

pub fn read_stereo(path: &Path) -> Result<[Vec<f32>; 2], Box<dyn Error>> {
    let mut reader = WavReader::open(path)?;
    let spec = reader.spec();
    if spec.sample_rate != SAMPLE_RATE {
        return Err(format!("expected {} Hz audio, got {} Hz", SAMPLE_RATE, spec.sample_rate).into());
    }
    let channels = spec.channels as usize;
    let samples: Vec<f32> = match spec.sample_format {
        SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
        SampleFormat::Int => {
            let scale = 1.0 / (1u64 << (spec.bits_per_sample - 1)) as f32;
            reader.samples::<i32>().map(|sample| sample.map(|value| value as f32 * scale)).collect::<Result<_, _>>()?
        }
    };
    let frames = samples.len() / channels;
    let mut left = Vec::with_capacity(frames);
    let mut right = Vec::with_capacity(frames);
    for frame in 0..frames {
        left.push(samples[frame * channels]);
        right.push(samples[frame * channels + (channels - 1).min(1)]);
    }
    Ok([left, right])
}

pub fn write_stereo(path: &Path, channels: &[Vec<f32>; 2]) -> Result<(), Box<dyn Error>> {
    let spec = WavSpec { channels: 2, sample_rate: SAMPLE_RATE, bits_per_sample: 32, sample_format: SampleFormat::Float };
    let mut writer = WavWriter::create(path, spec)?;
    for index in 0..channels[0].len() {
        writer.write_sample(channels[0][index])?;
        writer.write_sample(channels[1][index])?;
    }
    writer.finalize()?;
    Ok(())
}
