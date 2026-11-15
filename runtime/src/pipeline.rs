use std::error::Error;
use std::path::Path;

use ort::ep::CPU;
use ort::environment::Environment;
use ort::session::Session;
use ort::session::builder::AutoDevicePolicy;
use ort::value::Tensor;

use crate::spectral::{BINS, HOP, Spectral, Spectrogram};

pub const DEFAULT_CHUNK_SAMPLES: usize = 485_100;
pub const SAMPLE_RATE: u32 = 44_100;
pub const STEM_NAMES: [&str; 4] = ["drums", "bass", "other", "vocals"];

pub type StemAudio = Vec<[Vec<f32>; 2]>;

pub struct Separator {
    chunk_samples: usize,
    _environment: Environment,
    session: Session,
    spectral: Spectral,
}

fn chunk_padding(chunk_samples: usize) -> usize {
    let mut padding = HOP - chunk_samples % HOP;
    if (chunk_samples + padding) / HOP % 2 == 0 {
        padding += HOP;
    }
    padding
}

fn blending_window(chunk_samples: usize) -> Vec<f32> {
    (0..chunk_samples)
        .map(|index| {
            let phase = 2.0 * std::f64::consts::PI * index as f64 / (chunk_samples - 1) as f64;
            ((0.5 - 0.5 * phase.cos()) as f32).max(1e-3)
        })
        .collect()
}

fn chunk_starts(frames: usize, hop_fraction: f64, chunk_samples: usize) -> Vec<usize> {
    let step = ((chunk_samples as f64 * hop_fraction) as usize).max(1);
    let mut starts: Vec<usize> = (0..=frames.saturating_sub(chunk_samples)).step_by(step).collect();
    if let Some(last) = starts.last() {
        if last + chunk_samples < frames {
            starts.push(frames - chunk_samples);
        }
    }
    starts
}

fn mean_and_deviation(left: &[f32], right: &[f32]) -> (f64, f64) {
    let count = left.len() as f64;
    let mean = left.iter().zip(right).map(|(a, b)| (*a as f64 + *b as f64) / 2.0).sum::<f64>() / count;
    let variance = left
        .iter()
        .zip(right)
        .map(|(a, b)| {
            let mono = (*a as f64 + *b as f64) / 2.0;
            (mono - mean) * (mono - mean)
        })
        .sum::<f64>()
        / (count - 1.0);
    (mean, variance.sqrt())
}

impl Separator {
    pub fn load(model_path: &Path, threads: usize, chunk_samples: usize) -> Result<Self, Box<dyn Error>> {
        let environment = ort::environment::init()
            .with_execution_providers([CPU::default().with_arena_allocator(false).build()])
            .build()?;
        let session = Session::builder(&environment)?
            .with_auto_device(AutoDevicePolicy::PreferCPU)?
            .with_intra_threads(threads)?
            .with_memory_pattern(false)?
            .with_execution_providers([CPU::default().with_arena_allocator(false).build()])?
            .commit_from_file(model_path)?;
        Ok(Self { chunk_samples, _environment: environment, session, spectral: Spectral::new() })
    }

    fn run_chunk(&mut self, chunk: &[Vec<f32>; 2]) -> Result<Vec<[Vec<f32>; 2]>, Box<dyn Error>> {
        let padding = chunk_padding(self.chunk_samples);
        let mut spectrograms: Vec<Spectrogram> = Vec::with_capacity(2);
        for channel in chunk {
            let mut padded = channel.clone();
            padded.resize(self.chunk_samples + padding, 0.0);
            spectrograms.push(self.spectral.stft(&padded));
        }
        let frames = spectrograms[0].frames;
        let plane = BINS * frames;
        let mut input = Vec::with_capacity(4 * plane);
        for spectrogram in &spectrograms {
            input.extend_from_slice(&spectrogram.real);
            input.extend_from_slice(&spectrogram.imaginary);
        }
        let tensor = Tensor::from_array(([1usize, 4, BINS, frames], input))?;
        let outputs = self.session.run(ort::inputs!["spectrum" => tensor])?;
        let (_, features) = outputs[0].try_extract_tensor::<f32>()?;
        let mut stems = Vec::with_capacity(4);
        for source in 0..4 {
            let mut channels: [Vec<f32>; 2] = [Vec::new(), Vec::new()];
            for (channel_index, channel_slot) in channels.iter_mut().enumerate() {
                let base = (source * 4 + channel_index * 2) * plane;
                let spectrogram = Spectrogram {
                    frames,
                    real: features[base..base + plane].to_vec(),
                    imaginary: features[base + plane..base + 2 * plane].to_vec(),
                };
                let mut waveform = self.spectral.istft(&spectrogram);
                waveform.truncate(self.chunk_samples);
                *channel_slot = waveform;
            }
            stems.push(channels);
        }
        Ok(stems)
    }

    pub fn separate(&mut self, left: &[f32], right: &[f32], hop_fraction: f64, normalise: bool) -> Result<StemAudio, Box<dyn Error>> {
        let frames = left.len();
        let (mean, deviation) = if normalise { mean_and_deviation(left, right) } else { (0.0, 1.0) };
        let normalise = |channel: &[f32]| -> Vec<f32> {
            channel.iter().map(|value| ((*value as f64 - mean) / deviation) as f32).collect()
        };
        let normalised = [normalise(left), normalise(right)];
        let chunk_samples = self.chunk_samples;
        let window = blending_window(chunk_samples);
        let mut output: StemAudio = (0..4).map(|_| [vec![0.0f32; frames], vec![0.0f32; frames]]).collect();
        let mut weight = vec![0.0f32; frames];
        for start in chunk_starts(frames, hop_fraction, chunk_samples) {
            let length = chunk_samples.min(frames - start);
            let chunk: [Vec<f32>; 2] = [
                normalised[0][start..start + length].to_vec(),
                normalised[1][start..start + length].to_vec(),
            ];
            let stems = self.run_chunk(&chunk)?;
            for (source, channels) in stems.iter().enumerate() {
                for (channel_index, waveform) in channels.iter().enumerate() {
                    for offset in 0..length {
                        output[source][channel_index][start + offset] += waveform[offset] * window[offset];
                    }
                }
            }
            for offset in 0..length {
                weight[start + offset] += window[offset];
            }
        }
        for source in output.iter_mut() {
            for channel in source.iter_mut() {
                for (index, value) in channel.iter_mut().enumerate() {
                    let blended = *value / weight[index].max(1e-6);
                    *value = ((blended as f64) * deviation + mean) as f32;
                }
            }
        }
        Ok(output)
    }
}
