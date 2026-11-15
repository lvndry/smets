use std::sync::Arc;

use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};

pub const FFT_SIZE: usize = 4096;
pub const HOP: usize = 1024;
pub const BINS: usize = FFT_SIZE / 2 + 1;

pub struct Spectrogram {
    pub frames: usize,
    pub real: Vec<f32>,
    pub imaginary: Vec<f32>,
}

pub struct Spectral {
    forward: Arc<dyn Fft<f32>>,
    inverse: Arc<dyn Fft<f32>>,
}

impl Default for Spectral {
    fn default() -> Self {
        Self::new()
    }
}

fn reflect_index(index: isize, length: usize) -> usize {
    let last = length as isize - 1;
    let mut position = index;
    if position < 0 {
        position = -position;
    }
    if position > last {
        position = 2 * last - position;
    }
    position as usize
}

impl Spectral {
    pub fn new() -> Self {
        let mut planner = FftPlanner::<f32>::new();
        Self {
            forward: planner.plan_fft_forward(FFT_SIZE),
            inverse: planner.plan_fft_inverse(FFT_SIZE),
        }
    }

    pub fn stft(&self, signal: &[f32]) -> Spectrogram {
        assert!(signal.len() % HOP == 0, "signal length must be a multiple of the hop");
        let frames = signal.len() / HOP + 1;
        let half = (FFT_SIZE / 2) as isize;
        let scale = 1.0 / (FFT_SIZE as f32).sqrt();
        let mut real = vec![0.0f32; BINS * frames];
        let mut imaginary = vec![0.0f32; BINS * frames];
        let mut buffer = vec![Complex::new(0.0f32, 0.0f32); FFT_SIZE];
        for frame in 0..frames {
            let origin = (frame * HOP) as isize - half;
            for (offset, slot) in buffer.iter_mut().enumerate() {
                let sample = signal[reflect_index(origin + offset as isize, signal.len())];
                *slot = Complex::new(sample, 0.0);
            }
            self.forward.process(&mut buffer);
            for bin in 0..BINS {
                real[bin * frames + frame] = buffer[bin].re * scale;
                imaginary[bin * frames + frame] = buffer[bin].im * scale;
            }
        }
        Spectrogram { frames, real, imaginary }
    }

    pub fn istft(&self, spectrogram: &Spectrogram) -> Vec<f32> {
        let frames = spectrogram.frames;
        let full_length = FFT_SIZE + HOP * (frames - 1);
        let mut accumulator = vec![0.0f32; full_length];
        let mut envelope = vec![0.0f32; full_length];
        let mut buffer = vec![Complex::new(0.0f32, 0.0f32); FFT_SIZE];
        let scale = (FFT_SIZE as f32).sqrt() / FFT_SIZE as f32;
        for frame in 0..frames {
            for bin in 0..BINS {
                let real = spectrogram.real[bin * frames + frame];
                let imaginary = if bin == 0 || bin == BINS - 1 {
                    0.0
                } else {
                    spectrogram.imaginary[bin * frames + frame]
                };
                buffer[bin] = Complex::new(real, imaginary);
                if bin > 0 && bin < BINS - 1 {
                    buffer[FFT_SIZE - bin] = Complex::new(real, -imaginary);
                }
            }
            self.inverse.process(&mut buffer);
            let origin = frame * HOP;
            for offset in 0..FFT_SIZE {
                accumulator[origin + offset] += buffer[offset].re * scale;
                envelope[origin + offset] += 1.0;
            }
        }
        let start = FFT_SIZE / 2;
        let length = HOP * (frames - 1);
        (start..start + length)
            .map(|index| {
                if envelope[index] > 1e-11 {
                    accumulator[index] / envelope[index]
                } else {
                    accumulator[index]
                }
            })
            .collect()
    }
}
