# Integrating with Euphoria

This is a proposal for Euphoria's maintainers. Nothing here depends on Euphoria and nothing in Euphoria depends on this project.

## What is proposed

Replace htdemucs with SCNet-S (about 10M parameters, code MIT) as the stems model. Euphoria already runs ONNX models in Rust, so the integration is a new graph file plus the pre- and post-processing described below. `runtime/` is a small reference implementation of that processing that has been checked against the Python pipeline.

## Evidence

All numbers are plain whole-song SDR per stem, averaged over the 50 MUSDB18-HQ test songs, with the same function for every model. They are not the median-based cSDR quoted in the Demucs README, so compare only within the table. Details and the other experiments are in `research/02-results.md`.

| Model | Parameters | vocals | drums | bass | other | average |
|---|---|---|---|---|---|---|
| htdemucs (current) | 42M | 8.70 | 9.93 | 8.63 | 6.18 | 8.36 |
| SCNet-S through the ONNX graph and the Rust-equivalent pipeline | about 10M | 9.76 | 10.28 | 8.75 | 6.76 | 8.89 |

Cost, measured on one Mac with 4 threads on the same 7-minute stereo 44.1 kHz track (Euphoria's `scripts/stems/measure_cost.py` for htdemucs, `/usr/bin/time -l` for the runtime here):

| | Seconds per minute of audio | Peak resident memory |
|---|---|---|
| htdemucs in Euphoria | 39.7 | 2,434 MB |
| SCNet-S, `runtime/`, 2x overlap | 44.1 | 2,087 MB |

The runtime's peak memory footprint (a different counter) was 2.65 GB. The ONNX Runtime memory arena and memory pattern are switched off, which is what keeps memory down. With no chunk overlap the time should roughly halve; that setting scored 8.72 on MUSDB test against 8.94 with 2x overlap in PyTorch. Not re-measured in Rust. The machine was not idle, so treat these as one pair of runs and repeat them before relying on a 10% difference.

What this does not show:

- No per-genre breakdown and no listening test yet. Euphoria's adoption rule needs both.
- MUSDB only. The SCNet-S checkpoint appears to have been picked as the best epoch on MUSDB test, so 8.9 is probably slightly optimistic.
- Fine-tuning on extra data (pseudo-labelled FMA clips, Slakh) did not beat the untouched weights in any run, so there is no model of our own yet.

## Licence

The SCNet code is MIT. The weights were trained on MUSDB18-HQ, whose audio is research-only, and no weights licence is stated by the authors. This is the same situation as htdemucs, whose README says the licence of its weights is not clear. A clean-licence retrain is not started.

## Model contract

The exported graph covers everything between the STFT and the inverse STFT.

- Input `spectrum`, float32, shape `[1, 4, 2049, 476]`. Channels are `left real, left imaginary, right real, right imaginary`. Axes are channel, frequency bin, frame.
- Output `features`, float32, shape `[1, 4, 4, 2049, 476]`. Axes are source, channel, bin, frame. Sources are `drums, bass, other, vocals`. The four channels are laid out as in the input.
- The shape is fixed to one 11 s chunk (485,100 samples at 44.1 kHz). The runtime chunks the song.

Pre- and post-processing, all in the runtime:

1. Standardise the mixture: subtract the mean of the mono mix and divide by its standard deviation (sample standard deviation). Undo it on the outputs. Skipping this costs about 0.4 dB.
2. Chunk with 2x overlap (hop 242,550 samples), pad the last chunk with zeros, blend chunk outputs with a non-periodic Hann window clamped at 1e-3 and divide by the summed window.
3. For each chunk and channel: append `padding` zeros, where `padding = 1024 - 485100 % 1024`, plus 1024 more if the resulting frame count is even (1,300 samples here). Take the STFT with FFT size 4096, hop 1024, rectangular window, centred with reflect padding, scaled by 1/sqrt(4096), one-sided.
4. Run the graph. Inverse STFT with the same parameters (rectangular window, divide by the overlap count, scale to undo the forward normalisation), then drop the padding.

`runtime/src/spectral.rs` and `runtime/src/pipeline.rs` implement exactly this. `runtime/tests/golden.rs` compares the whole pipeline with the Python reference on a 25 s excerpt: signal-to-error is 120 dB or better on every stem and channel.

## Producing the graph

`src/export_scnet_onnx.py` loads the SCNet-S checkpoint from the Music-Source-Separation-Training repository and exports the core with opset 17. One change is needed: the separation network uses a real FFT along time, which ONNX cannot export, and the script replaces it with an equivalent matrix multiplication (difference from the FFT about 1e-6 for the fixed 476 frames).

## Pitfalls found

- ONNX Runtime on macOS registers the CoreML execution provider through its automatic device policy, even if only the CPU provider is requested. Loading this model then hangs while CoreML compiles it. Set the device policy to prefer the CPU.
- The ONNX Runtime memory arena and memory pattern options change peak memory a lot: 2.2 GB with the defaults against 1.4 GB without, at about 40% more time, for one chunk on a server CPU.
- The model needs fp32. The complex-number step does not work under bf16.
