# Integrating with Euphoria

This is a proposal for Euphoria's maintainers. Nothing here depends on Euphoria and nothing in Euphoria depends on this project.

## What is proposed

Replace htdemucs with SCNet-L (about 42M parameters, the same size as htdemucs, code MIT) run in 5.5 s chunks with a 25% overlap. SCNet-S (about 10M parameters) is the lighter alternative in the same family. Euphoria already runs ONNX models in Rust, so the integration is a new graph file plus the pre- and post-processing described below. `runtime/` is a small reference implementation of that processing that has been checked against the Python pipeline.

## Evidence

All scores are plain whole-song SDR per stem, averaged over the 50 MUSDB18-HQ test songs, with the same function for every model. They are not the median-based cSDR quoted in the Demucs README, so compare only within the table. Details and the other experiments are in `research/02-results.md`.

| Model and setting | Parameters | vocals | drums | bass | other | average |
|---|---|---|---|---|---|---|
| htdemucs (current) | 42M | 8.70 | 9.93 | 8.63 | 6.18 | 8.36 |
| SCNet-S, 11 s chunks, 2x overlap, ONNX graph through the pipeline | about 10M | 9.76 | 10.28 | 8.75 | 6.76 | 8.89 |
| SCNet-L, 5.5 s chunks, no overlap, ONNX graph through the pipeline | about 42M | 10.41 | 10.55 | 8.75 | 6.85 | 9.14 |
| SCNet-L, 5.5 s chunks, 25% overlap (hop 0.75), PyTorch | about 42M | 10.53 | 10.68 | 8.90 | 6.96 | 9.27 |
| SCNet-L, 5.5 s chunks, 2x overlap, PyTorch | about 42M | 10.66 | 10.80 | 9.05 | 7.10 | 9.41 |
| SCNet-L, 8.25 s chunks, 25% overlap (hop 0.75), PyTorch | about 42M | 10.73 | 10.90 | 9.13 | 7.07 | 9.46 |
| SCNet-L, 11 s chunks, 2x overlap, PyTorch | about 42M | 10.88 | 11.07 | 9.31 | 7.24 | 9.62 |

The ONNX score for SCNet-L equals the PyTorch score for the same setting (9.14), so the export loses nothing.

Cost, measured on one Mac with 4 threads on the same 7-minute stereo 44.1 kHz track (Euphoria's `scripts/stems/measure_cost.py` for htdemucs, `/usr/bin/time -l` for the runtime here):

| | Seconds per minute of audio | Peak resident memory |
|---|---|---|
| htdemucs in Euphoria | 39.7 | 2,434 MB |
| SCNet-S, `runtime/`, 11 s chunks, 2x overlap | 44.1 | 2,087 MB |
| SCNet-L, `runtime/`, 5.5 s chunks, no overlap | 31.9 | 2,818 MB |
| SCNet-L, `runtime/`, 5.5 s chunks, 25% overlap | about 42 (estimate, not measured) | 2,818 MB (same chunk, same model) |

The runtime's peak memory footprint (a different counter) was 2.65 GB for SCNet-S and 3.26 GB for SCNet-L. The ONNX Runtime memory arena and memory pattern are switched off, which is what keeps memory down. About 0.75 GB of the SCNet-L figure is audio held in memory (the 7-minute input and the four output stems); writing finished stems out as chunks complete would remove most of it. Overlap multiplies time: 2x overlap would roughly double SCNet-L's time for +0.27 dB (9.14 to 9.41). One pair of runs on a machine that was not idle, so repeat before relying on differences under about 10%.

Per song and per genre (SCNet-L at 5.5 s with no overlap against htdemucs, 50 songs): SCNet-L wins on 42 songs, with a mean gain of 0.77 dB. By stem it wins vocals on 46 songs (+1.71 dB), drums on 41 (+0.61), other on 40 (+0.67) and bass on 33 (+0.11, essentially a tie). By genre the gain is +1.03 dB on rock (17 songs, 17 wins), +1.67 on folk-acoustic (5, 5 wins), +0.90 on pop (8, 7 wins), +1.10 on hip-hop (3, 3 wins), +0.43 on electronic (8, 5 wins) and -0.03 on metal (2, 1 win). It loses on a drum & bass track and two tracks whose genre could not be found. Genre labels are approximate (artist-level style lines mapped by hand, four unknown) and the counts are small. These figures use the weakest blending (no overlap); the 25% overlap setting improves the average and mostly the bass, but the per-genre table has not been recomputed for it. Full table in `research/02-results.md`.

What this does not show:

- No listening test result yet. Euphoria's adoption rule needs one. `src/make_listening_page.py` builds a blind A/B page for it.
- Electronic music, the main DJ genre, has the smallest gain, and no cleared DJ-style test audio has been used.
- MUSDB only. The checkpoints appear to have been picked as the best epoch on MUSDB test, so the scores are probably slightly optimistic. SCNet-L's published figure is 9.70 with 4x overlap and 11 s chunks; the 9.62 here is the same checkpoint at 2x.
- Fine-tuning SCNet-S on extra data (pseudo-labelled FMA clips, Slakh) did not beat the untouched weights in any run, so there is no model of our own yet.

## Assessment

The measured gain is real on MUSDB (about +0.9 dB average for SCNet-L at 5.5 s chunks and 25% overlap, 42 of 50 songs better) but a blind listening test with 26 trials from one listener did not show a clear audible improvement: 10 preferred SCNet-L, 6 htdemucs, 10 no difference. SDR differences of 2 to 5 dB on a single stem were often not heard. The two large losses, on a drum & bass track and one other song, were heard and went to htdemucs. Electronic material, the main DJ genre, gains least on the metric and was mixed in the listening trials.

On this evidence SCNet-L is a modest improvement with a few worse failure cases, not a clear replacement. It costs about 0.4 GB more memory unless output is streamed, plus the integration work. A decision to adopt would need more listening trials on electronic material and a check of those failure cases. `research/02-results.md` has the details.

## Licence

The SCNet code is MIT. The weights were trained on MUSDB18-HQ, whose audio is research-only, and no weights licence is stated by the authors. The SCNet-L checkpoint may also have used another dataset; that is unverified. This is the same situation as htdemucs, whose README says the licence of its weights is not clear. A clean-licence retrain is not started.

## Model contract

The exported graph covers everything between the STFT and the inverse STFT. The chunk length is fixed when the graph is exported (`--chunk-samples`).

- Input `spectrum`, float32, shape `[1, 4, 2049, T]`. Channels are `left real, left imaginary, right real, right imaginary`. Axes are channel, frequency bin, frame.
- Output `features`, float32, shape `[1, 4, 4, 2049, T]`. Axes are source, channel, bin, frame. Sources are `drums, bass, other, vocals`. The four channels are laid out as in the input.
- For a chunk of N samples, `padding = 1024 - N % 1024`, plus 1024 more if `(N + padding) / 1024` is even, and `T = (N + padding) / 1024 + 1`. For N = 242,550 (5.5 s) that is padding 138 and T = 238. For N = 485,100 (11 s) it is padding 1,300 and T = 476.

Pre- and post-processing, all in the runtime:

1. Standardise the mixture: subtract the mean of the mono mix and divide by its standard deviation (sample standard deviation). Undo it on the outputs. Skipping this costs about 0.4 dB.
2. Chunk with the chosen overlap (hop N divided by the overlap), pad the last chunk with zeros, blend chunk outputs with a non-periodic Hann window clamped at 1e-3 and divide by the summed window.
3. For each chunk and channel: append `padding` zeros, then take the STFT with FFT size 4096, hop 1024, rectangular window, centred with reflect padding, scaled by 1/sqrt(4096), one-sided.
4. Run the graph. Inverse STFT with the same parameters (rectangular window, divide by the overlap count, scale to undo the forward normalisation), then drop the padding.

`runtime/src/spectral.rs` and `runtime/src/pipeline.rs` implement exactly this. `runtime/tests/golden.rs` compares the whole pipeline with the Python reference on a 25 s excerpt with the SCNet-S graph: signal-to-error is 120 dB or better on every stem and channel. The same code produced the SCNet-L timings above.

## Producing the graph

`src/export_scnet_onnx.py` loads a checkpoint from the Music-Source-Separation-Training repository with its matching config and exports the core with opset 17. Use the config that ships with the checkpoint; the SCNet-L checkpoint needs `config_musdb18_scnet_large_starrytong.yaml` from the v1.0.9 release, and a different Large config scored 0.9 dB lower. One change is needed: the separation network uses a real FFT along time, which ONNX cannot export, and the script replaces it with an equivalent matrix multiplication (difference from the FFT about 1e-6 for the fixed frame count).

## Pitfalls found

- ONNX Runtime on macOS registers the CoreML execution provider through its automatic device policy, even if only the CPU provider is requested. Loading this model then hangs while CoreML compiles it. Set the device policy to prefer the CPU.
- The ONNX Runtime memory arena and memory pattern options change peak memory a lot: for SCNet-L at 11 s chunks, 6.5 GB with the defaults against 3.6 GB without, at about 30% more time, for one chunk on a server CPU.
- Chunk length is the other memory lever: SCNet-L at 5.5 s chunks needs 2.0 GB instead of 3.6 GB, and costs 0.2 to 0.3 dB.
- The model needs fp32. The complex-number step does not work under bf16.
