# Results log

All scores are plain whole-song SDR per stem, averaged over songs, on the 50 MUSDB18-HQ test songs, computed with ZFTurbo's `valid.py` (htdemucs scored with `src/score_htdemucs_test.py`, same definition). They are not the median-based cSDR in the Demucs README, so compare only within this table.

## Quality on MUSDB18-HQ test

| Model | Params | vocals | drums | bass | other | average |
|---|---|---|---|---|---|---|
| htdemucs (current Euphoria model) | 42M | 8.70 | 9.93 | 8.63 | 6.18 | 8.36 |
| SCNet-S, 2x overlap | ~10M | 9.80 | 10.34 | 8.81 | 6.82 | 8.94 |
| SCNet-S, no overlap | ~10M | 9.59 | 10.13 | 8.56 | 6.60 | 8.72 |
| SCNet-L, wrong config (ignore), 2x overlap | ~42M | 10.15 | 10.35 | 8.06 | 6.27 | 8.71 |
| SCNet-L (starrytong checkpoint, its own config), 11 s chunk, 2x overlap | ~42M | 10.88 | 11.07 | 9.31 | 7.24 | 9.62 |
| SCNet-L, 11 s chunk, no overlap | ~42M | 10.71 | 10.88 | 9.10 | 7.05 | 9.44 |
| SCNet-L, 8.25 s chunk, 2x overlap | ~42M | 10.82 | 11.02 | 9.24 | 7.19 | 9.57 |
| SCNet-L, 5.5 s chunk, 2x overlap | ~42M | 10.66 | 10.80 | 9.05 | 7.10 | 9.41 |
| SCNet-L, 5.5 s chunk, no overlap | ~42M | 10.41 | 10.55 | 8.74 | 6.85 | 9.14 |
| SCNet XL IHF (teacher) | n/a | 11.42 | n/a | n/a | 7.88 | 10.09 |

SCNet-L needs the config published with its checkpoint (`config_musdb18_scnet_large_starrytong.yaml` in the v1.0.9 release of Music-Source-Separation-Training); the earlier 8.71 used a different Large config. All rows above SCNet XL IHF are PyTorch with ZFTurbo's `valid.py`; the published figure for this checkpoint is 9.70 at 4x overlap.

## CPU cost, 7-minute stereo 44.1 kHz track, 8 threads, PyTorch fp32, server CPU

| Model | Overlap | Seconds per minute of audio | Peak memory |
|---|---|---|---|
| htdemucs | 25% | 10.3 | 2.53 GB |
| SCNet-S | 2x, batch 1 | 9.7 | 3.39 GB |
| SCNet-S | none, batch 1 | 5.2 | 3.38 GB |
| SCNet-L | 2x, batch 1 | 41.8 | 3.61 GB |
| SCNet-L | none, batch 1 | 21.3 | 3.63 GB |
| SCNet XL IHF | default (4x) | 190.7 | 6.79 GB |

SCNet XL IHF is about 18x slower than htdemucs and cannot ship; it is only a teacher.

## SCNet-L cost, ONNX Runtime with the arena and memory pattern off

Server CPU, 8 threads, one chunk (core plus one chunk, no audio buffers):

| Chunk | Seconds per minute, no overlap | Peak memory |
|---|---|---|
| 11 s | 21.8 | 3.62 GB |
| 5.5 s | 18.5 | 2.01 GB |

The 5.5 s graph (`--chunk-samples 242550`, input `[1, 4, 2049, 238]`) run through the Rust runtime on the Mac, 4 threads, no overlap, 7-minute track: 31.9 s per minute, 2.82 GB maximum resident, 3.26 GB peak footprint. The same Mac and track with Euphoria's htdemucs: 39.7 s per minute, 2.43 GB. Peak memory includes about 0.75 GB of audio buffers that streaming the stems out would remove.

## ONNX export of SCNet-S (src/export_scnet_onnx.py)

The graph covers everything between the STFT and the iSTFT, with a fixed 11 s chunk (485100 samples, input `[1, 4, 2049, 476]`). The STFT (n_fft 4096, hop 1024, rectangular window, centre padding, normalised) and iSTFT stay outside, in the runtime. The real FFT along time in the separation network is replaced by DFT matrices (difference from the FFT 1.2e-6); the exported graph matches PyTorch to 3.6e-4 maximum absolute difference on the feature tensor. A waveform-level check on real audio is still to do.

ONNX Runtime alone, CPU, 8 threads, one chunk at a time:

| Setting | Seconds per chunk | Seconds per minute at 2x overlap | Peak memory (model and one chunk) |
|---|---|---|---|
| default arena | 0.72 | 7.9 | 2.22 GB |
| arena and memory pattern off | 1.02 | 11.1 | 1.40 GB |

With a 7-minute track held in memory (input 0.15 GB, four output stems 0.6 GB) the lean setting gives roughly 2.1 GB, under the 2.5 GB limit, at about the speed of htdemucs. This is an estimate until the Rust runtime measures it.

## Fine-tuning SCNet-S

| Run | Settings | MUSDB test average |
|---|---|---|
| untouched SCNet-S | 2x overlap | 8.94 |
| `scnet_s_pseudo`, step 4000 | lr 1e-4, 50% pseudo, 10% Slakh | 8.83 |
| `scnet_s_pseudo`, step 20000 | same | 7.93 (forgetting; stopped) |
| `scnet_s_gentle`, step 6000 | lr 1e-5, 30% pseudo, no Slakh | 8.87 |
| `scnet_s_gentle`, step 20000 | same | 8.61 |
| `scnet_s_control`, step 6000 | lr 1e-5, MUSDB only, my L1 + spectral loss | 8.65 |
| `scnet_s_mse`, step 6000 | lr 1e-5, 30% pseudo, waveform MSE, 11 s crop | 8.90 |
| `scnet_s_mse`, step 20000 | same | 8.70 |

Every fine-tune recipe ended below the untouched SCNet-S. The pseudo-labelled data helped against the MUSDB-only control at equal steps (8.87 vs 8.65), and the MSE loss was the least harmful, but score fell as training continued in all runs. FMA pseudo-label stems carry much less energy above 16 kHz than MUSDB (5x less for vocals, about 160x for other), so the student may learn the mp3 cutoff. The released SCNet-S checkpoint is named by its validation score, so it was probably picked as the best epoch on MUSDB test; 8.94 is likely a little optimistic and continued training would regress towards the mean.

Decision: stop fine-tuning, propose the untouched SCNet-S.

## End-to-end ONNX check (src/score_onnx.py)

Exported core in ONNX Runtime, STFT and iSTFT in torch, 11 s chunks with 2x overlap and Hann blending, mixture standardised by mean and standard deviation before the model and restored after (`normalize: true` in the ZFTurbo inference config). 50 MUSDB test songs.

| Pipeline | vocals | drums | bass | other | average |
|---|---|---|---|---|---|
| PyTorch, ZFTurbo `valid.py` | 9.80 | 10.34 | 8.81 | 6.82 | 8.94 |
| ONNX core, my chunking, no normalisation | 9.50 | 9.95 | 8.19 | 6.35 | 8.50 |
| ONNX core, my chunking, with normalisation | 9.76 | 10.28 | 8.75 | 6.76 | 8.89 |

The runtime must apply the normalisation. The remaining 0.05 dB is probably the chunk blending window.

## Student from scratch (Moises-Light)

`fullband_large` (5.5M) never learned bass or other (about -2 dB). `fullband_large_wide` (9.7M) reached about 6.3 dB on a 30 s validation monitor after 80k steps, far below htdemucs. Stopped in favour of fine-tuning SCNet-S.

## Decision

Fine-tune SCNet-S (MIT code, 10M parameters) on MUSDB, Slakh and 7,196 FMA clips pseudo-labelled by SCNet XL IHF. Test-set score of the EMA weights is the only number that counts; the in-training validation songs were already seen by the pretrained SCNet-S.

## Open items

- Peak memory is above htdemucs in PyTorch (3.4 GB vs 2.5 GB). Measure in the Rust runtime and try smaller chunks.
- ONNX export needs the STFT and iSTFT moved out of the graph.
- Per-genre evaluation and a listening test are not done.
- Weights licence: code is MIT, weights come from MUSDB-trained checkpoints; the clean-licence fallback retrain is not started.
