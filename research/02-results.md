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

## SCNet-L chunk and hop sweep (PyTorch, `src/sweep_scnet_l.py`)

Own overlap-add with Hann blending, standardised input, 50 MUSDB test songs. Hop is the step between chunk starts as a fraction of the chunk.

| Chunk | Hop | vocals | drums | bass | other | average | Compute vs hop 1.0 |
|---|---|---|---|---|---|---|---|
| 5.5 s | 1.0 | 10.41 | 10.55 | 8.74 | 6.85 | 9.14 | 1.0x |
| 5.5 s | 0.75 | 10.53 | 10.68 | 8.90 | 6.96 | 9.27 | 1.33x |
| 5.5 s | 0.5 | 10.61 | 10.74 | 8.99 | 7.04 | 9.34 | 2.0x |
| 8.25 s | 0.75 | 10.73 | 10.90 | 9.13 | 7.07 | 9.46 | about 1.4x |
| 8.25 s | 0.5 | 10.79 | 10.97 | 9.19 | 7.13 | 9.52 | about 2.1x |

Hop 1.0 means chunks that only touch, which leaves near-zero blending weight at every chunk boundary; bass suffers most. Songs hit hardest at hop 1.0: "PR - Happy Daze" bass 10.7 to 0.0 dB, "Speak Softly - Broken Man" bass 17.4 to 12.5, "Skelpolu - Resurrection" bass 15.5 to 13.6, "Hollow Ground - Ill Fate" bass 2.6 to 0.0.

The runtime's default is now 5.5 s chunks with hop 0.75. Its speed was not measured because the Mac was under heavy load (load average 144); the estimate is the hop 1.0 measurement times 1.33. Peak memory does not depend on the hop.

## Per song and per genre (`src/score_per_song.py`, `src/genre_report.py`)

htdemucs against SCNet-S (11 s, 2x) and SCNet-L (5.5 s, hop 1.0), 50 songs. SCNet-L beat htdemucs on 42 of 50 songs: mean +0.77 dB, median +0.97. By stem: vocals 46 of 50 songs (+1.71), drums 41 (+0.61), bass 33 (+0.11), other 40 (+0.67).

Genre labels in `musdb_test_genres.csv` come from each artist's "Style" line in the Cambridge Mixing Secrets library, mapped by hand onto a fixed list. They are per artist, not per track, were not checked against the raw page, and four songs are unknown. Counts per genre are small.

| Genre | Songs | htdemucs | SCNet-S | SCNet-L (hop 1.0) | SCNet-L minus htdemucs | SCNet-L wins |
|---|---|---|---|---|---|---|
| rock | 17 | 8.68 | 9.52 | 9.71 | +1.03 | 17/17 |
| folk-acoustic | 5 | 9.31 | 10.71 | 10.98 | +1.67 | 5/5 |
| pop | 8 | 8.60 | 9.17 | 9.50 | +0.90 | 7/8 |
| hip-hop | 3 | 9.21 | 9.77 | 10.31 | +1.10 | 3/3 |
| world | 3 | 7.78 | 8.10 | 8.91 | +1.13 | 3/3 |
| electronic | 8 | 8.20 | 8.67 | 8.63 | +0.43 | 5/8 |
| metal | 2 | 5.81 | 6.01 | 5.79 | -0.03 | 1/2 |
| unknown | 4 | 6.76 | 5.14 | 5.65 | -1.11 | 1/4 |

Electronic is the weakest genre for the gain, which matters for a DJ application. SCNet-L loses on the drum & bass track and on both "PR" tracks. These per-genre numbers use hop 1.0, which the sweep shows is the weakest blending; they have not been recomputed at hop 0.75.

## Listening test

`src/make_listening_clips.py` and `src/make_listening_page.py` build a local blind A/B page (36 trials: 12 clips of 12 s, each for vocals, instrumental and drums, htdemucs against SCNet-L at 5.5 s and hop 0.75, order randomised). The audio is MUSDB (research-only) and stays in the git-ignored `listening-test/` folder. No results yet.
