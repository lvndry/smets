# A slow, higher-quality stems option

A proposal for Euphoria, not a change to it. DJs prepare tracks ahead of time, so a stems option that takes minutes per track instead of seconds can be acceptable if the result is clearly better. This note says which model would fill that role, what it costs in time and memory, and where more memory stops paying off. See `docs/integrating.md` for the faster model.

## The model

SCNet XL IHF, a checkpoint from the Music-Source-Separation-Training project (config `config_musdb18_scnet_xl_more_wide_v5.yaml`, file `model_scnet_ep_36_sdr_10.0891.ckpt`, 214 MB). Same SCNet architecture family as `runtime/`, so `smets` runs it once the graph is exported with `src/export_scnet_onnx.py` (exported at 11 s, 5.5 s, 4.1 s and 2.75 s chunks; ONNX file about 216 to 222 MB).

One difference matters: this checkpoint's config says `normalize: false`, so the mixture must not be standardised before the model. SCNet-S and SCNet-L use `normalize: true`. Standardising XL's input by mistake drops it from 9.51 to 7.92 dB at 5.5 s chunks. `smets` has `--no-normalise` for this.

## Quality against memory, measured

Plain whole-song SDR averaged over the 50 MUSDB18-HQ test songs, own overlap-add, hop 0.75 unless stated. Memory is ONNX Runtime peak resident size for the model and one chunk, arena and memory pattern off, on a server CPU with 8 threads, excluding the audio held for the whole track (about 0.1 GB per minute of audio, avoidable by streaming the output). For comparison htdemucs scores 8.36 in about 2.5 GB.

| Model, chunk | Average | vocals | drums | bass | other | Memory | Seconds per minute of audio, no overlap |
|---|---|---|---|---|---|---|---|
| SCNet-L, 5.5 s | 9.27 | 10.53 | 10.68 | 8.90 | 6.96 | 2.0 GB | 18.5 |
| XL IHF, 2.75 s | 8.86 | 10.48 | 10.44 | 7.82 | 6.70 | 1.75 GB | 40.2 |
| XL IHF, 4.1 s | 9.30 | 10.84 | 10.91 | 8.29 | 7.15 | 2.45 GB | 42.3 |
| XL IHF, 5.5 s | 9.51 | 11.00 | 11.14 | 8.54 | 7.36 | 3.2 GB | 46 |
| XL IHF, 11 s, hop 0.5 | 9.91 | 11.29 | 11.61 | 9.03 | 7.70 | 5.9 GB | 51 (about 102 at hop 0.5) |

The published figure for XL IHF at 11 s with 4x overlap is 10.09.

What this says:

- Memory is roughly 0.3 GB plus 0.5 GB per second of chunk. ONNX Runtime options (prepacking, graph optimisation) saved only about 0.1 GB.
- Below about 3 GB, SCNet-L is the better model: at 5.5 s it scores 9.27 in 2.0 GB, where XL needs 4.1 s and 2.45 GB for 9.30.
- At 3.2 GB XL gains 0.24 dB over SCNet-L and is about 2.5 times slower. Going to 5.9 GB gains another 0.4 dB.
- XL only separates itself from the fast model with 6 GB or more: 9.91 against 8.36 for htdemucs, +1.55 dB. The listening test of the fast model showed 2 to 5 dB single-stem differences were often inaudible, so even that gain is not guaranteed to be heard. It was not blind-tested.

## Cost in time

A 4-minute track with XL at 11 s and hop 0.5 takes roughly 7 minutes on the 8-core server, and 5.5 s chunks with no overlap about 3 minutes. A laptop with 4 threads ran the faster SCNet models about four times slower than the server, so expect something like 12 to 30 minutes per track on a laptop CPU. That is an estimate; XL was not run on a laptop.

## Storage

The ONNX graph is about 220 MB, comparable to htdemucs (about 170 MB). Half-precision weights would halve it; not tried. Prepared stems are about 170 MB per 4-minute track as 16-bit WAV, depending on the format Euphoria writes.

## Suggested shape

- Off by default, run as a background job per track, with a time estimate shown first and a cancel button.
- Pick the chunk length from available memory: SCNet-L at 5.5 s (about 2 GB) on small machines, XL at 5.5 s (about 3.2 GB) on 16 GB machines, XL at 11 s (about 6 GB) on machines with more.
- Write finished stems out as chunks complete, not after the whole track.
- Keep the fast model as the instant default.

## Speed-ups worth testing

- GPU or Apple Neural Engine inference: a model this size should take seconds there. ONNX Runtime's CoreML path hung while compiling the faster model on macOS (see `docs/integrating.md`), so this needs its own work.
- 8-bit or half-precision weights: not tried, quality loss unknown. They would help the file size more than the memory, which is dominated by activations.

## Caveats

- The weights licence is unresolved. The checkpoint was trained on MUSDB18-HQ, whose audio is research-only, and its authors state no weights licence. Same open question as htdemucs.
- The checkpoint's own figure is probably the best epoch on MUSDB test, so the scores are somewhat optimistic.
- No listening test, no per-genre numbers for XL, and no laptop measurement.

## Measured on a Mac

One Mac (10 cores, 4 performance cores, 24 GB), 4 threads at low priority, the same 7-minute stereo 44.1 kHz track, run back to back:

| | Seconds per minute of audio | Peak resident memory |
|---|---|---|
| Euphoria htdemucs | 16.7 | 2,504 MB |
| XL IHF through `smets`, 5.5 s chunks, hop 0.75, `--no-normalise` | 141.3 | 4,440 MB (peak footprint 5.2 GB) |

About 8.5 times slower than htdemucs: a 4-minute track takes about 9.4 minutes. No swapping. Machine load rose from about 6 to 21 during the XL run, so the time is an upper bound. The memory includes about 0.75 GB of audio held for the whole track. The output of `smets` for XL matches the PyTorch reference to 41 to 52 dB signal-to-error on a 25 s excerpt.
