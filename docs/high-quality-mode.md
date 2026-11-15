# A slow, higher-quality stems option

A proposal for Euphoria, not a change to it. DJs prepare tracks ahead of time, so a stems option that takes minutes per track instead of seconds can be acceptable if the result is clearly better. This note says which model would fill that role and what it costs. See `docs/integrating.md` for the faster model.

## The model

SCNet XL IHF, a checkpoint from the Music-Source-Separation-Training project (config `config_musdb18_scnet_xl_more_wide_v5.yaml`, file `model_scnet_ep_36_sdr_10.0891.ckpt`, 214 MB). The same SCNet architecture family as `runtime/`, so `smets` runs it unchanged once the graph is exported with `src/export_scnet_onnx.py`. The export worked at 11 s and 5.5 s chunks (matrix DFT replacing the FFT, difference about 1e-6).

## Quality

10.09 dB average on the 50 MUSDB18-HQ test songs, scored with the same function as the other models in `research/02-results.md` (vocals 11.42, bass 9.23, drums 11.81, other 7.88 per the project's published figures; vocals and other re-measured here), against 8.36 for htdemucs. That is +1.7 dB, a gap larger than the 2 to 5 dB per-stem differences that were often inaudible in the listening test for the faster model, but it was not blind-tested, and gains of this size are not guaranteed to be audible either.

Not measured: quality at the chunk and overlap settings below. Shorter chunks and less overlap lose some quality; SCNet-L lost 0.5 dB going from 11 s chunks with 2x overlap to 5.5 s chunks with no overlap.

## Cost, measured

ONNX Runtime on a server CPU, 8 threads, memory arena and pattern off, one chunk at a time (so memory excludes the audio held for the whole track, about 0.75 GB for 7 minutes). For comparison htdemucs ran at about 10 s per minute on the same machine.

| Chunk | Seconds per minute of audio, no overlap | with 2x overlap | Peak memory |
|---|---|---|---|
| 11 s | 51 | about 102 | 5.9 GB |
| 5.5 s | 46 | about 93 | 3.2 GB |

A 4-minute track therefore takes roughly 3 to 7 minutes on that server CPU. A laptop with 4 threads ran the faster SCNet models about four times slower than the server, so expect something like 12 to 30 minutes per track on a laptop CPU. That is an estimate; XL IHF was not run on a laptop.

## Suggested shape

- A "high quality" option, off by default, run as a background job per track, with progress and a time estimate shown before it starts.
- Two presets: 5.5 s chunks (about 3.2 GB plus audio buffers, for 16 GB laptops) and 11 s chunks (about 6 GB plus buffers, for machines with more memory), both with at least 25% overlap.
- Write finished stems out as chunks complete instead of holding the whole track.
- Allow the user to cancel, and keep the fast model as the instant default.

## Speed-ups worth testing

- GPU or Apple Neural Engine inference. A model this size should run in seconds on a GPU. ONNX Runtime's CoreML path hung while compiling the faster model on macOS (see `docs/integrating.md`), so this needs its own work, not a flag.
- 8-bit weights. Not tried; quality loss unknown.

## Caveats

- The weights licence is unresolved. The checkpoint was trained on MUSDB18-HQ, whose audio is research-only, and its authors state no weights licence. Same open question as htdemucs.
- The 10.09 figure is from the checkpoint's own validation, probably the best epoch on MUSDB test, so it is somewhat optimistic.
- No listening test, no per-genre numbers for this model, and no laptop measurement.
