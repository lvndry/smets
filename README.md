# smets

Split a song into drums, bass, vocals and other, on a laptop CPU, with a model and code anyone can use.

## What this is

smets is the groundwork for an open stem separation model for [Euphoria](https://github.com/lvndry/euphoria), a free DJ app. The goal is a small model (about 10M parameters) whose code **and weights** are permissively licensed, so Euphoria can ship it to every DJ, including those who play paid gigs.

## Status: early research, no model yet

There is no smets model today. What exists:

- **A reference runtime** in Rust (`runtime/`) that runs an exported SCNet model on the CPU: STFT, chunking, normalisation, inverse STFT, WAV output.
- **An export script** (`src/export_scnet_onnx.py`) that turns a published SCNet checkpoint into the ONNX graph the runtime loads.
- **Benchmarks and experiments** (`research/`, `src/`) comparing SCNet with htdemucs, the model Euphoria uses now.

The benchmarks use existing SCNet checkpoints trained on MUSDB18-HQ, whose audio is research-only. Those weights have no stated licence, so **they cannot be shipped**. They show what the architecture can do, not what smets will release. Training a model on data with a clean licence is the next step; the plan and the data audit are in `research/00-plan.md`.

## Results so far

Average SDR in dB over the 50 MUSDB18-HQ test songs, same scoring function for every model (higher is better):

| Model | Parameters | vocals | drums | bass | other | average |
|---|---|---|---|---|---|---|
| htdemucs | 42M | 8.70 | 9.93 | 8.63 | 6.18 | 8.36 |
| SCNet-S (11 s chunks, 2x overlap, ONNX) | about 10M | 9.76 | 10.28 | 8.75 | 6.76 | 8.89 |
| SCNet-L (5.5 s chunks, no overlap, ONNX) | about 42M | 10.41 | 10.55 | 8.75 | 6.85 | 9.14 |
| SCNet-L (5.5 s chunks, 25% overlap, PyTorch) | about 42M | 10.53 | 10.68 | 8.90 | 6.96 | 9.27 |

Read these with care:

- This is plain whole-song SDR, not the median-based cSDR in the Demucs README, so compare only within the table.
- The SCNet checkpoints were probably picked as the best epoch on the MUSDB test set, so their scores are likely a little optimistic.
- The listening test is only partly done.
- Speed, measured on one Mac with a 7-minute track: SCNet-L at 5.5 s chunks took 31.9 s per minute of audio and 2.8 GB of memory. htdemucs in Euphoria took 39.7 s and 2.4 GB.
- Fine-tuning SCNet-S on extra data (FMA clips labelled by a larger teacher, Slakh) never beat the original weights. See `research/02-results.md`.

## Try the runtime

You need Rust, Python with the dependencies of [Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training), and an SCNet checkpoint with its config. Weights and audio are not in this repository.

1. Export the graph (SCNet-S shown; for SCNet-L use its own config and add `--chunk-samples 242550`):

   ```sh
   python src/export_scnet_onnx.py \
     --zfturbo-directory path/to/Music-Source-Separation-Training \
     --config path/to/Music-Source-Separation-Training/configs/config_musdb18_scnet.yaml \
     --checkpoint path/to/scnet_checkpoint_musdb18.ckpt \
     --output scnet_s_core.onnx
   ```

2. Separate a 44.1 kHz stereo WAV file (the input must already be 44.1 kHz; pass the same `--chunk-samples` value you exported with):

   ```sh
   cd runtime
   cargo build --release
   ./target/release/smets --model ../scnet_s_core.onnx --input song.wav --output-directory stems --threads 4
   ```

   The output is `drums.wav`, `bass.wav`, `other.wav` and `vocals.wav`.

To check the runtime against the Python pipeline, set `STEMS_MODEL` to the graph and `STEMS_GOLDEN` to a folder made by `src/make_golden.py`, then run `cargo test --release` in `runtime/`.

## Where to read next

- `docs/integrating.md`: the proposal for Euphoria, with the exact model contract.
- `docs/high-quality-mode.md`: a slower, higher-quality option and its measured cost.
- `research/00-plan.md`: the plan for a model with a clean licence.
- `research/02-results.md`: every experiment, including the ones that failed.
- `research/01-names.md`: notes on the project name.

## Layout

- `runtime/`: the Rust crate and the `smets` command.
- `src/`: Python for export, evaluation and the training experiments.
- `docs/`: integration and design notes.
- `research/`: plans, results and the raw score files.

## Licence

There is no licence file yet. The SCNet code is MIT. The weights used for the benchmarks are unresolved, as explained above, and htdemucs has the same open question. A licence for smets itself will be added with the first model that can be shipped.
