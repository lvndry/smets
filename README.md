# smets

Small, open music source separation: split a song into drums, bass, other and vocals on a laptop CPU.

## Status

Early. There is no model of our own yet. What exists is a reference runtime in Rust for the SCNet family (SCNet-S at about 10M parameters, SCNet-L at about 42M), the export that feeds it, and the measurements that say it is worth proposing to a downstream app.

On the 50 MUSDB18-HQ test songs, scored with the same function for every model:

| Model | Parameters | vocals | drums | bass | other | average |
|---|---|---|---|---|---|---|
| htdemucs | 42M | 8.70 | 9.93 | 8.63 | 6.18 | 8.36 |
| SCNet-S, 11 s chunks, 2x overlap, ONNX graph through the pipeline | about 10M | 9.76 | 10.28 | 8.75 | 6.76 | 8.89 |
| SCNet-L, 5.5 s chunks, no overlap, ONNX graph through the pipeline | about 42M | 10.41 | 10.55 | 8.75 | 6.85 | 9.14 |
| SCNet-L, 5.5 s chunks, 25% overlap, PyTorch | about 42M | 10.53 | 10.68 | 8.90 | 6.96 | 9.27 |

Plain whole-song SDR per stem, averaged over songs. This is not the median-based cSDR in the Demucs README, so compare only within the table. The per-genre breakdown is rough (approximate labels, small counts; electronic gains least) and there is no listening test result yet, and the checkpoints appear to have been picked as the best epoch on MUSDB test, so the SCNet scores are probably slightly optimistic. On the same Mac and 7-minute track, SCNet-L at 5.5 s chunks ran at 31.9 s per minute of audio with 2.8 GB resident, against 39.7 s and 2.4 GB for htdemucs in Euphoria; see `docs/integrating.md`.

Fine-tuning SCNet-S on extra data (FMA clips labelled by a larger teacher, Slakh) did not beat the untouched weights in any run. The experiments and negative results are in `research/02-results.md`.

## Licence

No licence file yet. The SCNet code is MIT. The weights come from checkpoints trained on MUSDB18-HQ, whose audio is research-only, and the authors state no weights licence. htdemucs has the same open question. Treat the weights as unresolved until that is settled.

## Layout

- `runtime/`: the Rust crate and `smets` command. It does the STFT, chunking, normalisation and inverse STFT and runs the ONNX graph with ONNX Runtime on the CPU.
- `src/`: Python for export, evaluation and the training experiments.
- `docs/integrating.md`: how a Rust application would adopt the model, with the exact model contract.
- `research/`: plan, name notes and results log.

Model weights and audio are not in the repository.

## Run the reference runtime

You need an exported graph. Export one from a checkpoint and the config published with it (SCNet-S shown; for SCNet-L use its own config and `--chunk-samples 242550`, and pass the same value to `smets --chunk-samples`):

```sh
python src/export_scnet_onnx.py \
  --zfturbo-directory path/to/Music-Source-Separation-Training \
  --config path/to/Music-Source-Separation-Training/configs/config_musdb18_scnet.yaml \
  --checkpoint path/to/scnet_checkpoint_musdb18.ckpt \
  --output scnet_s_core.onnx
```

Then separate a 44.1 kHz stereo WAV file:

```sh
cd runtime
cargo build --release
./target/release/smets --model ../scnet_s_core.onnx --input song.wav --output-directory stems --threads 4
```

The output is `drums.wav`, `bass.wav`, `other.wav` and `vocals.wav`. Input must already be 44.1 kHz.

To check the runtime against the Python pipeline, set `STEMS_MODEL` to the graph and `STEMS_GOLDEN` to a folder made by `src/make_golden.py`, then run `cargo test --release`.
