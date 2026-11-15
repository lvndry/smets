import argparse
import json
import os

import soundfile
import torch

from score_per_song import SCNET_ORDER, load_scnet, read, separate_htdemucs, separate_scnet

SAMPLE_RATE = 44100


def write_flac(path, audio):
    soundfile.write(path, audio.T.numpy(), SAMPLE_RATE, format="FLAC", subtype="PCM_16")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zfturbo-directory", required=True)
    parser.add_argument("--test-root", required=True)
    parser.add_argument("--scnet-l-config", required=True)
    parser.add_argument("--scnet-l-checkpoint", required=True)
    parser.add_argument("--clips", required=True, help="JSON list of {song, start_seconds}")
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--output-directory", required=True)
    arguments = parser.parse_args()

    device = "cuda"
    from demucs.pretrained import get_model

    htdemucs = get_model("htdemucs").eval().to(device)
    scnet_l = load_scnet(arguments.zfturbo_directory, arguments.scnet_l_config, arguments.scnet_l_checkpoint, device)
    os.makedirs(arguments.output_directory, exist_ok=True)
    with open(arguments.clips) as clips_file:
        clips = json.load(clips_file)

    for index, clip in enumerate(clips):
        song, start = clip["song"], int(clip["start_seconds"] * SAMPLE_RATE)
        length = int(arguments.seconds * SAMPLE_RATE)
        mixture = read(os.path.join(arguments.test_root, song, "mixture.wav"))
        models = {
            "htdemucs": dict(zip(htdemucs.sources, separate_htdemucs(htdemucs, mixture, device))),
            "candidate": dict(zip(SCNET_ORDER, separate_scnet(scnet_l, mixture, 242550, 1, device, int(242550 * 0.75)))),
        }
        clip_directory = os.path.join(arguments.output_directory, f"clip{index:02d}")
        os.makedirs(clip_directory, exist_ok=True)
        write_flac(os.path.join(clip_directory, "mixture.flac"), mixture[:, start : start + length])
        for model_name, stems in models.items():
            instrumental = stems["drums"] + stems["bass"] + stems["other"]
            for stem_name, audio in (("vocals", stems["vocals"]), ("instrumental", instrumental), ("drums", stems["drums"])):
                write_flac(os.path.join(clip_directory, f"{model_name}_{stem_name}.flac"), audio[:, start : start + length])
        print("wrote", clip_directory, song, flush=True)


if __name__ == "__main__":
    main()
