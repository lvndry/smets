import argparse
import os
import sys

import soundfile
import torch

from score_per_song import SCNET_ORDER, load_scnet, read, separate_scnet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zfturbo-directory", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--excerpt", required=True)
    parser.add_argument("--chunk-samples", type=int, required=True)
    parser.add_argument("--hop-fraction", type=float, required=True)
    parser.add_argument("--no-normalise", action="store_true")
    parser.add_argument("--output-directory", required=True)
    arguments = parser.parse_args()

    device = "cuda"
    network = load_scnet(arguments.zfturbo_directory, arguments.config, arguments.checkpoint, device)
    mixture = read(arguments.excerpt)
    step = int(arguments.chunk_samples * arguments.hop_fraction)
    estimate = separate_scnet(network, mixture, arguments.chunk_samples, 1, device, step, not arguments.no_normalise)
    os.makedirs(arguments.output_directory, exist_ok=True)
    for index, name in enumerate(SCNET_ORDER):
        soundfile.write(os.path.join(arguments.output_directory, f"expected_{name}.wav"), estimate[index].numpy().T, 44100, subtype="FLOAT")
    print("written", tuple(estimate.shape))


if __name__ == "__main__":
    main()
