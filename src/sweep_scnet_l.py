import argparse
import glob
import json
import os
import statistics

from score_per_song import SCNET_ORDER, STEM_NAMES, load_scnet, read, separate_scnet, stem_sdr

CONFIGURATIONS = [
    {"chunk": 242550, "step_fraction": 1.0},
    {"chunk": 242550, "step_fraction": 0.75},
    {"chunk": 242550, "step_fraction": 0.5},
    {"chunk": 363825, "step_fraction": 0.75},
    {"chunk": 363825, "step_fraction": 0.5},
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zfturbo-directory", required=True)
    parser.add_argument("--test-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()

    device = "cuda"
    network = load_scnet(arguments.zfturbo_directory, arguments.config, arguments.checkpoint, device)
    songs = sorted(glob.glob(os.path.join(arguments.test_root, "*")))
    results = []
    for configuration in CONFIGURATIONS:
        step_samples = int(configuration["chunk"] * configuration["step_fraction"])
        per_song = {}
        for directory in songs:
            mixture = read(os.path.join(directory, "mixture.wav"))
            estimate = separate_scnet(network, mixture, configuration["chunk"], 1, device, step_samples)
            scores = {}
            for index, name in enumerate(SCNET_ORDER):
                scores[name] = stem_sdr(estimate[index], read(os.path.join(directory, name + ".wav")))
            per_song[os.path.basename(directory)] = scores
        averages = {name: statistics.mean(per_song[song][name] for song in per_song) for name in STEM_NAMES}
        averages["average"] = statistics.mean(averages[name] for name in STEM_NAMES)
        results.append({**configuration, "averages": averages, "per_song": per_song})
        print(configuration, " ".join(f"{name}={value:.2f}" for name, value in averages.items()), flush=True)
        with open(arguments.output, "w") as output_file:
            json.dump(results, output_file, indent=1)


if __name__ == "__main__":
    main()
