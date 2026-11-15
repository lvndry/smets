import argparse
import glob
import os

import soundfile
import torch
from demucs.apply import apply_model
from demucs.pretrained import get_model

STEM_NAMES = ["vocals", "drums", "bass", "other"]


def stem_sdr(estimate, reference):
    noise = (reference - estimate).pow(2).sum()
    signal = reference.pow(2).sum()
    return 10 * torch.log10((signal + 1e-8) / (noise + 1e-8)).item()


def read(path):
    audio, _ = soundfile.read(path, dtype="float32", always_2d=True)
    return torch.from_numpy(audio.T)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-root", required=True)
    parser.add_argument("--model", default="htdemucs")
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()

    model = get_model(arguments.model).eval()
    scores = {name: [] for name in STEM_NAMES}
    for directory in sorted(glob.glob(os.path.join(arguments.test_root, "*"))):
        mixture = read(os.path.join(directory, "mixture.wav"))
        reference = mixture.mean(0)
        normalised = (mixture - reference.mean()) / reference.std()
        with torch.no_grad():
            output = apply_model(model, normalised[None], device=arguments.device, split=True, overlap=0.25, shifts=0, progress=False)[0]
        output = (output * reference.std() + reference.mean()).cpu()
        by_name = {name: output[index] for index, name in enumerate(model.sources)}
        for name in STEM_NAMES:
            scores[name].append(stem_sdr(by_name[name], read(os.path.join(directory, name + ".wav"))))
    averages = {name: sum(values) / len(values) for name, values in scores.items()}
    averages["average"] = sum(averages.values()) / len(STEM_NAMES)
    print(f"htdemucs_on_musdb_test songs={len(scores['vocals'])} " + " ".join(f"{name}={value:.2f}" for name, value in averages.items()))


if __name__ == "__main__":
    main()
