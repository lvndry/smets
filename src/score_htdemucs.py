import argparse

import torch
from demucs.apply import apply_model
from demucs.pretrained import get_model

from stemdata import STEM_NAMES, ValidationExcerpts, split_musdb


def stem_sdr(estimate, reference):
    noise = (reference - estimate).pow(2).sum()
    signal = reference.pow(2).sum()
    return 10 * torch.log10((signal + 1e-8) / (noise + 1e-8)).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--musdb-root", required=True)
    parser.add_argument("--model", default="htdemucs")
    parser.add_argument("--threads", type=int, default=8)
    arguments = parser.parse_args()

    torch.set_num_threads(arguments.threads)
    model = get_model(arguments.model).eval()
    _, validation = split_musdb(arguments.musdb_root)
    excerpts = ValidationExcerpts(validation)
    scores = {name: [] for name in STEM_NAMES}
    for mixture, stems in excerpts.items:
        reference = mixture.mean(0)
        normalised = (mixture - reference.mean()) / reference.std()
        with torch.no_grad():
            output = apply_model(model, normalised[None], device="cpu", split=True, overlap=0.25, shifts=0, progress=False)[0]
        output = output * reference.std() + reference.mean()
        by_name = {name: output[index] for index, name in enumerate(model.sources)}
        for index, name in enumerate(STEM_NAMES):
            scores[name].append(stem_sdr(by_name[name], stems[index]))
    averages = {name: sum(values) / len(values) for name, values in scores.items()}
    averages["average"] = sum(averages.values()) / len(STEM_NAMES)
    print(f"htdemucs_on_validation_excerpts model={arguments.model} " + " ".join(f"{name}={value:.2f}" for name, value in averages.items()))


if __name__ == "__main__":
    main()
