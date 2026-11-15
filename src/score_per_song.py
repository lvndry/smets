import argparse
import glob
import json
import os
import sys

import soundfile
import torch
import yaml

STEM_NAMES = ["vocals", "drums", "bass", "other"]
SCNET_ORDER = ["drums", "bass", "other", "vocals"]


def stem_sdr(estimate, reference):
    noise = (reference - estimate).pow(2).sum()
    signal = reference.pow(2).sum()
    return 10 * torch.log10((signal + 1e-8) / (noise + 1e-8)).item()


def read(path):
    audio, _ = soundfile.read(path, dtype="float32", always_2d=True)
    return torch.from_numpy(audio.T)


def separate_scnet(network, mixture, chunk_samples, overlap, device, step_samples=None):
    mono = mixture.mean(0)
    mean, deviation = mono.mean(), mono.std()
    normalised = (mixture - mean) / deviation
    frames = normalised.shape[-1]
    step = step_samples or chunk_samples // overlap
    window = torch.hann_window(chunk_samples, periodic=False).clamp_min(1e-3).to(device)
    output = torch.zeros((4, 2, frames), device=device)
    weight = torch.zeros(frames, device=device)
    starts = list(range(0, max(frames - chunk_samples, 0) + 1, step))
    if starts[-1] + chunk_samples < frames:
        starts.append(frames - chunk_samples)
    for start in starts:
        chunk = normalised[:, start : start + chunk_samples]
        if chunk.shape[-1] < chunk_samples:
            chunk = torch.nn.functional.pad(chunk, (0, chunk_samples - chunk.shape[-1]))
        with torch.no_grad():
            prediction = network(chunk[None].to(device))[0]
        length = min(chunk_samples, frames - start)
        output[:, :, start : start + length] += prediction[:, :, :length] * window[:length]
        weight[start : start + length] += window[:length]
    return ((output / weight.clamp_min(1e-6)).cpu() * deviation + mean)


def separate_htdemucs(model, mixture, device):
    from demucs.apply import apply_model

    reference = mixture.mean(0)
    normalised = (mixture - reference.mean()) / reference.std()
    with torch.no_grad():
        output = apply_model(model, normalised[None], device=device, split=True, overlap=0.25, shifts=0, progress=False)[0]
    return (output * reference.std() + reference.mean()).cpu()


def load_scnet(zfturbo_directory, config_path, checkpoint_path, device):
    sys.path.insert(0, zfturbo_directory)
    from models.scnet.scnet import SCNet

    with open(config_path) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    network = SCNet(**config["model"])
    network.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    return network.eval().to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zfturbo-directory", required=True)
    parser.add_argument("--test-root", required=True)
    parser.add_argument("--scnet-s-config", required=True)
    parser.add_argument("--scnet-s-checkpoint", required=True)
    parser.add_argument("--scnet-l-config", required=True)
    parser.add_argument("--scnet-l-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()

    device = "cuda"
    from demucs.pretrained import get_model

    htdemucs = get_model("htdemucs").eval().to(device)
    scnet_s = load_scnet(arguments.zfturbo_directory, arguments.scnet_s_config, arguments.scnet_s_checkpoint, device)
    scnet_l = load_scnet(arguments.zfturbo_directory, arguments.scnet_l_config, arguments.scnet_l_checkpoint, device)

    results = []
    for directory in sorted(glob.glob(os.path.join(arguments.test_root, "*"))):
        song = os.path.basename(directory)
        mixture = read(os.path.join(directory, "mixture.wav"))
        references = {name: read(os.path.join(directory, name + ".wav")) for name in STEM_NAMES}
        estimates = {
            "htdemucs": {name: stem for name, stem in zip(htdemucs.sources, separate_htdemucs(htdemucs, mixture, device))},
            "scnet_s_11s_overlap2": {name: stem for name, stem in zip(SCNET_ORDER, separate_scnet(scnet_s, mixture, 485100, 2, device))},
            "scnet_l_5s_overlap1": {name: stem for name, stem in zip(SCNET_ORDER, separate_scnet(scnet_l, mixture, 242550, 1, device))},
        }
        for model_name, stems in estimates.items():
            row = {"song": song, "model": model_name}
            for name in STEM_NAMES:
                row[name] = stem_sdr(stems[name], references[name])
            results.append(row)
        print(song, " ".join(f"{row['model']}={sum(row[name] for name in STEM_NAMES) / 4:.2f}" for row in results[-3:]), flush=True)
        with open(arguments.output, "w") as output_file:
            json.dump(results, output_file, indent=1)


if __name__ == "__main__":
    main()
