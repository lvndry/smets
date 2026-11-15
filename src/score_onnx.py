import argparse
import glob
import os
import sys

import numpy as np
import onnxruntime
import soundfile
import torch
import yaml

CHUNK_SAMPLES = 485100
STEM_NAMES = ["vocals", "drums", "bass", "other"]
SCNET_ORDER = ["drums", "bass", "other", "vocals"]


def stem_sdr(estimate, reference):
    noise = (reference - estimate).pow(2).sum()
    signal = reference.pow(2).sum()
    return 10 * torch.log10((signal + 1e-8) / (noise + 1e-8)).item()


def separate(network, session, mixture, overlap):
    step = CHUNK_SAMPLES // overlap
    frames = mixture.shape[-1]
    window = torch.hann_window(CHUNK_SAMPLES, periodic=False).clamp_min(1e-3)
    output = torch.zeros((4, 2, frames))
    weight = torch.zeros(frames)
    starts = list(range(0, max(frames - CHUNK_SAMPLES, 0) + 1, step))
    if starts[-1] + CHUNK_SAMPLES < frames:
        starts.append(frames - CHUNK_SAMPLES)
    from export_scnet_onnx import from_spectrum, to_spectrum

    for start in starts:
        chunk = mixture[:, start : start + CHUNK_SAMPLES]
        if chunk.shape[-1] < CHUNK_SAMPLES:
            chunk = torch.nn.functional.pad(chunk, (0, CHUNK_SAMPLES - chunk.shape[-1]))
        spectrum, padding = to_spectrum(network, chunk[None])
        features = session.run(None, {"spectrum": spectrum.numpy()})[0]
        waveform = from_spectrum(network, torch.from_numpy(features), padding)[0]
        length = min(CHUNK_SAMPLES, frames - start)
        output[:, :, start : start + length] += waveform[:, :, :length] * window[:length]
        weight[start : start + length] += window[:length]
    return output / weight.clamp_min(1e-6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zfturbo-directory", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--test-root", required=True)
    parser.add_argument("--overlap", type=int, default=2)
    parser.add_argument("--threads", type=int, default=8)
    arguments = parser.parse_args()

    sys.path.insert(0, arguments.zfturbo_directory)
    from models.scnet.scnet import SCNet

    with open(arguments.config) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    network = SCNet(**config["model"]).eval()
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = arguments.threads
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    session = onnxruntime.InferenceSession(arguments.model, options, providers=["CPUExecutionProvider"])
    torch.set_num_threads(arguments.threads)

    scores = {name: [] for name in STEM_NAMES}
    for directory in sorted(glob.glob(os.path.join(arguments.test_root, "*"))):
        audio, _ = soundfile.read(os.path.join(directory, "mixture.wav"), dtype="float32", always_2d=True)
        mixture = torch.from_numpy(audio.T)
        mono = mixture.mean(0)
        mean, deviation = mono.mean(), mono.std()
        estimate = separate(network, session, (mixture - mean) / deviation, arguments.overlap) * deviation + mean
        by_name = {name: estimate[index] for index, name in enumerate(SCNET_ORDER)}
        for name in STEM_NAMES:
            reference, _ = soundfile.read(os.path.join(directory, name + ".wav"), dtype="float32", always_2d=True)
            scores[name].append(stem_sdr(by_name[name], torch.from_numpy(reference.T)))
        print(os.path.basename(directory), " ".join(f"{name}={scores[name][-1]:.2f}" for name in STEM_NAMES), flush=True)
    averages = {name: float(np.mean(values)) for name, values in scores.items()}
    averages["average"] = float(np.mean(list(averages.values())))
    print("onnx_pipeline_on_musdb_test " + " ".join(f"{name}={value:.2f}" for name, value in averages.items()))


if __name__ == "__main__":
    main()
