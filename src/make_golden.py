import argparse
import os
import sys

import onnxruntime
import soundfile
import torch
import yaml

from score_onnx import SCNET_ORDER, separate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zfturbo-directory", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mixture", required=True)
    parser.add_argument("--start-seconds", type=float, default=30.0)
    parser.add_argument("--seconds", type=float, default=25.0)
    parser.add_argument("--output-directory", required=True)
    arguments = parser.parse_args()

    sys.path.insert(0, arguments.zfturbo_directory)
    from models.scnet.scnet import SCNet

    with open(arguments.config) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    network = SCNet(**config["model"]).eval()
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 8
    session = onnxruntime.InferenceSession(arguments.model, options, providers=["CPUExecutionProvider"])

    audio, rate = soundfile.read(arguments.mixture, dtype="float32", always_2d=True, start=int(arguments.start_seconds * 44100), frames=int(arguments.seconds * 44100))
    os.makedirs(arguments.output_directory, exist_ok=True)
    soundfile.write(os.path.join(arguments.output_directory, "excerpt.wav"), audio, rate, subtype="FLOAT")
    mixture = torch.from_numpy(audio.T)
    mono = mixture.mean(0)
    mean, deviation = mono.mean(), mono.std()
    estimate = separate(network, session, (mixture - mean) / deviation, 2) * deviation + mean
    for index, name in enumerate(SCNET_ORDER):
        soundfile.write(os.path.join(arguments.output_directory, f"expected_{name}.wav"), estimate[index].numpy().T, rate, subtype="FLOAT")
    print("golden written", tuple(estimate.shape))


if __name__ == "__main__":
    main()
