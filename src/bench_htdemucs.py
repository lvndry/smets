import argparse
import time

import soundfile
import torch
from demucs.apply import apply_model
from demucs.pretrained import get_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--model", default="htdemucs")
    arguments = parser.parse_args()

    torch.set_num_threads(arguments.threads)
    model = get_model(arguments.model).eval()
    audio, sample_rate = soundfile.read(arguments.audio, dtype="float32", always_2d=True)
    mixture = torch.from_numpy(audio.T)
    seconds = mixture.shape[-1] / sample_rate
    reference = mixture.mean(0)
    normalised = (mixture - reference.mean()) / reference.std()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    started = time.time()
    with torch.no_grad():
        apply_model(model, normalised[None], device="cpu", split=True, overlap=0.25, shifts=0, progress=False)
    elapsed = time.time() - started
    print(f"model={arguments.model} parameters={parameter_count} audio_seconds={seconds:.1f} threads={arguments.threads} elapsed={elapsed:.1f}s seconds_per_minute={elapsed / seconds * 60:.1f}")


if __name__ == "__main__":
    main()
