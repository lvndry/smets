import argparse
import copy
import json
import os
import sys
import time

import torch
import yaml
from moises_light import MoisesLight, configs

from stemdata import STEM_NAMES, SAMPLE_RATE, MixingDataset, ValidationExcerpts, split_musdb

SCNET_OUTPUT_ORDER = [3, 0, 1, 2]
AUTOCAST = {"enabled": True}


class ScnetAdapter(torch.nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, mixture):
        return self.network(mixture)[:, SCNET_OUTPUT_ORDER]


def build_model(arguments):
    if arguments.preset == "scnet":
        sys.path.insert(0, arguments.zfturbo_directory)
        from models.scnet.scnet import SCNet

        with open(arguments.scnet_config) as config_file:
            config = yaml.load(config_file, Loader=yaml.FullLoader)
        return ScnetAdapter(SCNet(**config["model"]))
    return MoisesLight(**configs[arguments.preset])


def spectral_loss(prediction, target):
    total = 0.0
    for fft_size in (512, 1024, 2048, 4096):
        window = torch.hann_window(fft_size, device=prediction.device)
        flat_prediction = prediction.reshape(-1, prediction.shape[-1]).float()
        flat_target = target.reshape(-1, target.shape[-1]).float()
        predicted_spectrum = torch.stft(flat_prediction, fft_size, fft_size // 4, window=window, return_complex=True).abs()
        target_spectrum = torch.stft(flat_target, fft_size, fft_size // 4, window=window, return_complex=True).abs()
        total = total + (predicted_spectrum - target_spectrum).abs().mean() + (
            torch.log1p(predicted_spectrum * 10) - torch.log1p(target_spectrum * 10)
        ).abs().mean()
    return total / 4


def separate(model, mixture, chunk_frames, device):
    step = chunk_frames // 2
    frames = mixture.shape[-1]
    window = torch.hann_window(chunk_frames, periodic=False, device=device).clamp_min(1e-3)
    output = torch.zeros((len(STEM_NAMES), 2, frames), device=device)
    weight = torch.zeros(frames, device=device)
    starts = list(range(0, max(frames - chunk_frames, 0) + 1, step))
    if starts[-1] + chunk_frames < frames:
        starts.append(frames - chunk_frames)
    for start in starts:
        chunk = mixture[:, start:start + chunk_frames]
        if chunk.shape[-1] < chunk_frames:
            chunk = torch.nn.functional.pad(chunk, (0, chunk_frames - chunk.shape[-1]))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=AUTOCAST["enabled"]):
            prediction = model(chunk.unsqueeze(0).to(device))[0].float()
        length = min(chunk_frames, frames - start)
        output[:, :, start:start + length] += prediction[:, :, :length] * window[:length]
        weight[start:start + length] += window[:length]
    return output / weight.clamp_min(1e-6)


def stem_sdr(estimate, reference):
    noise = (reference - estimate).pow(2).sum()
    signal = reference.pow(2).sum()
    return 10 * torch.log10((signal + 1e-8) / (noise + 1e-8)).item()


def validate(model, excerpts, chunk_frames, device):
    model.eval()
    scores = {stem_name: [] for stem_name in STEM_NAMES}
    for mixture, stems in excerpts.items:
        estimate = separate(model, mixture, chunk_frames, device).cpu()
        for index, stem_name in enumerate(STEM_NAMES):
            scores[stem_name].append(stem_sdr(estimate[index], stems[index]))
    model.train()
    averages = {stem_name: sum(values) / len(values) for stem_name, values in scores.items()}
    averages["average"] = sum(averages.values()) / len(STEM_NAMES)
    return averages


def update_ema(ema_model, model, momentum):
    with torch.no_grad():
        for ema_parameter, parameter in zip(ema_model.parameters(), model.parameters()):
            ema_parameter.mul_(momentum).add_(parameter.detach(), alpha=1 - momentum)
        for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
            ema_buffer.copy_(buffer)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--preset", default="fullband_large")
    parser.add_argument("--musdb-root", required=True)
    parser.add_argument("--slakh-root", default=None)
    parser.add_argument("--runs-directory", required=True)
    parser.add_argument("--pseudo-root", default=None)
    parser.add_argument("--pseudo-share", type=float, default=0.5)
    parser.add_argument("--slakh-share", type=float, default=0.5)
    parser.add_argument("--init-from", default=None)
    parser.add_argument("--loss", choices=["l1_spectral", "mse"], default="l1_spectral")
    parser.add_argument("--scnet-config", default=None)
    parser.add_argument("--zfturbo-directory", default=None)
    parser.add_argument("--crop-seconds", type=float, default=6.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--total-steps", type=int, default=200000)
    parser.add_argument("--validate-every", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()

    torch.manual_seed(arguments.seed)
    device = torch.device("cuda")
    if arguments.preset == "scnet":
        AUTOCAST["enabled"] = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    run_directory = os.path.join(arguments.runs_directory, arguments.run_name)
    os.makedirs(run_directory, exist_ok=True)
    with open(os.path.join(run_directory, "arguments.json"), "w") as arguments_file:
        json.dump(vars(arguments), arguments_file, indent=2)

    musdb_train, musdb_validation = split_musdb(arguments.musdb_root)
    slakh_directories = []
    if arguments.slakh_root:
        import glob
        slakh_directories = sorted(glob.glob(os.path.join(arguments.slakh_root, "train", "*")))
    crop_frames = int(arguments.crop_seconds * SAMPLE_RATE)
    pseudo_directories = []
    if arguments.pseudo_root:
        import glob
        pseudo_directories = [
            directory for directory in sorted(glob.glob(os.path.join(arguments.pseudo_root, "*")))
            if all(glob.glob(os.path.join(directory, stem_name + ".flac")) for stem_name in STEM_NAMES)
            and time.time() - max(os.path.getmtime(path) for path in glob.glob(os.path.join(directory, "*.flac"))) > 300
        ]
        print(f"pseudo_tracks={len(pseudo_directories)}", flush=True)
    dataset = MixingDataset(musdb_train, slakh_directories, crop_frames, arguments.total_steps * arguments.batch_size, arguments.seed, slakh_share=arguments.slakh_share, pseudo_directories=pseudo_directories, pseudo_share=arguments.pseudo_share)
    loader = torch.utils.data.DataLoader(dataset, batch_size=arguments.batch_size, num_workers=arguments.workers, shuffle=False, drop_last=True, persistent_workers=True, prefetch_factor=4)
    excerpts = ValidationExcerpts(musdb_validation)

    model = build_model(arguments).to(device)
    ema_model = copy.deepcopy(model).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=arguments.learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=arguments.learning_rate, total_steps=arguments.total_steps, pct_start=0.02)
    step = 0
    if arguments.init_from:
        initial_state = torch.load(arguments.init_from, map_location=device)
        if "ema" in initial_state:
            model.load_state_dict(initial_state["ema"])
        elif "model" in initial_state:
            model.load_state_dict(initial_state["model"])
        else:
            model.network.load_state_dict(initial_state)
        ema_model.load_state_dict(model.state_dict())
    checkpoint_path = os.path.join(run_directory, "last.pt")
    if os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state["model"])
        ema_model.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        step = state["step"]
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"preset={arguments.preset} parameters={parameter_count} start_step={step}", flush=True)

    model.train()
    best_average = -1e9
    started = time.time()
    data_iterator = iter(loader)
    while step < arguments.total_steps:
        mixture, stems = next(data_iterator)
        mixture = mixture.to(device, non_blocking=True)
        stems = stems.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=AUTOCAST["enabled"]):
            prediction = model(mixture)
        prediction = prediction.float()
        if arguments.loss == "mse":
            loss = (prediction - stems).pow(2).mean() * 100
        else:
            loss = (prediction - stems).abs().mean() + 0.5 * spectral_loss(prediction, stems)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        update_ema(ema_model, model, 0.999)
        step += 1
        if step % 100 == 0:
            print(f"step={step} loss={loss.item():.4f} lr={scheduler.get_last_lr()[0]:.2e} elapsed={time.time() - started:.0f}s", flush=True)
        if step % arguments.validate_every == 0 or step == arguments.total_steps:
            raw_scores = validate(model, excerpts, crop_frames, device)
            print(f"validation_raw step={step} " + " ".join(f"{name}={value:.2f}" for name, value in raw_scores.items()), flush=True)
            scores = validate(ema_model, excerpts, crop_frames, device)
            print(f"validation step={step} " + " ".join(f"{name}={value:.2f}" for name, value in scores.items()), flush=True)
            state = {"model": model.state_dict(), "ema": ema_model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "step": step}
            torch.save(state, checkpoint_path)
            if arguments.preset == "scnet":
                torch.save(ema_model.network.state_dict(), os.path.join(run_directory, "ema_network.pt"))
            if scores["average"] > best_average:
                best_average = scores["average"]
                torch.save({"ema": ema_model.state_dict(), "step": step, "scores": scores, "preset": arguments.preset}, os.path.join(run_directory, "best.pt"))


if __name__ == "__main__":
    main()
