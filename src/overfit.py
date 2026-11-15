import glob
import os
import sys
import numpy as np
import torch
from moises_light import MoisesLight, configs
from stemdata import STEM_NAMES, SAMPLE_RATE, read_crop
from train import spectral_loss

device = torch.device("cuda")
torch.manual_seed(0)
song_directories = sorted(glob.glob(os.path.join(sys.argv[1], "train", "*")))[:2]
crop_frames = 6 * SAMPLE_RATE
stems = []
for directory in song_directories:
    stems.append(np.stack([read_crop(os.path.join(directory, name + ".wav"), 60 * SAMPLE_RATE, crop_frames) for name in STEM_NAMES]))
stems = torch.from_numpy(np.stack(stems)).to(device)
mixture = stems.sum(dim=1)
model = MoisesLight(**configs[sys.argv[2]]).to(device).train()
optimizer = torch.optim.AdamW(model.parameters(), lr=float(sys.argv[3]))
for step in range(1, 601):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        prediction = model(mixture)
    prediction = prediction.float()
    loss = (prediction - stems).abs().mean() + 0.5 * spectral_loss(prediction, stems)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if step % 100 == 0:
        scores = []
        for index in range(4):
            reference = stems[:, index]
            noise = (reference - prediction[:, index]).pow(2).sum()
            scores.append((10 * torch.log10(reference.pow(2).sum() / noise)).item())
        print(f"step={step} loss={loss.item():.4f} " + " ".join(f"{name}={score:.1f}" for name, score in zip(STEM_NAMES, scores)), flush=True)
