import glob
import os
import sys
import torch
from moises_light import MoisesLight, configs
from stemdata import STEM_NAMES, MixingDataset, split_musdb

device = torch.device("cuda")
state = torch.load(sys.argv[1], map_location=device)
model = MoisesLight(**configs["fullband_large"]).to(device)
model.load_state_dict(state["model"])
train_directories, _ = split_musdb(sys.argv[2])
slakh_directories = sorted(glob.glob(os.path.join(sys.argv[3], "train", "*")))
for label, slakh_share in (("musdb_only", 0.0), ("slakh_only", 1.0)):
    dataset = MixingDataset(train_directories, slakh_directories, 6 * 44100, 24, 123, slakh_share=slakh_share)
    batches = [dataset[index] for index in range(24)]
    mixture = torch.stack([item[0] for item in batches]).to(device)
    stems = torch.stack([item[1] for item in batches]).to(device)
    for mode in ("eval", "train_batchstats"):
        model.eval() if mode == "eval" else model.train()
        scores = [[] for _ in STEM_NAMES]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for start in range(0, 24, 4):
                prediction = model(mixture[start:start + 4]).float()
                for sample in range(prediction.shape[0]):
                    for index in range(4):
                        reference = stems[start + sample, index]
                        noise = (reference - prediction[sample, index]).pow(2).sum()
                        scores[index].append((10 * torch.log10((reference.pow(2).sum() + 1e-8) / (noise + 1e-8))).item())
        print(label, mode, " ".join(f"{name}={sum(values) / len(values):.2f}" for name, values in zip(STEM_NAMES, scores)), flush=True)
