import sys
import torch
from moises_light import MoisesLight, configs
from stemdata import STEM_NAMES, ValidationExcerpts, split_musdb
from train import separate, stem_sdr

device = torch.device("cuda")
state = torch.load(sys.argv[1], map_location=device)
_, validation = split_musdb(sys.argv[2])
excerpts = ValidationExcerpts(validation[:4], seconds=20)
chunk_frames = 6 * 44100

def score(model, label):
    model.eval()
    totals = [[] for _ in STEM_NAMES]
    for mixture, stems in excerpts.items:
        estimate = separate(model, mixture, chunk_frames, device).cpu()
        for index in range(len(STEM_NAMES)):
            totals[index].append(stem_sdr(estimate[index], stems[index]))
    print(label, " ".join(f"{name}={sum(values)/len(values):.2f}" for name, values in zip(STEM_NAMES, totals)), flush=True)

model = MoisesLight(**configs["fullband_large"]).to(device)
buffer_names = [name for name, _ in model.named_buffers()]
print("buffers:", len(buffer_names), buffer_names[:6])
model.load_state_dict(state["model"])
score(model, "raw_weights")
model.load_state_dict(state["ema"])
score(model, "ema_as_saved")
raw = MoisesLight(**configs["fullband_large"]).to(device)
raw.load_state_dict(state["model"])
ema_buffers = dict(model.named_buffers())
for name, buffer in raw.named_buffers():
    ema_buffers[name].copy_(buffer)
score(model, "ema_with_raw_buffers")
module_types = sorted({type(module).__name__ for module in raw.modules()})
print(module_types)
