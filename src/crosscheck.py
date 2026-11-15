import sys
import torch
from moises_light import MoisesLight, configs
from stemdata import STEM_NAMES, ValidationExcerpts, split_musdb
from train import separate, stem_sdr

device = torch.device("cuda")
state = torch.load(sys.argv[1], map_location=device)
model = MoisesLight(**configs["fullband_large"]).to(device)
model.load_state_dict(state["model"])
model.eval()
train_directories, validation = split_musdb(sys.argv[2])
if len(sys.argv) > 3:
    validation = train_directories
excerpts = ValidationExcerpts(validation[:6], seconds=20)
matrix = torch.zeros(4, 4)
energy = torch.zeros(4)
reference_energy = torch.zeros(4)
for mixture, stems in excerpts.items:
    estimate = separate(model, mixture, 6 * 44100, device).cpu()
    for estimate_index in range(4):
        energy[estimate_index] += estimate[estimate_index].pow(2).mean()
        reference_energy[estimate_index] += stems[estimate_index].pow(2).mean()
        for reference_index in range(4):
            matrix[estimate_index, reference_index] += stem_sdr(estimate[estimate_index], stems[reference_index])
matrix /= len(excerpts.items)
print("rows=estimate, cols=reference", STEM_NAMES)
for index, name in enumerate(STEM_NAMES):
    print(name, [round(value, 2) for value in matrix[index].tolist()], "estimate_rms/reference_rms=", round((energy[index] / reference_energy[index]).sqrt().item(), 2))
