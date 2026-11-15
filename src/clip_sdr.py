import json
import os
import sys

import soundfile
import torch

clips_directory, test_root, clips_path, output_path = sys.argv[1:5]
clips = json.load(open(clips_path))
sample_rate = 44100
seconds = 12


def read_window(path, start_seconds):
    audio, _ = soundfile.read(path, dtype="float32", always_2d=True, start=int(start_seconds * sample_rate), frames=seconds * sample_rate)
    return torch.from_numpy(audio.T)


def sdr(estimate, reference):
    noise = (reference - estimate).pow(2).sum()
    return (10 * torch.log10((reference.pow(2).sum() + 1e-8) / (noise + 1e-8))).item()


results = {}
for index, clip in enumerate(clips):
    song, start = clip["song"], clip["start_seconds"]
    song_directory = os.path.join(test_root, song)
    references = {name: read_window(os.path.join(song_directory, name + ".wav"), start) for name in ("vocals", "drums", "bass", "other")}
    references["instrumental"] = references["drums"] + references["bass"] + references["other"]
    for stem_type in ("vocals", "instrumental", "drums"):
        row = {}
        for model in ("candidate", "htdemucs"):
            audio, _ = soundfile.read(os.path.join(clips_directory, f"clip{index:02d}", f"{model}_{stem_type}.flac"), dtype="float32", always_2d=True)
            row[model] = sdr(torch.from_numpy(audio.T), references[stem_type])
        results[f"{song}|{stem_type}"] = row
json.dump(results, open(output_path, "w"), indent=1)
print(len(results), "trials")
