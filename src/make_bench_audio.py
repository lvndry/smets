import glob
import os
import sys

import numpy as np
import soundfile

root, output_directory = sys.argv[1], sys.argv[2]
os.makedirs(output_directory, exist_ok=True)
songs = sorted(glob.glob(os.path.join(root, "test", "*", "mixture.wav")))
first, _ = soundfile.read(songs[0], dtype="float32", always_2d=True)
soundfile.write(os.path.join(output_directory, "clip_60s.wav"), first[: 60 * 44100], 44100)
pieces = []
total = 0
for path in songs:
    audio, _ = soundfile.read(path, dtype="float32", always_2d=True)
    pieces.append(audio)
    total += audio.shape[0]
    if total >= 420 * 44100:
        break
soundfile.write(os.path.join(output_directory, "track_7min.wav"), np.concatenate(pieces)[: 420 * 44100], 44100)
