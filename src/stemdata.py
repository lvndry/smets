import glob
import os
import random

import numpy as np
import soundfile
import torch
import yaml

STEM_NAMES = ["vocals", "drums", "bass", "other"]
SAMPLE_RATE = 44100


def read_crop(path, start, frames):
    audio, _ = soundfile.read(path, start=start, frames=frames, dtype="float32", always_2d=True)
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    if audio.shape[0] < frames:
        audio = np.pad(audio, ((0, frames - audio.shape[0]), (0, 0)))
    return audio.T


class MusdbTrack:
    def __init__(self, directory):
        self.directory = directory
        self.frames = soundfile.info(os.path.join(directory, "mixture.wav")).frames

    def stem_paths(self, stem_name):
        return [os.path.join(self.directory, stem_name + ".wav")]


class SlakhTrack:
    def __init__(self, directory):
        self.directory = directory
        with open(os.path.join(directory, "metadata.yaml")) as metadata_file:
            metadata = yaml.safe_load(metadata_file)
        self.paths_by_stem = {stem_name: [] for stem_name in STEM_NAMES}
        for stem_id, stem_info in metadata["stems"].items():
            if not stem_info.get("audio_rendered", False):
                continue
            path = os.path.join(directory, "stems", stem_id + ".flac")
            if not os.path.exists(path):
                continue
            if stem_info["is_drum"]:
                self.paths_by_stem["drums"].append(path)
            elif stem_info["inst_class"] == "Bass":
                self.paths_by_stem["bass"].append(path)
            else:
                self.paths_by_stem["other"].append(path)
        reference = os.path.join(directory, "mix.flac")
        self.frames = soundfile.info(reference).frames

    def stem_paths(self, stem_name):
        return self.paths_by_stem[stem_name]


class PseudoTrack:
    def __init__(self, directory):
        self.directory = directory
        self.paths_by_stem = {}
        for stem_name in STEM_NAMES:
            matches = glob.glob(os.path.join(directory, stem_name + ".*"))
            self.paths_by_stem[stem_name] = matches[:1]
        self.frames = soundfile.info(self.paths_by_stem["vocals"][0]).frames

    def stem_paths(self, stem_name):
        return self.paths_by_stem[stem_name]


def load_stem(track, stem_name, start, frames):
    paths = track.stem_paths(stem_name)
    if not paths:
        return np.zeros((2, frames), dtype=np.float32)
    total = np.zeros((2, frames), dtype=np.float32)
    for path in paths:
        total += read_crop(path, start, frames)
    return total


def split_musdb(root, validation_count=14):
    directories = sorted(glob.glob(os.path.join(root, "train", "*")))
    validation_indices = set(np.linspace(0, len(directories) - 1, validation_count).astype(int).tolist())
    train = [directory for index, directory in enumerate(directories) if index not in validation_indices]
    validation = [directory for index, directory in enumerate(directories) if index in validation_indices]
    return train, validation


class MixingDataset(torch.utils.data.Dataset):
    def __init__(self, musdb_directories, slakh_directories, crop_frames, samples_per_epoch, seed, slakh_share=0.5, shuffle_share=0.5, pseudo_directories=(), pseudo_share=0.0):
        self.musdb_tracks = [MusdbTrack(directory) for directory in musdb_directories]
        self.slakh_tracks = [SlakhTrack(directory) for directory in slakh_directories]
        self.pseudo_tracks = [PseudoTrack(directory) for directory in pseudo_directories]
        self.crop_frames = crop_frames
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed
        self.slakh_share = slakh_share if self.slakh_tracks else 0.0
        self.pseudo_share = pseudo_share if self.pseudo_tracks else 0.0
        self.shuffle_share = shuffle_share

    def __len__(self):
        return self.samples_per_epoch

    def draw_track(self, generator):
        draw = generator.random()
        if draw < self.pseudo_share:
            return generator.choice(self.pseudo_tracks)
        if draw < self.pseudo_share + self.slakh_share:
            return generator.choice(self.slakh_tracks)
        return generator.choice(self.musdb_tracks)

    def draw_stem(self, track, stem_name, generator):
        max_start = max(track.frames - self.crop_frames, 0)
        start = generator.randint(0, max_start)
        audio = load_stem(track, stem_name, start, self.crop_frames)
        audio = audio * generator.uniform(0.5, 1.5)
        if generator.random() < 0.5:
            audio = audio[::-1]
        if generator.random() < 0.5:
            audio = -audio
        return np.ascontiguousarray(audio)

    def __getitem__(self, index):
        worker = torch.utils.data.get_worker_info()
        worker_id = worker.id if worker else 0
        generator = random.Random(self.seed * 1000003 + index * 31 + worker_id)
        base_track = self.draw_track(generator)
        stems = []
        for stem_name in STEM_NAMES:
            track = base_track
            if generator.random() < self.shuffle_share:
                track = self.draw_track(generator)
            stems.append(self.draw_stem(track, stem_name, generator))
        stems = np.stack(stems)
        mixture = stems.sum(axis=0)
        return torch.from_numpy(mixture), torch.from_numpy(stems)


class ValidationExcerpts:
    def __init__(self, directories, seconds=30):
        self.items = []
        for directory in directories:
            track = MusdbTrack(directory)
            frames = seconds * SAMPLE_RATE
            start = max((track.frames - frames) // 2, 0)
            stems = np.stack([read_crop(os.path.join(directory, stem_name + ".wav"), start, frames) for stem_name in STEM_NAMES])
            self.items.append((torch.from_numpy(stems.sum(axis=0)), torch.from_numpy(stems)))
