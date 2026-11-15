import csv
import json
import statistics
import sys

STEM_NAMES = ["vocals", "drums", "bass", "other"]
MODELS = ["htdemucs", "scnet_s_11s_overlap2", "scnet_l_5s_overlap1"]

scores_path, genres_path = sys.argv[1], sys.argv[2]
scores = {}
for row in json.load(open(scores_path)):
    scores.setdefault(row["song"], {})[row["model"]] = row
genres = {row["song"]: row["genre"] for row in csv.DictReader(open(genres_path))}


def song_average(song, model):
    return statistics.mean(scores[song][model][name] for name in STEM_NAMES)


print("genre | songs | htdemucs | scnet-s | scnet-l | scnet-l minus htdemucs | scnet-l wins")
for genre in sorted(set(genres.values())):
    songs = [song for song in scores if genres.get(song) == genre]
    averages = {model: statistics.mean(song_average(song, model) for song in songs) for model in MODELS}
    wins = sum(song_average(song, MODELS[2]) > song_average(song, MODELS[0]) for song in songs)
    print(f"{genre} | {len(songs)} | {averages[MODELS[0]]:.2f} | {averages[MODELS[1]]:.2f} | {averages[MODELS[2]]:.2f} | {averages[MODELS[2]] - averages[MODELS[0]]:+.2f} | {wins}/{len(songs)}")
