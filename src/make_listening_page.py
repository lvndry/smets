import argparse
import base64
import csv
import json
import os
import random

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Stem separation listening test</title>
<style>
  :root { color-scheme: light dark; --bg: #fafafa; --text: #1a1a1a; --card: #fff; --line: #d8d8d8; --accent: #1f6feb; }
  @media (prefers-color-scheme: dark) { :root { --bg: #151515; --text: #eaeaea; --card: #1f1f1f; --line: #3a3a3a; --accent: #6ea8ff; } }
  body { background: var(--bg); color: var(--text); font: 16px/1.5 system-ui, sans-serif; margin: 0 auto; max-width: 760px; padding: 16px; }
  h1 { font-size: 1.4rem; }
  .trial { background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 12px 16px; margin: 16px 0; }
  .trial h2 { font-size: 1rem; margin: 0 0 4px; }
  .hint { color: #777; font-size: 0.9rem; margin: 0 0 8px; }
  .players { display: grid; grid-template-columns: 1fr; gap: 8px; }
  .players label { display: block; font-weight: 600; }
  audio { width: 100%; }
  .choices { margin-top: 8px; display: flex; gap: 16px; flex-wrap: wrap; }
  button { background: var(--accent); color: #fff; border: 0; border-radius: 6px; padding: 8px 14px; font-size: 1rem; cursor: pointer; }
  #result { white-space: pre-wrap; background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 12px 16px; margin-top: 16px; }
</style>
</head>
<body>
<h1>Stem separation listening test</h1>
<p>Each trial plays the original mix, then two separated versions, A and B. Listen with headphones. Pick the one with the cleaner separation: less of the other instruments leaking in, fewer watery or metallic artefacts, and a more natural sound. Choose "no difference" if you cannot tell. Which system is A or B is random. You can stop at any point and press Reveal.</p>
<div id="trials"></div>
<p><button id="reveal">Reveal results</button> <button id="reset" style="background:#777">Clear my answers</button></p>
<pre id="result" hidden></pre>
<script>
const TRIALS = __TRIALS__;
const KEY = JSON.parse(atob("__KEY__"));
const STORAGE = "listening-test-answers";
let answers = {};
try { answers = JSON.parse(localStorage.getItem(STORAGE) || "{}"); } catch (error) { answers = {}; }
function save() { try { localStorage.setItem(STORAGE, JSON.stringify(answers)); } catch (error) {} }
const container = document.getElementById("trials");
TRIALS.forEach((trial, index) => {
  const element = document.createElement("div");
  element.className = "trial";
  element.innerHTML = `<h2>Trial ${index + 1} of ${TRIALS.length}: ${trial.label}</h2><p class="hint">${trial.hint}</p>
    <div class="players">
      <div><label>Original mix</label><audio controls preload="none" src="${trial.mixture}"></audio></div>
      <div><label>A</label><audio controls preload="none" src="${trial.a}"></audio></div>
      <div><label>B</label><audio controls preload="none" src="${trial.b}"></audio></div>
    </div>
    <div class="choices">
      ${["A", "B", "none"].map((choice) => `<label><input type="radio" name="t${index}" value="${choice}"> ${choice === "none" ? "No difference" : choice + " is cleaner"}</label>`).join("")}
    </div>`;
  container.appendChild(element);
  element.querySelectorAll("input").forEach((input) => {
    if (answers[index] === input.value) input.checked = true;
    input.addEventListener("change", () => { answers[index] = input.value; save(); });
  });
});
document.getElementById("reset").addEventListener("click", () => {
  answers = {}; save();
  document.querySelectorAll("input[type=radio]").forEach((input) => { input.checked = false; });
  document.getElementById("result").hidden = true;
});
document.getElementById("reveal").addEventListener("click", () => {
  const tally = {};
  const bump = (group, outcome) => { tally[group] = tally[group] || { candidate: 0, htdemucs: 0, none: 0 }; tally[group][outcome] += 1; };
  let answered = 0;
  const lines = [];
  TRIALS.forEach((trial, index) => {
    const choice = answers[index];
    if (!choice) return;
    answered += 1;
    const outcome = choice === "none" ? "none" : KEY[index][choice];
    bump("all", outcome); bump(trial.type, outcome);
    lines.push(`Trial ${index + 1}: ${trial.song} (${trial.genre}), ${trial.type}: ${outcome === "none" ? "no difference" : outcome + " preferred"}`);
  });
  const summary = Object.entries(tally).map(([group, counts]) => `${group}: candidate ${counts.candidate}, htdemucs ${counts.htdemucs}, no difference ${counts.none}`);
  const box = document.getElementById("result");
  box.hidden = false;
  box.textContent = `Answered ${answered} of ${TRIALS.length}\\n\\n` + summary.join("\\n") + "\\n\\n" + lines.join("\\n");
});
</script>
</body>
</html>
"""

TYPES = {
    "vocals": ("Vocals only", "Listen for instruments bleeding into the vocal, and for watery or clipped consonants."),
    "instrumental": ("Everything except vocals", "Listen for leftover vocal fragments and for holes in the instruments."),
    "drums": ("Drums only", "Listen for bass or guitar bleed, and for smeared hits."),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips-directory", required=True)
    parser.add_argument("--clips", required=True)
    parser.add_argument("--genres", required=True)
    parser.add_argument("--types", default="vocals,instrumental,drums")
    parser.add_argument("--seed", type=int, default=7)
    arguments = parser.parse_args()

    with open(arguments.clips) as clips_file:
        clips = json.load(clips_file)
    genres = {row["song"]: row["genre"] for row in csv.DictReader(open(arguments.genres))}
    generator = random.Random(arguments.seed)
    trials, key = [], []
    for index, clip in enumerate(clips):
        folder = f"clip{index:02d}"
        for type_name in arguments.types.split(","):
            label, hint = TYPES[type_name]
            order = ["candidate", "htdemucs"]
            generator.shuffle(order)
            trials.append({
                "label": f"clip {index + 1}, {label}",
                "hint": hint,
                "type": type_name,
                "song": clip["song"],
                "genre": genres.get(clip["song"], "unknown"),
                "mixture": f"{folder}/mixture.flac",
                "a": f"{folder}/{order[0]}_{type_name}.flac",
                "b": f"{folder}/{order[1]}_{type_name}.flac",
            })
            key.append({"A": order[0], "B": order[1]})
    generator.shuffle(list(range(len(trials))))
    order = list(range(len(trials)))
    generator.shuffle(order)
    trials = [trials[position] for position in order]
    key = [key[position] for position in order]
    page = PAGE.replace("__TRIALS__", json.dumps(trials)).replace("__KEY__", base64.b64encode(json.dumps(key).encode()).decode())
    with open(os.path.join(arguments.clips_directory, "index.html"), "w") as page_file:
        page_file.write(page)
    print(f"wrote {len(trials)} trials")


if __name__ == "__main__":
    main()
