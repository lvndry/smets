# Phase 0 plan: an open music source separation model

Status: draft for owner review, 2026-11-14. Nothing has been trained. Folder is `stems-model-wip` until the project has a name (see `research/01-names.md`).

Licence readings were done on 2026-11-14 from the official pages. Entries marked UNVERIFIED could not be read (the page blocked automated fetching or no primary text was found). Nothing here is legal advice. Every architecture number is cited; numbers without a citation are labelled as estimates.

## 0. Summary and the hard truth

1. **The binding constraint is data, not architecture.** Every large, real-music stem corpus that the literature trains on is non-commercial or academic-only (MUSDB18, MoisesDB, MedleyDB). The only large stem set with a verified permissive licence is Slakh2100 (CC BY 4.0), and it is synthetic: MIDI rendered with sample libraries.
2. **A model trained only on permissive data will probably not beat htdemucs on real music at first.** htdemucs trained on MUSDB plus 800 internal songs. A permissive-only model has Slakh, synthetic mixes and whatever CC-BY real audio we can verify. No paper I found measures this gap, so any quality prediction for it is UNVERIFIED. Phase 1 must measure it early and cheaply, before large compute is spent.
3. **Decision (owner, 2026-11-14): one model, clean licence.** Euphoria must stay free for all DJs, including those who play paid gigs, so non-commercial weights are out. We build only the permissive model (Apache-2.0 code and weights). The research-only twin is dropped: no run is spent on a model Euphoria cannot ship. A short first run on cleared data (Slakh plus synthetic) tells us whether it can beat htdemucs on held-out per-stem and per-genre metrics. If it cannot, the next lever is more cleared real audio, not a worse model or non-commercial data.
4. **Architecture:** an SCNet-style model of about 10M parameters (single model, all stems in one pass), STFT and iSTFT kept outside the ONNX graph and implemented in the Rust crate. It is the only family with a measured CPU speed advantage over htdemucs and it fits the memory limit by construction.
5. **Stems:** ship drums, bass, vocals, other first. Piano and guitar are trained from Slakh and synthetic mixes and released as experimental channels only if they pass a real-audio check (section 3).
6. **Heads-up on Euphoria memory:** on 2026-11-14 the owner decided to keep htdemucs and do no RoFormer work. This brief replaces that for the new project: nothing here changes Euphoria, and Euphoria switches only through the gates in the brief.

Open questions for the owner are listed at the end (section 8).

## 1. Training data licence audit

### 1.1 Table

| Source | Licence (as read) | Allows training | Allows publishing weights openly | Attribution | Hours | Stems | Genres |
|---|---|---|---|---|---|---|---|
| MUSDB18 / MUSDB18-HQ (Zenodo 1117372, 3338373) | "Other (Non-Commercial)"; provided "for educational purposes only", "not for any commercial purpose"; SigSep page: "academic purposes only". Per-track terms inside: 46 MedleyDB tracks CC BY-NC-SA 4.0, 2 Easton Ellises CC BY-NC-SA 3.0, 2 Native Instruments with no terms, 100 DSD100/Mixing Secrets tracks with rights deferred to the original source | Research and education only | No (NC-SA tracks inside; academic-only terms) | Citation requested | about 10 (150 songs) | vocals, drums, bass, other | Mostly pop, rock, some metal, rap, electronic |
| MoisesDB (github moises-ai/moises-db) | CC BY-NC-SA 4.0 (LICENSE file blocked; read from README and repo metadata) | Non-commercial only | No | Yes | not stated (240 tracks, 45 artists, 12 genres per paper arXiv 2307.15913) | 2-level taxonomy including piano, guitar | 12 genres |
| Slakh2100 (Zenodo 4599666) | CC BY 4.0 | Yes | Yes, with caveats below | Yes (cite Manilow et al., WASPAA 2019) | 145 (mixtures) | piano, guitar, drums, bass, plus others; every mix has at least those four | Pop/rock MIDI rendered by samplers |
| Lakh MIDI | CC BY 4.0 | Yes | Yes | Yes | MIDI only | none | Scraped MIDI, composition copyright unaddressed |
| MedleyDB v1, v2 (Zenodo 1649325, 1715175) | CC BY-NC-SA 4.0, "non-commercial research use only", gated request. The MedleyDB site itself states no licence and says "royalty-free", which conflicts | NC only | No | Yes | not stated (122 + 196 tracks) | Multitrack, many instruments | Mixed |
| DSD100 | UNVERIFIED (no primary terms read) | Unknown | Unknown | Unknown | about 6 (100 songs, from general knowledge, UNVERIFIED) | vocals, drums, bass, other | Mixed |
| Cambridge MT / Mixing Secrets | UNVERIFIED (site returned 403) | Unknown | Unknown, treat as no | Unknown | n/a | Multitracks | Mixed |
| ccMixter | Per-track CC licences ("each submission is marked clearly with the license that applies to it"). Variants include NC and Sampling licences | Per track | Only the CC BY / CC0 subset, after your own filtering | Yes for BY | n/a (no stem collection exists; you build it) | a cappellas, remix stems | Electronic, hip hop, remix culture |
| FMA (github mdeff/fma) | Metadata CC BY 4.0, audio "the license chosen by the artist", "meant for research purposes" | Per track | Only CC BY / CC0 tracks, mixtures only | Yes for BY | about 8,200 (fma_full) | none | 161 genres |
| MTG-Jamendo | "solely for non-commercial research and academic use"; audio per-track CC | NC research only | No | n/a | not stated (55k tracks) | none | Wide |
| MAESTRO | CC BY-NC-SA 4.0 | NC only | No | Yes | 198.7 | solo piano, no stems | Classical piano |
| GuitarSet (Zenodo 3371780) | CC BY 4.0 | Yes | Yes | Yes | not stated (excerpts) | solo acoustic guitar, hexaphonic | Acoustic guitar styles |
| FSD50K / Freesound | Set CC BY; per-clip CC0, BY, BY-NC, Sampling+. About 43k clips CC0 or BY | CC0 and BY clips | Yes for those | Yes for BY | 108.3 | one-shots and sound events, not music stems | Sound events |
| Synthetic mixing from licensed one-shots, loops, MIDI | Depends on the pieces. Verified so far: Lakh MIDI (CC BY), GuitarSet (CC BY), Freesound CC0/BY clips | Yes if every piece is cleared | Yes if every piece is cleared | Per piece | Unlimited | Whatever we construct | Whatever we construct |
| URMP, EGDB, MAPS, ENST-Drums, IDMT, DAMP, Songs of the Open Multitrack | UNVERIFIED (not read) | Unknown | Unknown | Unknown | | | |

### 1.2 Caveats that matter

- **Slakh's chain is not clean.** The CC BY text on the dataset is verified. The sample libraries used for rendering are not named on the pages I read, so whether the rendered audio can be redistributed is UNVERIFIED. The MIDI comes from scraped Lakh files whose compositions have unaddressed copyright. We use the audio as training data only and never redistribute it; the open question is whether weights trained on it inherit anything. Action in Phase 1: read the Slakh paper and repository for the library list.
- **Synthetic-only audio is a poor stand-in for real music.** Slakh is pop and rock MIDI through samplers. Expect a domain gap on live recordings, real vocals and mastered mixes.
- **Share-alike and non-commercial terms on weights are legally unsettled.** Whether trained weights are an "adaptation" under CC BY-SA or BY-NC, and whether training counts as non-commercial, has no settled answer. EU text-and-data-mining exceptions and US fair use are separate arguments I have not researched. Dataset terms like MUSDB's "academic use only" are conditions on whoever accepted them. We therefore do not use any NC, SA or ND source in the permissive model, whatever the legal theory.
- **CC BY real audio (FMA, ccMixter) is a risk, not a clean grant.** Use only per-track CC0 and CC BY, keep a manifest with licence and checksum per track, and decide with the owner before relying on it.
- **Do not distil from htdemucs or any model with unclear weights.** Pseudo-labelling FMA mixtures with htdemucs would carry its unclear weights licence into ours. Self-training from our own earlier checkpoint is fine.

### 1.3 Proposed training sets

**Option A, permissive (Apache-2.0 weights, CC BY attribution list in the model card):**
- Slakh2100 (all stems, piano and guitar included).
- Synthetic mixes: Lakh MIDI rendered through open-licensed soundfonts and synths that we verify one by one, GuitarSet guitar, Freesound CC0 and CC BY drum one-shots and instrument samples, mixed with random gain, EQ, reverb and compression.
- Real-audio extras, if cleared: ccMixter and FMA tracks filtered to CC0 and CC BY, as mixtures for self-training and as stems where ccMixter provides them.
- Accepting that this is smaller and less realistic than anything in the literature.

**Option B, research-only: DROPPED by the owner on 2026-11-14 (kept below only as the record of what was considered).**
(weights under CC BY-NC-SA 4.0, never shipped to Euphoria):
- A plus MUSDB18-HQ train, MoisesDB, MedleyDB. Same architecture, same configs, one flag different.
- This is the data setup in the SCNet paper (MUSDB-HQ + MoisesDB), where adding about 235 songs raised the average from 9.00 to 9.25 for the base model and from 9.69 to 9.92 for the large one (arXiv 2401.13276). It gives the data-vs-licence trade-off in numbers.

**Recommendation: A as the product, B as the measuring stick.** If A lands more than about 1.5 dB below htdemucs per stem on held-out data, the right outcome is to publish that finding and look for more cleared real stems (for example commissioning or recording a CC0 multitrack set) rather than ship a worse model.

Evaluation only, never trained on: MUSDB18-HQ test (the brief says evaluation only) and MoisesDB for per-genre numbers. Both are non-commercial; using them to score a model is within academic evaluation, and we delete the audio after use as the brief requires. Hold out part of any cleared real audio as a licence-clean test set too.

### 1.4 Slakh and synthetic-mix findings (2026-11-14, not legal advice)

- Slakh's paper and Zenodo record name no sample libraries ("professional-grade sample-based virtual instruments", 187 patches). The libraries and their licences are UNVERIFIED, so Slakh's chain stays uncertain until the authors say which libraries they used.
- Lakh MIDI is CC BY 4.0 for the compilation only; the compositions are scraped and many are transcriptions of copyrighted songs. Training on audio rendered from it is lower risk than publishing it, but it is not cleared.
- Cleanest rendering stack found: FluidR3_GM or MuseScore_General (MIT), VCSL (CC0), VSCO 2 CE (CC0, LICENSE file to be read), Salamander Grand Piano (CC BY 3.0), Freepats MuldjordKit (CC BY 4.0). Avoid until checked: Sonatina Symphonic Orchestra (obsolete Sampling Plus licence), Vital presets, TimGM6mb, Surge and Dexed content.
- Consequence: the synthetic set should use MIDI we generate or that is CC0, not Lakh, wherever possible. Slakh stays in only if its authors confirm the libraries or the owner accepts the residual risk.

## 2. Architecture survey

cSDR in dB on MUSDB18-HQ test, V/B/D/O = vocals, bass, drums, other. All from the cited source; the metric differs slightly between papers (a 2026 paper warns that cSDR and museval numbers differ, arXiv 2609.12201 as cited by the survey, UNVERIFIED by me).

| Model | V / B / D / O (avg) | Training data | Params | CPU speed and memory | ONNX | Licence (code / weights) |
|---|---|---|---|---|---|---|
| htdemucs, 800 extra songs | 8.93 / 9.78 / 10.05 / 6.42 (8.80) [1] | MUSDB-HQ + 800 internal songs | about 42M [2] | CPU about 1.5x track duration, 3 GB minimum GPU RAM, 7.8 s segments [4]. Euphoria measures 2.5 GB peak, 15 to 19 s per minute of audio on an M5 | Complex STFT breaks torch export; Mixxx rewrote it as real convolutions, about 18% faster than PyTorch on CPU, under 0.1 dB difference [24] | Code MIT. Weights: README states none; author says unclear |
| htdemucs_ft | 9.20 / 10.39 / 10.08 / 6.32 (9.00) [1] | same | same | about 4x slower [4] | same | same |
| htdemucs_6s | README gives no SDR. MVSep sets: guitar 5.22, piano 2.23 [7] | not stated | not stated | | | same |
| BS-RoFormer, L=6, MUSDB only | 10.66 / 11.31 / 9.49 / 7.73 (9.80) [6] | MUSDB-HQ | 72.2M | No CPU figures. Euphoria saw 7.8 GB for a Mel-RoFormer in ONNX | | lucidrains reimplementation MIT [14]. No weights in the paper |
| BS-RoFormer, L=12 + extra | 12.72 / 13.32 / 12.91 / 9.01 (11.99) [6] | MUSDB-HQ + 450 in-house songs | 93.4M | | | weights not public |
| Mel-RoFormer, L=6 | 11.21 / 9.64 / 9.91 / 7.81 [8] (column labelling in my source is inconsistent, check the PDF before quoting) | MUSDB-HQ | 84.2M | | Mel-RoFormer ONNX exports keep STFT outside the graph [26] | Kim weights tagged MIT on Hugging Face but training data undocumented and no LICENSE in the repo [10][11] |
| SCNet base | 9.89 / 8.82 / 10.51 / 6.76 (9.00) [13] | MUSDB-HQ | 10.08M | CPU RTF 0.669 against 1.38 for HT Demucs, single thread Xeon, about 48% [13] | Convolutions and a small dual-path RNN, no complex ops needed once STFT is outside | Code MIT [15]; weights no licence, Google Drive |
| SCNet-large | 10.86 / 9.49 / 10.98 / 7.44 (9.69) [13] | MUSDB-HQ | 41.2M | not reported | same | same |
| SCNet + MoisesDB | 10.17 / 9.21 / 10.78 / 6.84 (9.25); large 11.10 / 9.86 / 11.23 / 7.51 (9.92) [13] | MUSDB-HQ + MoisesDB | same | | | MoisesDB is NC-SA |
| Moises-Light | 10.92 / 10.04 / 10.93 / 7.95 (9.96) [18] | MUSDB-HQ | about 5M per stem, one model per stem | No speed or memory figures [18] | not read | UNVERIFIED |
| Moises-Light-S | 9.87 / 8.81 / 9.13 / 6.79 (8.65) [18] | MUSDB-HQ | 2.2M per stem | | | UNVERIFIED |
| Band-SCNet, causal | avg 7.79 [19] | not read | 2.59M, 92 ms latency | | | not read |
| Open-Unmix umxhq | 6.25 / 5.07 / 6.04 / 4.28 [17] | MUSDB-HQ | not stated | | | code MIT; umxl weights CC BY-NC-SA |

Sources are the numbered references at the end. No source gave measured CPU peak memory for any model. The RoFormer memory explanation (time attention over about 800 frames and band attention per frame without fused attention in ONNX Runtime CPU) is my reasoning, not a measurement.

### Recommendation

**An SCNet-style model of about 10M parameters, four to six stems from one network.**

- Quality on MUSDB-only data is about 9.0 dB average, close to htdemucs's 8.8 (with 800 extra songs) and well below Mel-RoFormer's reported range. That is the realistic target for a small model, not 12.
- It is the only family with a measured CPU speed advantage over htdemucs [13] and the smallest parameter count with competitive quality, so the memory and time limits are likely to hold. Size would be about 40 MB in float32 and 20 MB in float16, against htdemucs's 84 MB.
- Design choices (my reasoning, to be validated in Phase 1):
  - STFT and iSTFT outside the network, real-valued, static shapes; the Rust crate does them (for example with `realfft`). This avoids the ONNX complex-op problem [25] and makes the graph small.
  - Chunks of 6 to 8 s with 25 to 50% overlap-add (htdemucs uses 7.8 s with a quarter overlap, Moises-Light 9 s with half).
  - STFT 4096, hop about 1024 (SCNet's setting [13]).
  - Few coarse bands with grouped convolutions, with attention or RNN only after strong downsampling.
  - One model, all stems in one pass; Moises-Light's one model per stem would multiply inference cost by the stem count.
- **What it takes to reach Mel-RoFormer quality within Euphoria's limits:** probably not reachable. Mel-RoFormer's gains come with 84M or more parameters and extra data, and Euphoria's own ONNX run needed 7.8 GB. The honest target is parity with or a small gain over htdemucs inside the limits, with a larger research-only variant (SCNet-large size, 41M) as the ceiling we can afford. If the owner still wants a Mel-RoFormer-class model, the path is a distilled or pruned RoFormer, which needs a teacher with a clean licence; none exists today.
- **Fallback if SCNet disappoints on our data:** Moises-Light-style band-split U-Net with a small transformer. Same data pipeline, same runtime.
- **Not recommended:** HTDemucs (unmaintained, large, complex STFT in the graph) and full RoFormers (memory).

## 3. Stems

- **Ship four stems first:** drums, bass, vocals, other. Only these are well defined on real data.
- **Piano and guitar:** the only verified permissive labels are Slakh (synthetic) plus GuitarSet (solo guitar). Real piano labels exist only in NC sets (MAESTRO has solo piano without stems; MoisesDB has piano and guitar tracks under NC-SA). The literature shows how hard they are: htdemucs_6s scores about 5.2 dB guitar and 2.2 dB piano on MVSep sets [7], and BS-RoFormer SW, a much larger model, 9.05 and 7.83 [7].
- **Recommendation:** train with six output channels from the start using masked losses, since real-audio data without piano or guitar labels cannot supervise those channels. Release piano and guitar as experimental channels only if they beat htdemucs_6s on a real-audio check (GuitarSet mixed with other cleared stems, and MoisesDB piano and guitar tracks as an evaluation-only set). Otherwise ship four stems and say piano and guitar are not ready. A weak piano stem that users believe is good is worse than none.

## 4. Compute plan

Training runs on lysk-server's shared GPU, where a drone model also trains. I do not know its GPU model or free memory; that is an open question. The figures below are estimates, not measurements.

| Experiment | Purpose | Estimated cost |
|---|---|---|
| E0 data pipeline smoke test | Manifest, checksums, mixing, loaders | Minutes on CPU, GPU none |
| E1 SCNet base on Slakh only, 11 s crops | Baseline and sanity | About 1 to 2 GPU-days on one 24 GB card (estimate; SCNet's paper does not give training cost in the part I read) |
| E2 A: Slakh + synthetic + cleared real | The permissive model | 2 to 4 GPU-days |
| E3 (dropped) research-only twin | Not run, see section 1.3 | none |
| E4 larger config (41M) of the better of A and B | Ceiling | 4 to 8 GPU-days, only if E2 or E3 shows scaling pays |
| E5 six-stem variant | Piano and guitar | 2 to 4 GPU-days |

Memory per experiment: a 10M parameter model with 11 s stereo 44.1 kHz crops fits in about 10 to 16 GB at batch size 8 to 16 with mixed precision (estimate). Use gradient accumulation if the other job holds more.

Scheduling around the drone job:
- Ask the owner for lysk-server's GPU model, free memory and the drone job's schedule before E1.
- Cap each run's memory (batch size and `PYTORCH_CUDA_ALLOC_CONF`), checkpoint every N steps and make every run resumable, so a run can be paused when drone needs the GPU.
- Run at night or in gaps, one training job at a time; evaluation and data prep are CPU-light and can overlap.
- Data lives on lysk-server, never on the Mac. The Mac only edits code and runs tiny tests; no CPU- or memory-saturating jobs (owner's rule, after the 2026-11-14 out-of-memory reboot).
- Delegate server-side runs to the Jazz agent `lysk-server-vllm` with file-scoped prompts, as the owner prefers.

## 5. Evaluation plan

- **Metrics:** cSDR and uSDR per stem on MUSDB18-HQ test, computed with one fixed implementation and the metric named in every table (the same number can differ between museval and cSDR). MUSDB is used for evaluation only and deleted afterwards.
- **Per genre:** MoisesDB (12 genre tags) and MUSDB where genres are known, plus the owner's tracks for listening only. Required groups: hip hop, rock, jazz, classical with piano, live recordings, electronic dance. Live and classical-with-piano need cleared evaluation audio; MoisesDB and MUSDB have little of either, so Phase 1 must source it (open question).
- **Held out:** a licence-clean test set from cleared real audio, not seen in training, so we do not overfit to MUSDB.
- **Comparisons:** htdemucs, htdemucs_6s, plus Mel-RoFormer (Kim) as the ceiling, all scored with the same script on the same audio.
- **Sum check:** report the residual of the stems against the mix (Euphoria's current figure is about 76 dB on generated stems, limited by 16-bit rounding), and run each separation twice to prove it is deterministic.
- **Listening protocol:** the owner's five library excerpts as before, blind A/B against htdemucs, scoring bleed, artifacts and transients (kick and snare attack, hi-hat smearing, vocal breaths), with notes per stem. Fixed excerpts, order randomised, ratings written down.
- **Costs measured on the target machines:** peak memory and time per minute on Apple Silicon and x86-64 Linux CPU for a 7-minute track, against htdemucs on the same machine.

## 6. Adoption gates and deliverables (unchanged from the brief)

Euphoria adopts only if the model beats htdemucs on held-out per-stem and per-genre metrics and in the owner's listening test, fits memory and speed, has a clear licence, and ships the Rust runtime crate. Later phases produce: model card, ONNX export, `docs/integrating.md`, Rust crate with CI, configs, manifests and one script per experiment.

Interface note for `docs/integrating.md` (from Euphoria's stems module today): stereo, 44.1 kHz, 7.8 s segments with a quarter overlap, triangular blending, four stems in the order drums, bass, other, vocals, one separator kept alive across tracks. Euphoria's code is GPL-3.0-or-later; an Apache-2.0 runtime is compatible with it.

## 7. Risks

- Permissive-only data may cap quality below htdemucs (section 0).
- Slakh's sample-library licences are unread.
- Several data sources are UNVERIFIED (Cambridge MT, DSD100 and the small sets).
- Metric and test-set differences between papers make the survey's cross-model comparison approximate.
- GPU availability is shared.

## 8. Decisions and open questions

Decided 2026-11-14: clean licence only (Euphoria stays free for all DJs); no research-only twin; lysk-server GPU may be used for training, and the owner allows stopping what runs there (I will still look first and ask before stopping the vLLM engines). lysk-server: RTX PRO 6000 Blackwell 96 GB (about 70 GB held by a vLLM engine) and RTX 5090 32 GB (about 24 GB held by another), 32 cores, 188 GB RAM, 250 GB free on `/`.

Left to me by the owner: CC BY real audio from FMA and ccMixter (decide after the first results show whether real audio is needed), commissioning a CC0 multitrack set (not now), the name (shortlist in `01-names.md`).

Original questions, for the record:

1. Is the permissive-first plan, with the research-only twin as a measuring stick, right? Or do you want the research-only model to be the main product?
2. What is lysk-server's GPU model and free memory, and when does the drone job run?
3. Are you willing to rely on CC0 and CC BY tracks from FMA and ccMixter despite the legal uncertainty, or limit the permissive model to Slakh plus synthetic mixes?
4. Can we commission or record a small CC0 multitrack set (and live and piano-with-band evaluation audio)? It is the only clean fix for the data gap.
5. Which name (see `research/01-names.md`)?

## References

[1] https://arxiv.org/html/2211.08553 · [2] Moises-Light Table 4 via https://arxiv.org/html/2510.06785v1 · [3] https://raw.githubusercontent.com/facebookresearch/demucs/main/LICENSE · [4] https://github.com/facebookresearch/demucs · [6] https://arxiv.org/html/2309.02612 · [7] https://mvsep.com/algorithms and https://huggingface.co/jarredou/BS-ROFO-SW-Fixed · [8] https://arxiv.org/html/2310.01809v1 · [10] https://github.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model · [11] https://huggingface.co/KimberleyJSN/melbandroformer · [13] https://arxiv.org/html/2401.13276v1 · [14] https://github.com/lucidrains/BS-RoFormer · [15] https://github.com/starrytong/SCNet · [16] https://github.com/moises-ai/moises-db · [17] https://github.com/sigsep/open-unmix-pytorch · [18] https://arxiv.org/html/2510.06785v1 · [19] https://www.isca-archive.org/interspeech_2025/yang25d_interspeech.html · [24] https://mixxx.org/news/2025-10-27-gsoc2025-demucs-to-onnx-dhunstack/ · [25] https://github.com/pytorch/pytorch/issues/86746 · [26] https://huggingface.co/audiojs/mel-roformer

Dataset sources: Zenodo records 1117372, 3338373 (MUSDB18, HQ), 4599666 (Slakh2100), 1649325 and 1715175 (MedleyDB), 3371780 (GuitarSet); github.com/mdeff/fma; MTG-Jamendo and MAESTRO pages; FSD50K record; ccMixter about page; Lakh MIDI page.
