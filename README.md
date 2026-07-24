# Plasma Protein Aggregation Analysis

An image-processing and data-analysis toolkit for plasma protein aggregate
experiments: measuring aggregates from calibrated photographs, and comparing
oscilloscope waveform recordings from the same experiments.

## Start here: `Analysis Tools/`

This is the folder to use. It contains the two ready-to-run programs — no
digging through the rest of the repo required.

- **`Analysis Tools/Image_Analysis.py`** — measures plasma protein aggregates
  from photographs. Given a photo with a painted reference line of known
  length, it calibrates pixels-to-millimeters, isolates aggregates from the
  background, and reports each one's area, shape, and intensity (and volume,
  if you provide an aggregate thickness) — for a single photo or a batch,
  with a comparison table across the batch. Supports both reflected-light
  and backlit photography (you pick the mode when you run it).

- **`Analysis Tools/CSV_Analysis.py`** — compares oscilloscope CSV
  recordings ("wave compare"). Given one or more scope CSV exports, it offers
  five analysis modes to choose from:
  - **Task A** — compare cycles at fixed 1ms intervals within a recording
  - **Task B** — compare effective (RMS) voltage/current across recordings
  - **Task C** — per-cycle charge calculation with statistics
  - **Task D** — charge histogram (frequency distribution)
  - **Task G** — classify cycle behavior by peak pattern

### Running them

```
pip install -r requirements.txt
python "Analysis Tools/Image_Analysis.py"
python "Analysis Tools/CSV_Analysis.py"
```

Each script opens a popup window asking for the file(s) and any settings it
needs (e.g. reference length and calibration-line color for the image tool;
which analysis task for the CSV tool). If a popup can't launch on your
machine, the script automatically falls back to asking the same questions in
the terminal.

### Getting updates

These two scripts may be updated over time (bug fixes, new features). To get
the latest version, re-download (or `git pull`) this repository and use the
files in `Analysis Tools/` again — the file names stay the same, so nothing
else about how you run them changes.

## Everything else in this repo

The rest of the repository is the development history and supporting
material behind these two tools — useful if you want to understand how they
were built, but not needed just to run them:

- `data/used_plasma_pic/` — reference/test photographs
- `data/` — CSV logs of past analysis runs
- `tutorials/` — four standalone scripts breaking the image pipeline's core
  ideas (grayscale conversion, background subtraction, thresholding, contour
  detection) into individually runnable teaching examples
- `legacy/` — earlier iterations of the image pipeline, kept for history
  rather than deleted; see [`legacy/README.md`](legacy/README.md) for what's
  there and why
- `archive/` — abandoned/superseded scripts kept for reference only, renamed
  with an `.OLD_DO_NOT_RUN` suffix so they can't be run by mistake

## Known limitations

- No automated test suite — verification so far has been manual, against
  real photos and recordings (see the changelog comments at the top of
  `legacy/V4_hyesteresis.py` for the specific cases tested on the image
  pipeline's core logic).
- Aggregates in the 0.1–1.0 mm² range can't be reliably auto-separated from
  dust specks by area or shape alone (documented in-code); they're reported
  but flagged rather than silently trusted.
