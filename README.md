# Plasma Protein Aggregation Analysis

An image-processing pipeline that measures plasma protein aggregates from
photographs: a painted reference line of known length calibrates
pixels-to-millimeters, then thresholding and contour detection isolate and
measure each aggregate (area, shape, intensity, and optional volume).

## What it does

Given one or more photos, `src/pipeline.py`:
1. Detects a painted calibration line (red, green, or blue) by color and
   shape, and uses its known real-world length to convert pixels to mm.
2. Flattens uneven lighting across the frame, then uses Otsu's method to
   separate aggregate from background — computed with the calibration
   line's own pixels excluded, so the line itself can't bias the threshold.
3. Finds aggregate contours, filters noise/border artifacts, and measures
   area, perimeter, circularity, intensity relative to background, and
   (if an aggregate thickness is provided) volume.
4. Displays a 2x2 result window per photo (original / calibration overlay /
   annotated result / measurement table), plus a batch comparison table
   with averages when more than one photo is analyzed.

A separate backlit-photography setup was previously maintained at
`src/pipeline_backlight_variant.py`. It never received the bug fixes applied
to `legacy/B1.py` (which is the script actually used for backlit-photo runs
and the one writing `data/plasma_analysis_master_log_backlight.csv`), so it
was archived to
[`archive/pipeline_backlight_variant.py.OLD_DO_NOT_RUN`](archive/pipeline_backlight_variant.py.OLD_DO_NOT_RUN)
to prevent it from being run by mistake.

## Running

```
pip install -r requirements.txt
python src/pipeline.py
```

A popup window asks for the image file(s), reference length, calibration
line color, and (optionally) aggregate thickness. If a GUI can't launch, the
script automatically falls back to the same questions asked in the terminal.

## Structure
- `src/` — the current, canonical pipeline (`pipeline.py`)
- `tutorials/` — four standalone scripts breaking the pipeline's core ideas
  (grayscale conversion, background subtraction, thresholding, contour
  detection) into individually runnable teaching examples
- `data/used_plasma_pic/` — reference/test photographs
- `legacy/` — earlier pipeline iterations, kept for history rather than
  deleted; see [`legacy/README.md`](legacy/README.md) for what's there and why
- `archive/` — abandoned/superseded scripts kept for reference only, renamed
  with an `.OLD_DO_NOT_RUN` suffix so they can't be run by mistake

## Known limitations

- No automated test suite — verification so far has been manual, against
  real photos (see the changelog comments at the top of `src/pipeline.py`
  for the specific cases tested).
- Aggregates in the 0.1–1.0 mm² range can't be reliably auto-separated from
  dust specks by area or shape alone (documented in-code); they're reported
  but flagged rather than silently trusted.
