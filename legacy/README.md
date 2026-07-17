# Legacy / archived

These files are earlier iterations of the plasma aggregation pipeline, kept for
history rather than deleted. The canonical, current pipeline lives at
[`../src/pipeline.py`](../src/pipeline.py). Backlit-photography runs currently
go through `B1.py` in this folder; a separate `pipeline_backlight_variant.py`
that once lived at `../src/` was abandoned (never updated past `VERSION =
"B1.2"` while `B1.py` moved on to later versions) and has been moved to
[`../archive/pipeline_backlight_variant.py.OLD_DO_NOT_RUN`](../archive/pipeline_backlight_variant.py.OLD_DO_NOT_RUN).

- `v1.py` → `v4_2.py` — sequential versions leading up to the current pipeline.
  Each earlier file's in-code changelog comment documents what changed from
  the previous one.
- `v3_alt_tuning.py` — a variant of `v3.py` with different tuned constants
  (image path, RGB thresholds, minimum line length/thickness, minimum area),
  not a further iteration.
- `old_2pic_all_for_one.py` — an older single-file approach. References image
  files (`Test_Before.png`/`Test_After.png`) that no longer exist anywhere in
  this repo, so it is **not runnable as-is**.
- `test_video_processing.py` — a live face-detection + area-vs-time demo,
  unrelated to plasma protein analysis. Likely a personal learning exercise
  that ended up in this folder rather than part of the pipeline.

None of the files in this folder are guaranteed to run against the current
`data/used_plasma_pic/` layout — they may reference the old flat `Plasma/`
path structure.
