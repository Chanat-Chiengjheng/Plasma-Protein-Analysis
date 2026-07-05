# Introduction to this project

## Why

Measuring plasma protein aggregation from a photograph sounds simple until
you actually try it: lighting is never perfectly even, the calibration
marker used to convert pixels to millimeters can itself get mistaken for the
thing being measured, and a small real aggregate can look identical in size
to a speck of dust. This project exists to turn "eyeball the photo and
estimate" into a reproducible, quantitative measurement — and to be honest,
in the code and in the documentation, about exactly where that measurement
still has limits.

## What

Given a photo with a painted reference line of known length, the pipeline
calibrates pixel-to-millimeter scale, isolates protein aggregates from the
background, and reports area, shape (circularity), relative intensity, and
optionally volume for each one found — across a single photo or a batch,
with a comparison table when analyzing more than one.

## How

The pipeline went through several real iterations, each driven by a specific
failure seen on an actual photo rather than a hypothetical edge case: a
fixed brightness threshold that had to be manually retuned nearly every run
was replaced with Otsu's automatic method; a fixed-pixel calibration-line
filter that silently broke when camera zoom changed was replaced with a
scale-relative shape check; and when a real batch photo revealed the
calibration line's own pixels were skewing the threshold calculation, the
line was explicitly excluded from that calculation rather than just from the
final result. Every one of these changes is documented in the code itself —
what broke, what was tried, what was tested against, and what the tradeoffs
were — because "it works on my test photo" isn't the same claim as "it works
and here's the evidence."

## Outcome

The current pipeline (`src/pipeline.py`) reliably measures aggregates across
varying lighting conditions and photo batches, with known, explicitly
documented limitations rather than hidden edge cases — for example, very
small aggregates in a narrow size range that can't be automatically
distinguished from dust are reported and flagged for manual review instead
of being silently guessed at. The project's iteration history is kept in
`legacy/` rather than discarded, since the record of what didn't work (and
why) is as much a part of the result as the final version.
