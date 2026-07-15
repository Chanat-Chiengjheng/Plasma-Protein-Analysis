import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import csv
from datetime import datetime

VERSION = "B1.10"
BUILD_TAG = "lower-area-floor-2"

# =============================================
# CHANGELOG (quick reference)
# =============================================
# B1.0 - first version. Separate tool from the reflected-light V4.x line -
#        same overall architecture, adapted for backlight physics.
#        NOT YET TESTED against a real backlit photo with an actual
#        aggregate - only validated against a synthetic test image so far.
#        Every constant below is a reasoned starting point inherited from
#        reflected-light testing, not independently confirmed here.
#
# B1.1 - first real backlit photos tested (9 photos, same calibration line +
#        marker as the reflected-light side). Three real, confirmed,
#        generalizable fixes - all expected to matter regardless of future
#        lighting quality:
#
#        1. PORTED the Otsu-line-exclusion fix from the reflected-light
#           line (V4.4): Otsu was computing its threshold using the whole
#           image including the calibration line's pixels, which on these
#           backlit photos produced wildly wrong results (200+ mm2 fake
#           "aggregates" that were actually the line itself). Confirmed
#           fixed by excluding the line before computing the threshold.
#
#        2. PORTED resolution-relative blur/morph kernels and the lowered
#           area floor (0.4mm2, with a 1.0mm2 "confident" floor below which
#           a result is marked with "*" and drawn in orange) - same
#           reasoning as the reflected-light line, not re-derived here.
#
#        3. NEW, backlight-specific: added MAX_ASPECT_RATIO (6.0). The
#           transparent solution boundary can show a thin chromatic fringe
#           (light refracting/dispersing at the edge, lit from behind) that
#           passes the red color filter. When the real line failed the
#           existing lower bound (same known painting-consistency issue as
#           reflected light), this fringe was wrongly accepted as a
#           replacement instead of correctly reporting calibration failure -
#           confirmed on 3 of the 9 test photos. The fringe reached aspect
#           ratios of 9-16; no real line in this entire project (either
#           setup) has ever exceeded 3.91. Tried 4 other ways to tell the
#           fringe from a real line first (extent, saturation, straightness,
#           brightness) - none separated them reliably. The aspect-ratio
#           cap, grounded in measured real-line values rather than a new
#           guess, is what actually worked.
#
#        NOT FIXED, and deliberately not chased further this round: the
#        backlight source in these 9 photos was visibly uneven (individual
#        bulbs visible as a bright/dark pattern across the frame, not a
#        smooth gradient). Confirmed this is a different, harder problem
#        than the smooth-gradient case the illumination flattening was
#        built for - a single large blur kernel can flatten a monotonic
#        gradient, but a periodic multi-bulb pattern needs something else.
#        A smaller kernel measurably reduced the false "dark" area (59% of
#        the frame down to ~1%), but the resulting aggregate shapes had
#        suspiciously low circularity and oversized bounding boxes relative
#        to their area - not confident this is cleanly isolating the real
#        aggregate vs. still partly capturing lighting noise. Did not
#        change the default kernel size based on this one problematic
#        lighting setup - that risks the same mistake as before (tuning to
#        one dataset that doesn't generalize). Revisit once a more uniform
#        light source is available; the numbers from these 9 specific
#        photos should not be treated as final results.
#
# B1.2 - PORTED batch analysis from the reflected-light line: the popup
#        now accepts multiple images at once, each photo still gets its
#        own result window, and a comparison-table window appears at the
#        end if more than one photo was selected. Single-photo runs are
#        unaffected. Table columns adapted to backlight terminology
#        (optical density index / combined optical density instead of
#        relative intensity index / combined index) - same structure,
#        same "*" small-aggregate marking, otherwise unchanged from the
#        reflected-light version.
#
# B1.3 - Calibration line's saturation/value floors made ADAPTIVE per-photo
#        instead of fixed constants (old: S>=80, V>=40 always). Motivated by
#        a deliberate side-by-side test with the same setup shot under two
#        different lighting conditions: a dimmer/weaker backlight legitimately
#        lowers the brightness of every pixel in the frame, including the
#        calibration line's own paint, so a fixed absolute floor either passes
#        comfortably under bright light or fails outright under dim light with
#        no middle ground - not a measurement problem, a binary pass/fail
#        problem. Now each photo's own 99th-percentile S and V (not the true
#        max, to avoid one hot pixel skewing it) sets the reference, and the
#        line must clear 35% of that reference to count - so the requirement
#        scales with whatever this photo's own dynamic range actually is.
#        A small absolute backstop (S>=30, V>=15) is kept underneath so a
#        genuinely unusable/washed-out photo with nothing saturated or bright
#        in it still can't pass by relative comparison alone. Diagnostics
#        (measured 99th-pct S/V and the derived floors) are now printed for
#        every photo, pass or fail, specifically so this can be checked
#        empirically across different lighting setups instead of trusted
#        blindly - same evidence-first approach as B1.1's MAX_ASPECT_RATIO.
#
#        NOT ADDRESSED this round: HUE_TOLERANCE is still a fixed constant.
#        Hue is far less sensitive to exposure than S/V (that's why only S/V
#        needed to become adaptive here), but a strong white-balance shift
#        between lighting setups could still shift the paint's apparent hue
#        enough to matter - untested so far, revisit if a real photo shows it.
#
#        ALSO NOT ADDRESSED: photos analyzed during this test contained
#        several sizeable air bubbles in the droplet, which the aggregate
#        detector has no way to distinguish from a real aggregate (no shape/
#        isolation filter - area is the only criterion). This is independent
#        of the lighting-robustness work above and is a separate, still-open
#        false-positive risk.
#
#        Also added in B1.3: per-photo detection diagnostics (Otsu foreground
#        coverage % and the largest rejected candidate blob(s) with the reason
#        they were rejected). Motivated by a "0 aggregates found" result on a
#        real test photo with no way to tell WHY - whether Otsu found nothing
#        resembling an aggregate at all (real low-contrast problem) vs. found
#        something but it got thrown out by a filter (border-artifact size or
#        below the area floor - a tunable-constant problem instead). This
#        distinction is the difference between "the photo's contrast may be
#        too low for this method" and "a constant needs adjusting" - not
#        guessable from the final count alone.
#
#        Coverage is now reported BOTH raw (right after Otsu, before
#        morphology) and after morphology, instead of just one number.
#        Two real test photos both already showed 0.00% AFTER morphology -
#        but that alone doesn't distinguish "Otsu found nothing at all" from
#        "Otsu found something thin (e.g. the hand-traced pencil outline of
#        the droplet boundary, or bubble rims - both visibly higher-contrast
#        in these photos than the pale aggregate smear itself) that morphology
#        then eroded away." The raw number settles which of those it is -
#        not yet re-run against these two photos as of this entry.
#
# B1.4 - CORRECTNESS FIX, confirmed against a real 8-photo voltage sweep
#        (2.8V-3.5V) where every single photo's reported "aggregate" was
#        actually the calibration bar, not the real sample - visually (the
#        green contour traced the bar's shape) and numerically (area stayed
#        flat ~282-297mm2 across a 7x background-brightness range, and
#        optical density index read exactly -0.000/0.000 in all 8 photos).
#
#        Root cause was NOT the final-result line-exclusion logic (that
#        code was already present and correct, ported from V4). It was the
#        illumination-flattening step: its background-trend estimate uses
#        one Gaussian blur sized to half the image's smaller dimension, and
#        on this setup the calibration bar is a solid, sizeable object (not
#        a thin line) - large enough that its own darkness reached into
#        that blur radius and dragged the "expected local background" down
#        across a wide halo around it (visibly a bright ring once
#        flattened). Once the bar's own pixels were excluded from Otsu's
#        histogram (correct, unchanged from B1.1) but its contamination of
#        the *background estimate* was not, Otsu was left splitting a
#        subtly skewed near-uniform background against nothing - and
#        landed on a threshold (~143) that misread ~90% of the entire frame
#        as foreground. Confirmed by direct measurement on a real photo:
#        excluding the bar from the illumination blur too (filling its
#        region with the surrounding background's median level before that
#        blur runs) brought Otsu back to a sensible ~96-99 and foreground
#        coverage back to ~6% of frame, matching the bar+sample's real
#        physical footprint.
#
#        Fix: flatten_illumination() now accepts the same calibration-line
#        exclusion mask already computed for Otsu, and blanks that region
#        with the surrounding median gray level before running the giant
#        blur. Nothing else in the pipeline changed - the existing final-
#        result exclusion (geometric box + connected-component touching
#        the line's center) was already correct and now has a sane mask to
#        work with.
#
#        Re-validated on all 8 real voltage-sweep photos: the green contour
#        now sits on the actual sample in every photo, the calibration bar
#        is correctly excluded (drawn with no contour), and optical density
#        index now reports plausible non-zero values that vary sensibly
#        with exposure instead of a flat -0.000 across the board.
#
# B1.5 - PORTED two-threshold hysteresis edge recovery from the reflected-
#        light line (V4.4's hysteresis_edge_recovery / hysteresis_overgrowth_fix).
#        Confirmed real on this side too: a long, faint wispy tail/strand
#        visibly connected to the main aggregate in the raw photo was being
#        dropped entirely - the green contour stopped right at the solid
#        core, same visual symptom across the 2.9V/3.0V/3.3V photos of the
#        voltage sweep. Same root cause as the reflected-light case:
#        illumination flattening's background estimate gets pulled toward
#        the core's own brightness nearby, so a real-but-faint attached
#        edge reads as "about average" after flattening and never reaches
#        the single global Otsu cut.
#
#        Adapted for backlight's inverted polarity: V4's loose threshold is
#        raw_background_mean + N*std ("moderately brighter than background"
#        for a bright-on-dark aggregate). Here it's
#        raw_background_mean - N*std ("moderately darker than background"),
#        since the backlit aggregate is dark on a bright background. Derived
#        from this photo's own raw-image background pixels (core regions +
#        calibration bar excluded via the existing exclusion mask), not a
#        hardcoded brightness constant - same reasoning as the reflected-
#        light side. Growth from the confirmed core is only allowed through
#        direct pixel connectivity (cv2.connectedComponents reconstruction),
#        so isolated dust/noise that happens to pass the loose threshold on
#        its own is never pulled in.
#
#        Also ported, not re-derived: the no-man's-land fix
#        (build_growth_distance_cap() - a pixel reachable by more than one
#        confirmed core's allowed zone is contested ground, excluded from
#        growth for both, so hysteresis can never bridge two separate real
#        aggregates into one blob) and the growth-distance cap itself
#        (GROWTH_MAX_RADIUS_MULTIPLIER x each blob's own equivalent radius).
#        This was the actual root cause of an overgrowth bug found on the
#        reflected-light side - not the first thing tried there (a tighter
#        loose threshold was tried first and wiped out real recovery too;
#        see V4.4's own changelog) - so it's ported as the real fix from the
#        start here rather than repeating that trial-and-error.
#
#        Overgrowth flag ported too: if an aggregate's hysteresis-recovered
#        area exceeds OVERGROWTH_RATIO (50%) of its own core area, it's
#        drawn with a thick red outline and a "!" label plus a printed
#        warning, instead of looking identical to a clean detection.
#
#        Hysteresis-recovered pixels are rendered as a thin cyan OUTLINE
#        (2px), not a solid fill - applying from the start the same fix
#        already made to V4's cyan overlay (an early solid-fill version
#        obscured the underlying photo and was changed to outline-only).
#
#        Bubble exclusion (Hough circles) was NOT ported - no bubble-
#        adjacency failure mode has been observed on the backlight side yet;
#        revisit if one shows up in real photos.
#
#        Confirmed the calibration-bar exclusion from B1.4 is untouched and
#        still works with hysteresis added on top: growth is masked out of
#        the bar's region both before and after the connectivity step,
#        exactly where the bar's own exclusion mask already applied to the
#        core. Re-validated against the full 9-photo voltage sweep
#        (2.8V-3.5V plus the unlabeled 3V): core/recovered/total area
#        reported per aggregate for every photo, wispy tail now outlined and
#        included in the 2.9V/3.0V/3.3V photos where it was directly
#        observed being dropped, and output is pixel-identical to B1.4 on
#        every photo where hysteresis found nothing to grow.
#
# B1.6 - RENDER FIX, no detection/threshold/growth/area logic touched. A
#        follow-up diagnostic (tracing 3.0V/3.2V/3.4V pixel-by-pixel through
#        every hysteresis step) found the B1.5 computation itself was
#        correct and recovering MORE real area at higher brightness
#        (0.375mm2 at 3.2V, 0.568mm2 at 3.4V, vs 0.343mm2 at 3.0V) - but the
#        cyan outline meant to show that recovery was rendering as
#        essentially invisible at 3.2V/3.4V (27 actual cyan pixels in the
#        image vs 4211 at 3.0V), making it look like recovery had stopped
#        working when it hadn't.
#
#        Root cause: the cyan contour was found with
#        cv2.findContours(hysteresis_mask, RETR_EXTERNAL, ...) - since the
#        recovered ring sits directly against the outside of the core,
#        RETR_EXTERNAL's outer boundary for that ring is geometrically the
#        SAME path as the green core/total contour drawn right after it.
#        RETR_EXTERNAL never returns the ring's inner edge (the one facing
#        the core), which is the only boundary that would actually look
#        distinct from green. So cyan and green were always tracing the
#        same nominal path; green (drawn second) painted over it. Whatever
#        cyan survived was incidental subpixel jitter between two
#        independently-approximated polygons, not a deliberate
#        distinguishing render - explaining why it was sometimes barely
#        visible and sometimes not, independent of how much was actually
#        recovered.
#
#        Considered reversing the draw order alone (draw cyan on top of
#        green, same masks as before) - rejected after confirming via the
#        diagnostic that the outer boundaries genuinely coincide most of
#        the time, so reordering alone would just move the overdraw
#        problem to whichever line is now drawn first, not fix it.
#
#        Fix: cv2.findContours(hysteresis_mask, RETR_CCOMP, ...) instead of
#        RETR_EXTERNAL. Since hysteresis_mask is already ring-only (core
#        pixels excluded), RETR_CCOMP returns the ring's outer boundary AND
#        its inner boundary (as a hole) - drawContours(-1, ...) draws both.
#        The inner edge sits exactly at the true core/recovered interface
#        and is never part of green's path, so it now survives regardless
#        of draw order. Green is also now drawn BEFORE cyan (previously
#        cyan-then-green) so the ring's outer edge is fully visible too,
#        rather than fighting green for the same pixels - this reordering
#        alone was evaluated and found insufficient without the RETR_CCOMP
#        change, so both were needed together.
#
#        No detection, threshold, growth-cap, no-man's-land, or area
#        calculation code changed - core_area_mm2, hysteresis_area_mm2, and
#        true_area_mm2 are computed from the same pixel masks as B1.5 and
#        are numerically identical before/after this fix on all photos
#        tested. Re-validated on the full 8-photo voltage sweep: cyan is
#        now visibly present and scales with the printed recovered-area
#        number in every photo, most visibly at 3.2V/3.4V where it was
#        previously near-invisible; no other photo's rendering changed
#        beyond the intended cyan visibility fix.
#
# B1.7 - ADDED: CSV master log (csv-master-log), PORTED from the reflected-
#        light line's V4.5. Every run now appends one row per PHOTO to
#        plasma_analysis_master_log_backlight.csv, next to this script,
#        instead of results only living in on-screen matplotlib windows -
#        a separate file from V4's own master CSV, so the two lighting
#        setups' logs never mix. Per-photo summary rows (not per-
#        aggregate), matching the existing batch comparison table, and
#        append-only across every session indefinitely (header written
#        once on first use). Calibration failures are logged too, with
#        blank measurement fields and the failure reason recorded, and so
#        is the case where calibration succeeds but 0 aggregates are
#        found (aggregate_count = 0, not skipped) - both are real,
#        loggable outcomes, not errors. Reuses B1's own existing
#        compute_photo_summary() (already factored out since B1's first
#        batch-table version) for every number, so the CSV can never
#        disagree with what the on-screen tables show for the same photo.
#        Column names follow V4's CSV snake_case style, but only where the
#        underlying number is genuinely the same thing - B1's headline
#        number is optical_density_index / combined_optical_density (Beer-
#        Lambert-grounded transmittance), not V4's relative_intensity_index
#        / combined_index, so the columns are named for what they actually
#        are instead of forcing V4's names onto a different measurement.
#        Also logs B1's own adaptive HSV calibration floors (hsv_
#        saturation_min/value_min and their 99th-percentile references,
#        from B1.3) as B1-specific columns with no V4 equivalent, since
#        tracking those across many photos/lighting setups over time is
#        exactly the empirical check B1.3 was added to enable.
#
# B1.8 - CHANGED: MINIMUM_AREA_MM2 lowered from 0.4 to 0.2 (lower-area-
#        floor). WHY: a real aggregate was measured at ~0.24mm2, ~5 sigma
#        below background, consistently across 4 independent voltage
#        captures of the same sample (Acetic+BSA_3.8pH, 2.9-3.2V) - not a
#        single-photo guess. CONFIDENT_AREA_MM2 stays at 1.0, so anything
#        from 0.2 to 1.0mm2 (now including this aggregate's size range)
#        still gets the existing "*" flag / orange outline / "worth a
#        visual check" treatment, exactly the same mechanism already used
#        for 0.4-1.0mm2, just extended slightly lower - not a new
#        mechanism, and never auto-counted as unambiguous.
#        ACCEPTED TRADEOFF, not hidden: more small dust specks may now
#        also cross this lower floor and get counted-with-a-flag. This is
#        a known, deliberate cost of catching the real small aggregate,
#        not an oversight.
#        TESTED against 7 real photos (the 4 voltage captures above, plus
#        Black/Black2/Blue) before and after this change. Result: the 4
#        voltage photos and the 3 backlight (Black/Black2/Blue) photos
#        were UNCHANGED by this change specifically - in this batch, the
#        largest rejected below-floor candidate in any photo was
#        ~0.18mm2, still under the new 0.2 floor, and Black/Black2/Blue
#        found zero candidate blobs at all (not a floor issue - Otsu
#        found no qualifying dark region in those 3 frames). Lowering the
#        floor is still correct policy given the confirmed real aggregate
#        size, but flagging here that these specific 7 files don't
#        demonstrate it visually - see the before/after test table kept
#        alongside this change for the full numbers.
#
# B1.9 - VISIBILITY IMPROVEMENT, no detection/threshold/growth/area logic
#        touched (OVERGROWTH_RATIO, hysteresis growth, and the area floors
#        are all unchanged). PROMPTED BY B1.8: lowering the area floor to
#        0.2mm2 means small aggregates now routinely sit close to the size
#        where hysteresis-recovered edge pixels are a meaningful fraction
#        of the reported total, but the core-vs-recovered breakdown
#        (core_area_mm2 / hysteresis_area_mm2, computed since B1.5) only
#        ever reached console/debug output - there was no way to sanity-
#        check a small, overgrowth-flagged result at a glance without
#        reading the terminal log alongside the table.
#
#        Added "Core area (mm2)" and "Recovered area (mm2)" columns to
#        both the per-photo results table and the batch comparison table,
#        next to the existing "Area (mm2)" column. Per requirement, did
#        NOT just trust that core + recovered sums to the existing Area
#        column: core_area_mm2/hysteresis_area_mm2 are raster PIXEL COUNTS
#        (a partition of the aggregate's mask), while true_area_mm2 (the
#        existing Area column) is a cv2.contourArea (Green's-theorem
#        polygon) measurement - the same measure already used everywhere
#        else in this file for area-floor classification, so it has to
#        stay authoritative. Pixel-count area and contour-polygon area are
#        not the same number (boundary-rasterization effects), most
#        visible as a fraction of total area on exactly the small
#        aggregates this change was requested for. So the table's Core/
#        Recovered columns are DERIVED to always reconcile to the Area
#        column by construction (Recovered = Area - Core, with Core
#        clamped to never exceed Area) instead of showing two
#        independently-measured numbers that could silently fail to add
#        up. A console NOTE prints if that clamp ever actually engages on
#        a real photo, so the discrepancy stays visible instead of being
#        silently absorbed. The overgrowth flag itself (is_overgrown) is
#        untouched - still driven by the original raw pixel-count
#        hysteresis_area_mm2 vs core_area_mm2, per the no-detection-change
#        requirement.
#
#        Any row whose aggregate trips the existing overgrowth flag now
#        gets its Recovered area cell rendered bold red (same red as the
#        "!" marker/outline already used elsewhere), instead of looking
#        identical to a clean row. The actual overgrowth RATIO (recovered/
#        core, not just the boolean) is now printed for every flagged
#        aggregate in the console WARNING line, and shown next to the "!"
#        in both tables ("#1! (2.3x)" per-aggregate; "filename !(2.3x)"
#        using that photo's worst ratio in the batch table) - the closest
#        matplotlib's static table format gets to a tooltip.
#
# B1.10 - CHANGED: MINIMUM_AREA_MM2 lowered from 0.2 to 0.14 (lower-area-
#        floor-2). WHY: B1.9's per-candidate diagnostic logging (rejected
#        blobs, not just the top 5) was used to directly measure a specific
#        below-left blob sitting right next to the main wispy aggregate in
#        all 4 voltage shots (Acetic+BSA_3.8pH, 2.9-3.2V). Measured area:
#        0.1774mm2 (2.9V), 0.1626mm2 (3.0V), 0.1627mm2 (3.1V), 0.1459mm2
#        (3.2V) - a real, physically-measured target, not an estimate, and
#        the same pixel location (~x2565,y1919 in 2.9V, matching relative
#        position in the other 3) in every shot. The floor is set to
#        0.14mm2, below all 4 measurements with margin, so this blob is now
#        counted in every voltage shot.
#        TESTED against all 4 voltage shots: confirmed the target blob is
#        now a counted aggregate in all 4, and is mostly SOLID CORE, not
#        hysteresis growth (recovered area is 0 or near-0 in every shot -
#        see per-shot Core/Recovered breakdown kept alongside this change).
#        Also confirmed the next-largest previously-rejected candidate near
#        it (~x2700,y1982, 0.0747-0.0967mm2 across the 4 shots) still sits
#        well under the new 0.14 floor and is NOT counted - the new floor
#        doesn't just let everything below the old 0.2 through.
#        REGRESSION-TESTED against the full original 7-photo set (the 4
#        voltage shots above, plus Black/Black2/Blue) - see the before/
#        after test table kept alongside this change for the full numbers.
#        CONFIDENT_AREA_MM2 stays at 1.0, so this blob (well under 1.0mm2)
#        still gets the existing "*" flag / orange outline - counted, but
#        never presented as an unambiguous, no-need-to-double-check size.
# =============================================


# =============================================
# PLASMA PROTEIN AGGREGATION ANALYSIS TOOL — BACKLIGHT VERSION
# =============================================
# SCOPE: backlight setup only. Companion to the reflected-light V4.x line,
# not a replacement for it - the two are separate tools for separate
# physical setups.
#
# CORE PHYSICAL DIFFERENCE FROM REFLECTED LIGHT:
#   Reflected light: light source in front, aggregate reflects light back
#                     -> aggregate is BRIGHT on a dark/mid background.
#   Backlight:        light source behind the sample, aggregate blocks /
#                     scatters the transmitted light -> aggregate is DARK
#                     on a bright background. This flips which side of
#                     the Otsu threshold counts as "aggregate" - see the
#                     THRESH_BINARY_INV note below.
#
# INHERITED UNCHANGED FROM V4.2 (reflected-light), because none of this
# logic depends on which side is bright vs dark:
#   - Calibration line detection: same HSV hue-band approach, same
#     ratio-based shape filter (length vs thickness + length-vs-diagonal
#     floor), same explicit exclusion of the line's own region before
#     aggregate detection.
#   - Illumination flattening: a real green-background reflected-light
#     photo exposed a lighting-gradient bug that broke a single global
#     Otsu threshold (see V4.2's changelog). The same fix - estimate the
#     slow lighting trend with a large blur and subtract it out - is
#     included here from the start, since a backlight source can just as
#     easily be uneven (hot spots, vignetting) as room lighting can.
#   - Full-frame border-artifact guard.
#   - Parameter popup (image file, reference length, calibration line's
#     color, optional thickness), with the same terminal fallback.
#   - Volume: optional thickness x area, "TBD" if not given, never errors.
#   - 2x2 display layout (original / calibration / clean result / info).
#
# WHAT ACTUALLY CHANGES FOR BACKLIGHT:
#   1. Threshold direction: cv2.THRESH_BINARY_INV instead of
#      cv2.THRESH_BINARY, so the DARK side of the cut counts as
#      "aggregate" instead of the bright side. Otsu still picks the cut
#      value the same way; only which side is foreground changes.
#   2. Intensity index. The professor's Phase 4 note lists three
#      backlight-specific candidate parameters:
#        - transmitted intensity (raw mean brightness inside the
#          aggregate) - NOT used as the headline number, for the same
#          reason raw mean intensity wasn't used on the reflected-light
#          side: it's an absolute number, vulnerable to the backlight
#          source's own brightness drifting between shots.
#        - relative transmittance (aggregate_mean / background_mean) -
#          computed internally as a self-correcting ratio (same role as
#          relative_intensity_index on the reflected-light side), but not
#          shown as its own headline number, to avoid reporting two
#          numbers that carry the same information twice (see next item).
#        - optical-density-like index - USED as the headline number:
#            optical_density_index = -log10(relative_transmittance)
#          This is the one with a real physical grounding (Beer-Lambert):
#          unlike raw transmittance, optical density scales roughly
#          linearly with how much material is actually in the light's
#          path, which is a more meaningful "amount of aggregate" signal
#          than a fraction-of-light-passing-through number on its own.
#        - combined_optical_density = true_area_mm2 * optical_density_index
#          is the area-weighted version, playing the same role as
#          combined_index did on the reflected-light side: answers
#          "equal footprint, different density, should not score equal."
# =============================================


# =============================================
# INTERNAL CONSTANTS — fixed, not shown to the user.
# Inherited starting values from the reflected-light line; flagged with
# [UNCONFIRMED FOR BACKLIGHT] where they haven't been independently
# checked against a real backlit photo yet.
# =============================================

# --- Otsu thresholding ---
BLUR_SIZE_FRACTION  = 0.0043  # ~15px at the ~3456px-wide photos tested so far
MORPH_SIZE_FRACTION = 0.0009  # ~3px at the same reference resolution
# Ported from the reflected-light line: fixed pixel kernels don't scale with
# image resolution, which made a thin/small real feature more vulnerable to
# being eroded away on some resolutions than others.

# --- Illumination flattening (corrects uneven backlight brightness) ---
ILLUMINATION_KERNEL_FRACTION = 0.5
BORDER_ARTIFACT_AREA_FRACTION = 0.9

# --- Calibration line color detection (HSV) ---
HUE_CENTERS = {'RED': 0, 'GREEN': 60, 'BLUE': 120}  # OpenCV hue scale is 0-180
HUE_TOLERANCE  = 15
# Saturation/value floors are now ADAPTIVE per-photo (see compute_adaptive_hsv_floors)
# instead of fixed constants. Fixed absolute floors (old values: S>=80, V>=40) broke
# under deliberately-varied lighting: a dimmer/weaker backlight shot can legitimately
# push the whole frame's brightness down, including the calibration line's own pixels,
# causing a hard calibration failure even though the line is clearly visible to the eye.
# Instead we require the line to be among the most saturated/bright content in ITS OWN
# photo (relative to that photo's own 99th-percentile S/V), with a small absolute floor
# kept only as a backstop against accepting pure noise in a genuinely unusable photo.
RELATIVE_SATURATION_FRACTION = 0.35  # line's S must be >= 35% of this photo's own high-S reference
RELATIVE_VALUE_FRACTION      = 0.35  # same idea for V
ABSOLUTE_SATURATION_FLOOR    = 30    # backstop: below this, treat as noise regardless of exposure
ABSOLUTE_VALUE_FLOOR         = 15    # backstop: same idea for V

# --- Calibration line shape filter (ratio-based) ---
MIN_ASPECT_RATIO         = 3.0   # confirmed against real backlit photos: real
                                   # lines measured 2.27-3.91 across this whole
                                   # project (both setups), same known
                                   # painting-consistency issue as reflected
                                   # light - not a new backlight-specific
                                   # problem.
MAX_ASPECT_RATIO         = 6.0   # NEW, confirmed necessary by testing: the
                                   # transparent solution boundary can show a
                                   # thin chromatic fringe (a refraction/
                                   # dispersion artifact at the edge, lit from
                                   # behind) that passes the color filter and
                                   # can reach extreme aspect ratios (9-16,
                                   # measured directly) because a curving thin
                                   # band inflates minAreaRect's apparent
                                   # length. When the real line fails the
                                   # lower bound, this fringe was getting
                                   # wrongly accepted as a replacement instead
                                   # of correctly reporting calibration
                                   # failure. No real line in this entire
                                   # project has ever exceeded 3.91 - this cap
                                   # has a comfortable margin on both sides.
MIN_LINE_LENGTH_FRACTION = 0.05

# --- Aggregate noise filter ---
# Ported from the reflected-light line, where two shape-based attempts to
# auto-separate real small aggregates from dust both failed real-photo
# testing (see V4.4's changelog). The floor is set below confirmed-real
# small-aggregate sizes; anything between this and CONFIDENT_AREA_MM2 is
# counted but visually flagged (orange instead of green, "*" in tables)
# rather than trusted unconditionally.
MINIMUM_AREA_MM2   = 0.14
CONFIDENT_AREA_MM2 = 1.0
BORDERLINE_AREA_MM2 = 0.05  # true noise floor - nothing below this is even
                              # reported, let alone counted (single/few-pixel
                              # JPEG artifacts, not real candidates either way)

# --- Hysteresis edge recovery (recovers real-but-faint boundary pixels the
#     flattened Otsu threshold misses near a solid core) ---
# PORTED from the reflected-light line (V4.4). Direction adapted for
# backlight: the loose threshold there looks for pixels moderately
# BRIGHTER than background (bright-on-dark aggregate); here it looks for
# pixels moderately DARKER than background (dark-on-bright aggregate).
# Both are computed the same way - raw_background_mean +/- N*raw_background_std,
# using this photo's own background pixels (core regions and the
# calibration bar excluded) - so the threshold adapts per photo instead of
# using a fixed brightness constant.
LOOSE_THRESHOLD_STD_MULTIPLIER = 2.0

# Growth from any one core blob can only reach this many times that blob's
# own equivalent radius (sqrt(area/pi)) further out. On its own this only
# stops growth into empty space - it does NOT stop two separate confirmed
# aggregates sitting close together from bridging into one blob, since each
# one's own allowed zone can still reach the other. build_growth_distance_cap()
# also excludes any pixel claimed by more than one core's allowed zone
# (contested no-man's-land between two different real objects), which is
# what actually prevents that bridging - ported unchanged from the
# reflected-light line, where a tighter threshold alone was tried first and
# failed to fix an equivalent bridging bug.
GROWTH_MAX_RADIUS_MULTIPLIER = 1.5

# If an aggregate's hysteresis-recovered area exceeds this fraction of its
# own core area, the result is flagged as suspicious growth (red outline +
# printed warning) instead of being silently accepted.
OVERGROWTH_RATIO = 0.5

# BGR draw colors, just for the calibration overlay visualization
DRAW_COLORS = {'RED': (0, 0, 255), 'GREEN': (0, 255, 0), 'BLUE': (255, 0, 0)}
HYSTERESIS_COLOR_BGR = (255, 255, 0)  # cyan outline for grown-in pixels
OVERGROWN_COLOR_BGR = (0, 0, 255)     # red outline for flagged/suspicious growth


# =============================================
# CSV MASTER LOG (B1.7) — PORTED from the reflected-light line's V4.5
# =============================================
# One row is appended per PHOTO (not per aggregate) to a single, growing
# CSV that accumulates across every session, every student, indefinitely -
# a separate file from V4's own master CSV so the two lighting setups' logs
# never mix. Calibration failures are logged too (with blank measurement
# fields), since failure rate itself is useful data - same reasoning as V4.5.
#
# B1.9 FIX: MASTER_CSV_PATH used to be derived from os.path.dirname(__file__)
# - "next to whichever copy of this script happens to be running". That's
# wrong: this script gets copied to different folders (this repo's legacy/,
# plasma/src/, other working copies), and each copy silently started its
# own separate log next to itself instead of sharing one history. This
# already happened for real and fragmented the log across 6 separate files
# that had to be manually recovered and merged. Fixed to a single hardcoded
# absolute path at the project's data folder, independent of __file__, so
# every copy of this script (wherever it's run from) writes to the same
# file. The folder is created if it doesn't exist. The resolved path is
# still printed at startup (unchanged) so it's always obvious which file is
# being written to.
MASTER_CSV_FILENAME = "plasma_analysis_master_log_backlight.csv"
MASTER_CSV_DIR = r"C:\Users\66950\Desktop\Projects in github\Plasma\data"
os.makedirs(MASTER_CSV_DIR, exist_ok=True)
MASTER_CSV_PATH = os.path.join(MASTER_CSV_DIR, MASTER_CSV_FILENAME)

CSV_COLUMNS = [
    "run_timestamp", "filename", "version", "build_tag",
    "calibration_status", "calibration_failure_reason",
    "reference_mm", "calibration_color", "mm_per_px", "background_mean",
    "aggregate_count", "total_area_mm2", "total_holes", "avg_circularity",
    "avg_optical_density_index", "total_combined_optical_density",
    "total_volume_mm3", "has_flagged_small_aggregate",
    # B1-specific (adaptive HSV calibration floors, B1.3 - no V4 equivalent):
    "hsv_saturation_min", "hsv_value_min", "hsv_s_ref_p99", "hsv_v_ref_p99",
]


def append_photo_to_master_csv(filename, params, result):
    """Append one PER-PHOTO summary row to the master CSV log (creates the
    file with a header on first use, appends after that - never
    overwrites). Reuses compute_photo_summary() so this can never disagree
    with what the on-screen tables show for the same photo. Logs
    calibration failures too, with measurement fields left blank, so
    failure rate over time is visible in the same file rather than lost."""
    file_is_new = (not os.path.exists(MASTER_CSV_PATH)) or os.path.getsize(MASTER_CSV_PATH) == 0

    row = {col: "" for col in CSV_COLUMNS}
    row["run_timestamp"] = datetime.now().isoformat(timespec="seconds")
    row["filename"] = filename
    row["version"] = VERSION
    row["build_tag"] = BUILD_TAG
    row["reference_mm"] = params.get("reference_mm", "")
    row["calibration_color"] = params.get("color_mode", "")

    if not result.get("success"):
        row["calibration_status"] = "failed"
        row["calibration_failure_reason"] = result.get("reason", "").replace("\n", " ")
        hd = result.get("hsv_diagnostics")
        if hd:
            row["hsv_saturation_min"] = f'{hd["saturation_min"]:.1f}'
            row["hsv_value_min"] = f'{hd["value_min"]:.1f}'
            row["hsv_s_ref_p99"] = f'{hd["s_ref"]:.1f}'
            row["hsv_v_ref_p99"] = f'{hd["v_ref"]:.1f}'
    else:
        row["calibration_status"] = "success"
        row["mm_per_px"] = f'{result["mm_per_px"]:.5f}'
        row["background_mean"] = f'{result["background_mean"]:.1f}'
        hd = result["hsv_diagnostics"]
        row["hsv_saturation_min"] = f'{hd["saturation_min"]:.1f}'
        row["hsv_value_min"] = f'{hd["value_min"]:.1f}'
        row["hsv_s_ref_p99"] = f'{hd["s_ref"]:.1f}'
        row["hsv_v_ref_p99"] = f'{hd["v_ref"]:.1f}'

        summary = compute_photo_summary(result)
        if summary is None:
            # calibration succeeded but zero aggregates found - still a
            # real, loggable outcome, not an error
            row["aggregate_count"] = 0
        else:
            row["aggregate_count"] = summary["count"]
            row["total_area_mm2"] = f'{summary["total_area"]:.3f}'
            row["total_holes"] = summary["total_holes"]
            row["avg_circularity"] = f'{summary["avg_circularity"]:.3f}'
            row["avg_optical_density_index"] = f'{summary["avg_optical_density"]:.3f}'
            row["total_combined_optical_density"] = f'{summary["total_combined"]:.3f}'
            row["total_volume_mm3"] = (
                f'{summary["total_volume"]:.3f}' if summary["total_volume"] is not None else "TBD"
            )
            row["has_flagged_small_aggregate"] = "TRUE" if summary["has_small"] else "FALSE"

    with open(MASTER_CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if file_is_new:
            writer.writeheader()
        writer.writerow(row)


# =============================================
# ILLUMINATION FLATTENING (corrects uneven lighting before thresholding)
# =============================================

def make_odd(n):
    """Gaussian blur kernels must be odd-sized."""
    n = int(n)
    return n if n % 2 == 1 else n + 1


def classify_contours(binary_mask, minimum_area_px, confident_area_px,
                       borderline_area_px, border_artifact_area_px,
                       mm2_per_px2, collect_diagnostics=False):
    """Runs findContours + the area-floor/border-artifact rules on a binary
    mask. PORTED from the reflected-light line so the pre-growth core pass
    and the post-growth final pass apply exactly the same rules. When
    collect_diagnostics is True, also returns the same "why was this
    rejected" list B1 has always printed (border-artifact vs below-floor),
    computed from the core pass only - unaffected by hysteresis growth."""
    contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    outer_contours = []
    hole_contours = []
    small_indices = set()  # counted, but below the confident floor - gets a "*" mark
    rejected = []  # (area_mm2, reason) - only populated if collect_diagnostics
    if hierarchy is not None:
        for i, contour in enumerate(contours):
            parent_idx = hierarchy[0][i][3]
            area_px = cv2.contourArea(contour)
            if parent_idx == -1:
                area_mm2 = area_px * mm2_per_px2
                if area_px > border_artifact_area_px:
                    if collect_diagnostics:
                        rejected.append((area_mm2, 'border-artifact (near full-frame blob, never counted)'))
                elif area_px <= minimum_area_px:
                    if collect_diagnostics and area_px > borderline_area_px:
                        rejected.append((area_mm2, f'below minimum floor ({MINIMUM_AREA_MM2}mm2)'))
                else:
                    outer_contours.append((contour, i))
                    if area_px <= confident_area_px:
                        small_indices.add(i)
            else:
                hole_contours.append((contour, parent_idx))
    rejected.sort(key=lambda x: x[0], reverse=True)
    return contours, hierarchy, outer_contours, hole_contours, small_indices, rejected


def _dilated_crop_for_label(labels, label, stats, seed_mask_shape):
    """Local (bounding-box-limited) dilation of one connected-component
    label by its own GROWTH_MAX_RADIUS_MULTIPLIER x equivalent-radius cap.
    Returns (y0, y1, x0, x1, dilated_crop) so the caller can place it back.
    PORTED unchanged from the reflected-light line - polarity-agnostic, it
    only operates on the binary core mask's shape/connectivity."""
    area_px = stats[label, cv2.CC_STAT_AREA]
    equivalent_radius = np.sqrt(area_px / np.pi)
    cap_radius = max(3, int(round(GROWTH_MAX_RADIUS_MULTIPLIER * equivalent_radius)))

    x = stats[label, cv2.CC_STAT_LEFT]
    y = stats[label, cv2.CC_STAT_TOP]
    w = stats[label, cv2.CC_STAT_WIDTH]
    h = stats[label, cv2.CC_STAT_HEIGHT]
    pad = cap_radius + 2
    y0, y1 = max(0, y - pad), min(seed_mask_shape[0], y + h + pad)
    x0, x1 = max(0, x - pad), min(seed_mask_shape[1], x + w + pad)

    component_crop = np.where(labels[y0:y1, x0:x1] == label, 255, 0).astype(np.uint8)
    kernel_size = 2 * cap_radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated_crop = cv2.dilate(component_crop, kernel)
    return y0, y1, x0, x1, dilated_crop


def build_growth_distance_cap(seed_mask):
    """Caps how far hysteresis growth can spread from each core blob, AND
    stops growth from bridging two separate confirmed aggregates together.
    PORTED unchanged from the reflected-light line (see B1.5 changelog for
    why this, not a tighter threshold, is the real fix for overgrowth).

    Each blob may only grow up to GROWTH_MAX_RADIUS_MULTIPLIER times its own
    equivalent radius (sqrt(area/pi)) further out. A pixel reachable by more
    than one core's allowed zone is contested ground between two different
    real objects, not one object's own fringe, so it's excluded from growth
    entirely for everyone (each core keeps its own unambiguous territory
    only)."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(seed_mask, connectivity=8)

    claim_count = np.zeros(seed_mask.shape, dtype=np.uint8)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] <= 0:
            continue
        y0, y1, x0, x1, dilated_crop = _dilated_crop_for_label(labels, label, stats, seed_mask.shape)
        region = claim_count[y0:y1, x0:x1]
        region[dilated_crop == 255] = np.minimum(region[dilated_crop == 255] + 1, 255)

    allowed_mask = np.zeros(seed_mask.shape, dtype=np.uint8)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] <= 0:
            continue
        y0, y1, x0, x1, dilated_crop = _dilated_crop_for_label(labels, label, stats, seed_mask.shape)
        unambiguous = (dilated_crop == 255) & (claim_count[y0:y1, x0:x1] == 1)
        allowed_mask[y0:y1, x0:x1][unambiguous] = 255

    return allowed_mask


def flatten_illumination(gray, exclusion_mask=None):
    """Removes a slow-varying lighting gradient so Otsu compares each pixel
    to its own local expected background instead of one global brightness
    level. Direction-agnostic - works the same whether the foreground ends
    up being the bright side or the dark side of the resulting threshold.

    exclusion_mask (optional): region(s) - e.g. the calibration bar - to
    blank out with the surrounding background level before estimating the
    illumination trend. Confirmed necessary by testing: the illumination
    estimate here is a single blur kernel covering fully half the image's
    smaller dimension, so a large, solid-dark object (like a calibration
    bar sized to be legible/measurable, not a thin line) is well within
    that kernel's reach - the object's own darkness dragged the "expected
    local background" down over a very wide radius around it (a bright
    halo visibly bleeds out from the object once flattened), which in turn
    skewed Otsu's threshold choice badly enough that ~90% of the frame
    read as "foreground" on a real backlit test photo. Filling the
    excluded region before this blur keeps that object from contaminating
    the background estimate anywhere outside its own footprint."""
    h, w = gray.shape
    kernel_size = make_odd(min(h, w) * ILLUMINATION_KERNEL_FRACTION)
    if exclusion_mask is not None and np.any(exclusion_mask):
        gray_for_illum = gray.copy()
        bg_fill_value = int(np.median(gray[exclusion_mask == 0]))
        gray_for_illum[exclusion_mask == 255] = bg_fill_value
    else:
        gray_for_illum = gray
    illumination = cv2.GaussianBlur(gray_for_illum, (kernel_size, kernel_size), 0)
    diff = gray.astype(np.int16) - illumination.astype(np.int16)
    return np.clip(diff + 128, 0, 255).astype(np.uint8)


# =============================================
# CALIBRATION LINE DETECTION (HSV mask is adaptive per-photo from B1.3 on;
# shape filter still identical to reflected-light V4.2)
# =============================================

def compute_adaptive_hsv_floors(hsv_image):
    """Derive this photo's own saturation/value floors instead of using a
    fixed constant across all lighting conditions. Uses the 99th percentile
    (not the true max) so a single blown-out hot pixel can't inflate the
    reference. A photo shot under weaker/dimmer backlight will have a lower
    s_ref/v_ref, which correctly lowers the bar for what counts as "saturated
    enough" or "bright enough" to be the line - while the absolute floors
    below still reject a photo that's just uniformly noisy with nothing
    genuinely saturated or bright in it."""
    s = hsv_image[:, :, 1].astype(int)
    v = hsv_image[:, :, 2].astype(int)
    s_ref = float(np.percentile(s, 99))
    v_ref = float(np.percentile(v, 99))
    saturation_min = max(ABSOLUTE_SATURATION_FLOOR, RELATIVE_SATURATION_FRACTION * s_ref)
    value_min = max(ABSOLUTE_VALUE_FLOOR, RELATIVE_VALUE_FRACTION * v_ref)
    return saturation_min, value_min, s_ref, v_ref


def get_color_mask(hsv_image, color_mode, saturation_min, value_min):
    """HSV hue-band detection of the painted calibration line."""
    h = hsv_image[:, :, 0].astype(int)
    s = hsv_image[:, :, 1].astype(int)
    v = hsv_image[:, :, 2].astype(int)

    center = HUE_CENTERS[color_mode]

    if color_mode == 'RED':
        # red wraps around the 0/180 boundary on OpenCV's hue scale
        hue_mask = (h <= HUE_TOLERANCE) | (h >= 180 - HUE_TOLERANCE)
    else:
        hue_mask = np.abs(h - center) <= HUE_TOLERANCE

    return hue_mask & (s >= saturation_min) & (v >= value_min)


def find_calibration_line(color_mask):
    """Find the painted reference line by shape (length-to-thickness ratio
    + a relative length floor) instead of a fixed pixel cutoff."""
    mask_image = color_mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_h, img_w = color_mask.shape
    image_diagonal = float(np.hypot(img_h, img_w))
    min_length_px = MIN_LINE_LENGTH_FRACTION * image_diagonal

    candidates = []
    for contour in contours:
        if cv2.contourArea(contour) < 5:
            continue
        rect = cv2.minAreaRect(contour)
        (cx, cy), (rw, rh), angle = rect
        length_px = max(rw, rh)
        thickness_px = min(rw, rh)
        if thickness_px <= 0:
            continue
        aspect_ratio = length_px / thickness_px
        if MIN_ASPECT_RATIO <= aspect_ratio <= MAX_ASPECT_RATIO and length_px >= min_length_px:
            candidates.append({
                'rect': rect,
                'length_px': length_px,
                'thickness_px': thickness_px,
                'center': (cx, cy),
            })

    if not candidates:
        return None
    return max(candidates, key=lambda c: c['length_px'])


# =============================================
# PARAMETER INPUT — popup first, terminal as automatic fallback
# (identical to reflected-light V4.2)
# =============================================

def get_parameters_gui():
    import tkinter as tk
    from tkinter import filedialog, ttk

    result = {}
    selected_paths = []  # holds the real full paths; image_path_var only holds a display string
    root = tk.Tk()
    root.title("Plasma Aggregation Analysis (Backlight) - Setup")
    root.geometry("440x440")

    image_path_var = tk.StringVar()
    reference_var  = tk.StringVar()
    color_var      = tk.StringVar(value="Red")
    thickness_var  = tk.StringVar()
    error_var      = tk.StringVar()

    def browse_files():
        paths = filedialog.askopenfilenames(
            title="Select sample image(s) - choose more than one for batch analysis",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.JPG *.JPEG *.PNG"),
                       ("All files", "*.*")]
        )
        if paths:
            selected_paths.clear()
            selected_paths.extend(paths)
            if len(paths) == 1:
                image_path_var.set(os.path.basename(paths[0]))
            else:
                image_path_var.set(f"{len(paths)} files selected")

    def on_run():
        if not selected_paths:
            error_var.set("Please select at least one image file.")
            return

        try:
            reference_mm = float(reference_var.get().strip())
            if reference_mm <= 0:
                raise ValueError
        except ValueError:
            error_var.set("Reference length must be a positive number.")
            return

        color_mode = color_var.get().strip().upper()
        if color_mode not in ('RED', 'GREEN', 'BLUE'):
            color_mode = 'RED'  # dropdown is readonly, this is just a safety net

        thickness_text = thickness_var.get().strip()
        thickness_mm = None
        if thickness_text != "":
            try:
                t = float(thickness_text)
                if t > 0:
                    thickness_mm = t
            except ValueError:
                pass  # leave thickness_mm as None, no error

        result['image_paths'] = list(selected_paths)
        result['reference_mm'] = reference_mm
        result['color_mode']  = color_mode
        result['thickness_mm'] = thickness_mm
        root.destroy()

    pad = {'padx': 16, 'pady': (10, 2)}

    tk.Label(root, text="Image file(s)").pack(anchor='w', **pad)
    file_frame = tk.Frame(root)
    file_frame.pack(fill='x', padx=16)
    tk.Entry(file_frame, textvariable=image_path_var, state='readonly').pack(
        side='left', fill='x', expand=True)
    tk.Button(file_frame, text="Browse...", command=browse_files).pack(
        side='left', padx=(6, 0))
    tk.Label(root, text="Select more than one file for batch analysis",
             fg='gray').pack(anchor='w', padx=16, pady=(2, 0))

    tk.Label(root, text="Reference length (mm)").pack(anchor='w', **pad)
    tk.Entry(root, textvariable=reference_var).pack(fill='x', padx=16)

    tk.Label(root, text="Calibration line's color").pack(anchor='w', **pad)
    ttk.Combobox(root, textvariable=color_var, values=["Red", "Green", "Blue"],
                 state="readonly").pack(fill='x', padx=16)

    tk.Label(root, text="Aggregate thickness (mm) - optional").pack(anchor='w', **pad)
    tk.Entry(root, textvariable=thickness_var).pack(fill='x', padx=16)

    tk.Label(root, textvariable=error_var, fg='red').pack(pady=(10, 0))
    tk.Button(root, text="Run Analysis", command=on_run).pack(pady=16)

    root.mainloop()

    if not result:
        print("\nSetup window closed without running - exiting.")
        sys.exit(0)

    return result


def get_parameters_terminal():
    print("Popup window unavailable - using terminal input instead.\n")

    while True:
        path_input = input(
            "Enter path to image file (for batch analysis, separate "
            "multiple paths with commas): "
        ).strip()
        candidate_paths = [p.strip().strip('"\'') for p in path_input.split(',')]
        candidate_paths = [p for p in candidate_paths if p != '']
        missing = [p for p in candidate_paths if not os.path.isfile(p)]
        if candidate_paths and not missing:
            image_paths = candidate_paths
            break
        for p in missing:
            print(f"  File not found: '{p}'")
        print("  Please re-enter - try again.\n")

    while True:
        ref_input = input("Enter reference length in mm (e.g. 18): ").strip()
        try:
            reference_mm = float(ref_input)
            if reference_mm > 0:
                break
        except ValueError:
            pass
        print("  Please enter a positive number.\n")

    while True:
        color_input = input(
            "Enter calibration line's color [Red/Green/Blue] (default Red): "
        ).strip().upper()
        if color_input == "":
            color_mode = "RED"
            break
        elif color_input in ("RED", "GREEN", "BLUE"):
            color_mode = color_input
            break
        print("  Please enter Red, Green, or Blue.\n")

    thickness_input = input(
        "Enter aggregate thickness in mm, or press Enter to skip: "
    ).strip()
    thickness_mm = None
    if thickness_input != "":
        try:
            t = float(thickness_input)
            if t > 0:
                thickness_mm = t
            else:
                print("  Ignoring non-positive thickness - treating as not provided.")
        except ValueError:
            print("  Could not parse thickness - treating as not provided.")

    return {
        'image_paths': image_paths,
        'reference_mm': reference_mm,
        'color_mode': color_mode,
        'thickness_mm': thickness_mm,
    }


def get_parameters():
    try:
        return get_parameters_gui()
    except Exception as e:
        print(f"Popup window unavailable ({type(e).__name__}: {e})")
        return get_parameters_terminal()


# =============================================
# MAIN ANALYSIS PIPELINE
# =============================================

def analyze_image(image, reference_mm, color_mode, thickness_mm):
    """Run the full backlight analysis on a loaded BGR image. Returns a
    dict with everything display_results() needs, or a failure reason."""

    # ---- Calibration (HSV + ratio-based shape filter) ----
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation_min, value_min, s_ref, v_ref = compute_adaptive_hsv_floors(hsv)
    color_mask = get_color_mask(hsv, color_mode, saturation_min, value_min)
    line = find_calibration_line(color_mask)
    hsv_diagnostics = {
        'saturation_min': saturation_min, 'value_min': value_min,
        's_ref': s_ref, 'v_ref': v_ref,
    }

    if line is None:
        return {'success': False, 'reason': (
            f"No {color_mode.lower()} calibration line could be confirmed.\n"
            f"Possible causes: the line isn't in frame, lighting is too poor,\n"
            f"or the wrong color was selected for this photo.\n"
            f"(This photo's adaptive floors: saturation>={saturation_min:.0f} "
            f"[99th pct S={s_ref:.0f}], value>={value_min:.0f} [99th pct V={v_ref:.0f}])"
        ), 'hsv_diagnostics': hsv_diagnostics}

    line_px = line['length_px']
    mm_per_px = reference_mm / line_px
    mm2_per_px2 = mm_per_px ** 2

    # Calibration overlay for display
    cal_result = image.copy()
    overlay = cal_result.copy()
    overlay[color_mask] = DRAW_COLORS[color_mode]
    cv2.addWeighted(overlay, 0.4, cal_result, 0.6, 0, cal_result)
    box = np.int32(cv2.boxPoints(line['rect']))
    cv2.drawContours(cal_result, [box], 0, (0, 255, 255), 3)
    cx, cy = int(line['center'][0]), int(line['center'][1])
    cv2.putText(cal_result, f"{line_px:.0f}px = {reference_mm}mm",
                (max(cx - 150, 10), max(cy - 30, 30)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    # ---- Grayscale ----
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_h, img_w = gray.shape
    blur_size = max(3, make_odd(BLUR_SIZE_FRACTION * min(img_h, img_w)))
    morph_size = max(2, int(round(MORPH_SIZE_FRACTION * min(img_h, img_w))))

    # ---- Calibration line exclusion mask (computed early - used three times) ----
    # Ported from the reflected-light line, confirmed necessary here too by
    # testing: Otsu computed on the whole image (line included) gave wildly
    # wrong results on real backlit photos (200+ mm2 "aggregates" that were
    # actually the calibration line being misread) - the line's pixels were
    # skewing the threshold. Excluding it before computing Otsu fixes this.
    line_exclusion_mask = np.zeros(gray.shape, dtype=np.uint8)
    (lcx, lcy), (lw, lh), langle = line['rect']
    padded_rect = ((lcx, lcy), (lw + 20, lh + 20), langle)
    line_box = np.int32(cv2.boxPoints(padded_rect))
    cv2.fillPoly(line_exclusion_mask, [line_box], 255)

    # ---- Illumination flattening + blur ----
    # Confirmed by testing on a real voltage-sweep photo: the calibration
    # bar used on this setup is a solid, sizeable object (not a thin line),
    # large enough for its own darkness to reach into the illumination
    # estimate's blur radius (half the image's smaller dimension) and drag
    # the "expected background" down over a wide halo around it. That
    # contamination alone was enough to shift Otsu's threshold from ~96
    # (background vs. bar+sample, correct) to ~143, which misread ~90% of
    # the frame as foreground. Passing the same exclusion mask in here so
    # the bar can't skew the background estimate for the rest of the frame.
    flattened = flatten_illumination(gray, line_exclusion_mask)
    blurred = cv2.GaussianBlur(flattened, (blur_size, blur_size), 0)

    # ---- Otsu threshold, computed without the calibration line's pixels ----
    # THRESH_BINARY_INV (not THRESH_BINARY): backlight aggregate is DARK on
    # a bright background, so the dark side of the cut is foreground here.
    otsu_threshold, _ = cv2.threshold(
        blurred[line_exclusion_mask == 0], 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    _, binary = cv2.threshold(blurred, otsu_threshold, 255, cv2.THRESH_BINARY_INV)
    raw_foreground_fraction = float(np.count_nonzero(binary)) / (img_h * img_w)

    # ---- Noise removal ----
    kernel = np.ones((morph_size, morph_size), np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    # ---- Exclude the calibration line's own region from the final result ----
    # Ported from the reflected-light line: a fixed padding box undershot
    # the line's actual brightness footprint (translucent ink can have a
    # halo beyond its color-detected core). Whatever connected blob in the
    # actual thresholded result touches the line's known center gets
    # excluded entirely, however large its real footprint turns out to be.
    line_center_point = (int(lcx), int(lcy))
    pre_exclusion_contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in pre_exclusion_contours:
        if cv2.pointPolygonTest(c, line_center_point, False) >= 0:
            cv2.drawContours(cleaned, [c], -1, 0, -1)
    cleaned[line_exclusion_mask == 255] = 0  # geometric box too, as a floor

    # ---- Core contours + holes (the trusted, flattened-Otsu anchor - unchanged) ----
    minimum_area_px = MINIMUM_AREA_MM2 / mm2_per_px2
    confident_area_px = CONFIDENT_AREA_MM2 / mm2_per_px2
    borderline_area_px = BORDERLINE_AREA_MM2 / mm2_per_px2
    border_artifact_area_px = BORDER_ARTIFACT_AREA_FRACTION * img_h * img_w
    foreground_fraction = float(np.count_nonzero(cleaned)) / (img_h * img_w)

    (core_contours, core_hierarchy, core_outer_contours,
     core_hole_contours, core_small_indices, rejected) = classify_contours(
        cleaned, minimum_area_px, confident_area_px, borderline_area_px,
        border_artifact_area_px, mm2_per_px2, collect_diagnostics=True)

    detection_diagnostics = {
        'raw_foreground_fraction': raw_foreground_fraction,
        'foreground_fraction': foreground_fraction,
        'rejected_top5': rejected[:5],
    }

    # ---- Hysteresis edge recovery ----
    # PORTED from the reflected-light line (V4.4), adapted for backlight's
    # inverted polarity - see B1.5 changelog. seed_mask keeps only the
    # qualifying core blobs from `cleaned` (border artifacts and below-floor
    # noise erased); holes are protected separately below so growth never
    # fills a real hole back in.
    qualifying_core_idx = {i for _, i in core_outer_contours}
    seed_mask = cleaned.copy()
    if core_hierarchy is not None:
        for i, contour in enumerate(core_contours):
            parent_idx = core_hierarchy[0][i][3]
            if parent_idx == -1 and i not in qualifying_core_idx:
                cv2.drawContours(seed_mask, [contour], -1, 0, -1)

    core_hole_mask = np.zeros(gray.shape, dtype=np.uint8)
    for hole_contour, parent_idx in core_hole_contours:
        if parent_idx in qualifying_core_idx:
            cv2.drawContours(core_hole_mask, [hole_contour], -1, 255, -1)

    # LOOSE threshold on the RAW (unflattened) grayscale, derived from this
    # photo's own background statistics (core regions + calibration bar
    # excluded). Backlight aggregate is dark on bright background, so the
    # loose test looks for pixels moderately DARKER than background
    # (mean - N*std) - the mirror image of V4's "brighter than background"
    # test for its bright-on-dark case.
    raw_background_pixels = gray[(seed_mask == 0) & (line_exclusion_mask == 0)]
    if raw_background_pixels.size > 0:
        raw_background_mean = float(raw_background_pixels.mean())
        raw_background_std = float(raw_background_pixels.std())
    else:
        raw_background_mean, raw_background_std = float(gray.mean()), float(gray.std())
    loose_threshold = raw_background_mean - LOOSE_THRESHOLD_STD_MULTIPLIER * raw_background_std
    print(f"  Loose threshold (hysteresis, raw, darker-than-background): {loose_threshold:.1f}  "
          f"(background mean {raw_background_mean:.1f} - {LOOSE_THRESHOLD_STD_MULTIPLIER} "
          f"x std {raw_background_std:.1f})")

    loose_mask = np.where(gray < loose_threshold, 255, 0).astype(np.uint8)
    loose_mask[line_exclusion_mask == 255] = 0  # never grow into the calibration bar
    loose_mask[core_hole_mask == 255] = 0       # never grow into a real hole

    # ---- Growth distance cap: stops runaway spread AND stops two separate ----
    # confirmed aggregates from bridging together through contested ground
    # (no-man's-land) - see build_growth_distance_cap() docstring.
    growth_allowed_mask = build_growth_distance_cap(seed_mask)
    loose_mask[growth_allowed_mask == 0] = 0

    # Standard hysteresis/connected-component reconstruction: keep only the
    # connected components of (seed | loose) that actually touch a seed
    # pixel, so isolated dust/noise passing the loose threshold on its own
    # is never included.
    union_mask = cv2.bitwise_or(seed_mask, loose_mask)
    num_labels, labels = cv2.connectedComponents(union_mask, connectivity=8)
    seed_labels = set(np.unique(labels[seed_mask == 255])) - {0}
    if seed_labels:
        grown_mask = np.where(np.isin(labels, list(seed_labels)), 255, 0).astype(np.uint8)
    else:
        grown_mask = seed_mask.copy()

    # Re-apply the calibration-bar exclusion after growth, so it can never
    # be grown into via the raw/loose threshold path - the fix just
    # validated in B1.4 must still hold with hysteresis on top.
    grown_mask[line_exclusion_mask == 255] = 0
    post_growth_contours, _ = cv2.findContours(grown_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in post_growth_contours:
        if cv2.pointPolygonTest(c, line_center_point, False) >= 0:
            cv2.drawContours(grown_mask, [c], -1, 0, -1)

    # ---- Final contours + holes (core, possibly grown) ----
    (contours, hierarchy, outer_contours,
     hole_contours, small_indices, _) = classify_contours(
        grown_mask, minimum_area_px, confident_area_px, borderline_area_px,
        border_artifact_area_px, mm2_per_px2, collect_diagnostics=False)

    # ---- Background mean intensity (the bright, unobstructed transmission) ----
    full_mask = np.zeros(gray.shape, dtype=np.uint8)
    for contour, _ in outer_contours:
        cv2.drawContours(full_mask, [contour], -1, 255, -1)
    background_pixels = gray[full_mask == 0]
    background_mean = float(background_pixels.mean()) if background_pixels.size > 0 else 0.0

    # ---- Per-aggregate measurements ----
    measurements = []
    for i, (contour, contour_idx) in enumerate(outer_contours):
        outer_area_px = cv2.contourArea(contour)
        my_holes = [hc for hc, pidx in hole_contours if pidx == contour_idx]
        hole_area_px = sum(cv2.contourArea(hc) for hc in my_holes)
        true_area_px = outer_area_px - hole_area_px
        true_area_mm2 = true_area_px * mm2_per_px2

        perimeter_px = cv2.arcLength(contour, True)
        perimeter_mm = perimeter_px * mm_per_px
        circularity = (4 * np.pi * true_area_px / (perimeter_px ** 2)
                       if perimeter_px > 0 else 0)

        x, y, w, h = cv2.boundingRect(contour)
        M = cv2.moments(contour)
        center_x = int(M['m10'] / M['m00']) if M['m00'] > 0 else x
        center_y = int(M['m01'] / M['m00']) if M['m00'] > 0 else y

        # true-area mask for this aggregate (outer minus its own holes)
        agg_mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(agg_mask, [contour], -1, 255, -1)
        for hc in my_holes:
            cv2.drawContours(agg_mask, [hc], -1, 0, -1)
        agg_pixels = gray[agg_mask == 255]
        aggregate_mean = float(agg_pixels.mean()) if agg_pixels.size > 0 else 0.0

        # Backlight intensity index: optical-density-style transmittance.
        # relative_transmittance = fraction of light getting through the
        # aggregate compared to the open background; clamped to (0, 1] so
        # noise can't push it past physically sensible bounds.
        safe_background = max(background_mean, 1e-6)
        relative_transmittance = min(1.0, max(1e-6, aggregate_mean / safe_background))
        optical_density_index = -np.log10(relative_transmittance)
        combined_optical_density = true_area_mm2 * optical_density_index

        volume_mm3 = true_area_mm2 * thickness_mm if thickness_mm is not None else None

        # Hysteresis diagnostics: how much of this aggregate's area is the
        # confident core vs. grown-in via the loose raw-image threshold.
        core_area_px_count = int(np.count_nonzero((agg_mask == 255) & (seed_mask == 255)))
        hysteresis_area_px_count = int(np.count_nonzero((agg_mask == 255) & (seed_mask == 0)))
        core_area_mm2 = core_area_px_count * mm2_per_px2
        hysteresis_area_mm2 = hysteresis_area_px_count * mm2_per_px2

        # Overgrowth flag: don't let a suspiciously large hysteresis result
        # look identical to a correctly-recovered one - surface it instead.
        is_overgrown = hysteresis_area_mm2 > OVERGROWTH_RATIO * max(core_area_mm2, 1e-9)
        overgrowth_ratio = hysteresis_area_mm2 / max(core_area_mm2, 1e-9)
        if is_overgrown:
            print(f"  WARNING: aggregate #{i + 1} hysteresis-recovered area "
                  f"({hysteresis_area_mm2:.4f}mm2) exceeds {OVERGROWTH_RATIO * 100:.0f}% "
                  f"of its core area ({core_area_mm2:.4f}mm2) - flagged as suspicious growth "
                  f"(recovered/core = {overgrowth_ratio:.2f}x)")

        # B1.9 reporting addition: core/recovered breakdown for the visible
        # tables (B1.5 already computed core_area_mm2/hysteresis_area_mm2,
        # but only ever printed them to console). core_area_mm2 and
        # hysteresis_area_mm2 above are both raster PIXEL COUNTS (partition
        # of agg_mask by seed_mask), while true_area_mm2 is a cv2.contourArea
        # (Green's-theorem polygon) measurement - the same two quantities
        # used everywhere else in this file for area floors/classification,
        # so true_area_mm2 must stay the authoritative "Area" column.
        # Pixel-count area and contour-polygon area are not identical measures
        # (they differ by boundary-pixel effects, most visible on small
        # aggregates near the area floor - exactly the case this change was
        # requested for) so core_area_mm2 + hysteresis_area_mm2 is NOT
        # guaranteed to equal true_area_mm2 exactly. Checked, not assumed:
        # display_core/display_recovered below are DERIVED so they always
        # reconcile to true_area_mm2 by construction (display_recovered =
        # Area - display_core), instead of showing two independently-measured
        # numbers that could silently fail to add up in the table a user is
        # trusting at a glance. is_overgrown/overgrowth_ratio above are left
        # driven by the original raw pixel-count hysteresis_area_mm2 -
        # unchanged detection behavior, per spec.
        display_core_mm2 = min(core_area_mm2, true_area_mm2)
        display_recovered_mm2 = true_area_mm2 - display_core_mm2
        if core_area_mm2 > true_area_mm2 + 1e-6:
            print(f"  NOTE: aggregate #{i + 1} rasterized core pixel-count area "
                  f"({core_area_mm2:.4f}mm2) exceeds its contour-based total area "
                  f"({true_area_mm2:.4f}mm2) by {core_area_mm2 - true_area_mm2:.4f}mm2 - "
                  f"a boundary-rasterization discrepancy between the pixel-count and "
                  f"contour-polygon area measures, most visible on tiny aggregates. "
                  f"Table's Core/Recovered columns are clamped to reconcile exactly "
                  f"with the Area column; is_overgrown/ratio above still use the raw "
                  f"pixel-count numbers, unaffected by this display-only clamp.")

        measurements.append({
            'id': i + 1,
            'contour_idx': contour_idx,
            'true_area_px': true_area_px,
            'true_area_mm2': true_area_mm2,
            'core_area_mm2': core_area_mm2,
            'hysteresis_area_mm2': hysteresis_area_mm2,
            'display_core_mm2': display_core_mm2,
            'display_recovered_mm2': display_recovered_mm2,
            'overgrowth_ratio': overgrowth_ratio,
            'num_holes': len(my_holes),
            'perimeter_mm': perimeter_mm,
            'circularity': circularity,
            'is_small': contour_idx in small_indices,
            'is_overgrown': is_overgrown,
            'center_x': center_x, 'center_y': center_y,
            'bbox_x': x, 'bbox_y': y, 'bbox_w': w, 'bbox_h': h,
            'background_mean': background_mean,
            'aggregate_mean': aggregate_mean,
            'relative_transmittance': relative_transmittance,
            'optical_density_index': optical_density_index,
            'combined_optical_density': combined_optical_density,
            'volume_mm3': volume_mm3,
        })

    # ---- Clean result image (no text block baked into it) ----
    # Small aggregates (below CONFIDENT_AREA_MM2) are drawn in orange instead
    # of green, flagging them as "smaller, worth a second look" in the image
    # itself, not just as a number in a table.
    result_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # B1.6 render fix: the recovered (hysteresis-only) ring mask has TWO
    # boundaries - an outer edge (the new, grown-out extent) and an inner
    # edge (the interface with the original core). RETR_EXTERNAL only ever
    # returns the outer one, which - since the ring sits directly against
    # the core - is geometrically the SAME path as the green core/total
    # contour drawn below. Drawing order alone can't fix that (see B1.6
    # changelog): whichever line is drawn second simply overwrites the
    # first along a shared path, and any cyan that survived was just
    # subpixel rasterization jitter between two independently-approximated
    # contours, not a real distinguishing feature - which is why it was
    # visible at 3.0V (4211 cyan px) and essentially invisible at 3.2V (27
    # cyan px) despite 3.2V having MORE recovered area (0.375mm2 vs
    # 0.343mm2), confirmed by direct pixel-count measurement, not contour
    # rendering. Fix: run findContours with RETR_CCOMP (not RETR_EXTERNAL)
    # on the ring-only mask so its inner edge is also returned as a hole
    # contour - drawContours(-1, ...) draws both levels. That inner edge
    # sits at the true core/recovered interface and is NEVER part of the
    # green contour's path, so it survives regardless of draw order. Green
    # is now drawn first and cyan last purely so the ring's outer edge is
    # also fully visible on top rather than fighting for the same pixels -
    # this reordering alone was NOT sufficient by itself (that's option 2
    # from the fix plan, evaluated and rejected) without the RETR_CCOMP
    # change, since the outer edges still coincide.
    final_mask = np.zeros(gray.shape, dtype=np.uint8)
    for contour, _ in outer_contours:
        cv2.drawContours(final_mask, [contour], -1, 255, -1)
    for hc, _ in hole_contours:
        cv2.drawContours(final_mask, [hc], -1, 0, -1)
    hysteresis_mask = np.where((final_mask == 255) & (seed_mask == 0), 255, 0).astype(np.uint8)

    # Cyan fill drawn first (solid, from the pixel mask) so the green core
    # outline drawn after it stays on top and remains visible at the
    # core/recovered boundary, instead of green painting over cyan.
    if np.any(hysteresis_mask):
        result_image[hysteresis_mask == 255] = HYSTERESIS_COLOR_BGR

    overgrown_indices = {m['contour_idx'] for m in measurements if m['is_overgrown']}
    confident_contours = [c for c, i in outer_contours if i not in small_indices and i not in overgrown_indices]
    small_contours = [c for c, i in outer_contours if i in small_indices and i not in overgrown_indices]
    overgrown_contours = [c for c, i in outer_contours if i in overgrown_indices]
    cv2.drawContours(result_image, confident_contours, -1, (0, 255, 0), 2)
    cv2.drawContours(result_image, small_contours, -1, (0, 165, 255), 2)
    cv2.drawContours(result_image, overgrown_contours, -1, OVERGROWN_COLOR_BGR, 4)
    cv2.drawContours(result_image, [hc for hc, _ in hole_contours], -1, (0, 0, 255), 2)
    for m in measurements:
        if m['is_overgrown']:
            dot_color = OVERGROWN_COLOR_BGR
            label = f"#{m['id']}!"
        elif m['is_small']:
            dot_color = (0, 165, 255)
            label = f"#{m['id']}*"
        else:
            dot_color = (0, 255, 255)
            label = f"#{m['id']}"
        cv2.circle(result_image, (m['center_x'], m['center_y']), 6, dot_color, -1)
        cv2.putText(result_image, label, (m['bbox_x'], m['bbox_y'] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, dot_color, 2)

    return {
        'success': True,
        'image': image, 'cal_result': cal_result, 'result_image': result_image,
        'color_mode': color_mode, 'reference_mm': reference_mm,
        'line_px': line_px, 'mm_per_px': mm_per_px,
        'otsu_threshold': otsu_threshold,
        'loose_threshold': loose_threshold,
        'raw_background_mean': raw_background_mean,
        'raw_background_std': raw_background_std,
        'measurements': measurements,
        'background_mean': background_mean,
        'thickness_mm': thickness_mm,
        'hsv_diagnostics': hsv_diagnostics,
        'detection_diagnostics': detection_diagnostics,
    }


# =============================================
# DISPLAY — single window, 2x2 grid
# =============================================

def compute_photo_summary(r):
    """Per-photo totals. Factored out so the per-photo info table and the
    batch comparison table compute these the same way and never disagree."""
    measurements = r['measurements']
    if not measurements:
        return None
    total_volume = (sum(m['volume_mm3'] for m in measurements)
                     if r['thickness_mm'] is not None else None)
    overgrown = [m for m in measurements if m['is_overgrown']]
    return {
        'count': len(measurements),
        'total_area': sum(m['true_area_mm2'] for m in measurements),
        'total_core': sum(m['display_core_mm2'] for m in measurements),
        'total_recovered': sum(m['display_recovered_mm2'] for m in measurements),
        'total_holes': sum(m['num_holes'] for m in measurements),
        'avg_circularity': sum(m['circularity'] for m in measurements) / len(measurements),
        'avg_optical_density': sum(m['optical_density_index'] for m in measurements) / len(measurements),
        'total_combined': sum(m['combined_optical_density'] for m in measurements),
        'total_volume': total_volume,
        'has_small': any(m['is_small'] for m in measurements),
        'has_overgrown': len(overgrown) > 0,
        'max_overgrowth_ratio': max((m['overgrowth_ratio'] for m in overgrown), default=None),
    }


def display_results(r, filename=None):
    measurements = r['measurements']
    name_part = f"{filename}  |  " if filename else ""

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(
        f"Plasma Aggregation Analysis (Backlight) v{VERSION} [{BUILD_TAG}]  |  {name_part}"
        f"{r['color_mode'].capitalize()} line: {r['line_px']:.0f}px = {r['reference_mm']}mm  |  "
        f"Otsu threshold: {r['otsu_threshold']:.0f}",
        fontsize=11, fontweight='bold'
    )

    axes[0, 0].imshow(cv2.cvtColor(r['image'], cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(cv2.cvtColor(r['cal_result'], cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('Calibration')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(cv2.cvtColor(r['result_image'], cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title(f"Result ({len(measurements)} aggregate(s))")
    axes[1, 0].axis('off')

    axes[1, 1].axis('off')
    if measurements:
        summary = compute_photo_summary(r)
        axes[1, 1].text(
            0.02, 0.97, f"Background mean: {r['background_mean']:.1f}",
            transform=axes[1, 1].transAxes, fontsize=9, verticalalignment='top'
        )

        any_small = any(m['is_small'] for m in measurements)
        any_overgrown = any(m['is_overgrown'] for m in measurements)
        col_labels = ['#', 'Scale\n(mm/px)', 'Area\n(mm2)', 'Core area\n(mm2)',
                      'Recovered area\n(mm2)', 'Holes\n(count)',
                      'Circularity\n(unitless)', 'Optical density\nindex (unitless)',
                      'Combined optical\ndensity (mm2)', 'Volume\n(mm3)']
        RECOVERED_COL = 4
        scale_str = f"{r['mm_per_px']:.5f}"
        cell_text = []
        for m in measurements:
            vol_str = f"{m['volume_mm3']:.3f}" if m['volume_mm3'] is not None else "TBD"
            if m['is_overgrown']:
                id_str = f"{m['id']}! ({m['overgrowth_ratio']:.1f}x)"
            elif m['is_small']:
                id_str = f"{m['id']}*"
            else:
                id_str = f"{m['id']}"
            cell_text.append([
                id_str, scale_str, f"{m['true_area_mm2']:.3f}",
                f"{m['display_core_mm2']:.3f}", f"{m['display_recovered_mm2']:.3f}",
                f"{m['num_holes']}",
                f"{m['circularity']:.3f}", f"{m['optical_density_index']:.3f}",
                f"{m['combined_optical_density']:.3f}", vol_str,
            ])
        total_vol_str = f"{summary['total_volume']:.3f}" if summary['total_volume'] is not None else "TBD"
        cell_text.append([
            'TOTAL', scale_str, f"{summary['total_area']:.3f}",
            f"{summary['total_core']:.3f}", f"{summary['total_recovered']:.3f}",
            f"{summary['total_holes']}",
            f"{summary['avg_circularity']:.3f}", f"{summary['avg_optical_density']:.3f}",
            f"{summary['total_combined']:.3f}", total_vol_str,
        ])

        table = axes[1, 1].table(cellText=cell_text, colLabels=col_labels,
                                  loc='lower center', cellLoc='center',
                                  bbox=[0, 0, 1, 0.82])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.auto_set_column_width(col=list(range(len(col_labels))))
        for (row, col), cell in table.get_celld().items():
            if row == 0 or row == len(cell_text):
                cell.set_text_props(fontweight='bold')
        # Overgrowth flag (req. 2): make the Recovered area cell visually
        # distinct - same red used for the "!" marker/outline elsewhere -
        # for any row whose aggregate tripped the overgrowth flag.
        for row_idx, m in enumerate(measurements, start=1):
            if m['is_overgrown']:
                table[row_idx, RECOVERED_COL].set_text_props(fontweight='bold', color='red')
        caption_lines = []
        if any_small:
            caption_lines.append(f"* below {CONFIDENT_AREA_MM2}mm2 (shown in orange in the result image) - worth a visual check")
        if any_overgrown:
            caption_lines.append(f"! overgrowth flag: recovered area exceeds {OVERGROWTH_RATIO * 100:.0f}% of core area (red outline in the result image) - ratio shown in parentheses")
        if caption_lines:
            axes[1, 1].text(
                0.02, 0.88, "\n".join(caption_lines),
                transform=axes[1, 1].transAxes, fontsize=8, style='italic', verticalalignment='top'
            )
    else:
        axes[1, 1].text(0.02, 0.9, "No aggregates detected.\nTry a different photo or check lighting.",
                         transform=axes[1, 1].transAxes, fontsize=10, verticalalignment='top')

    plt.tight_layout()
    plt.show()


def display_failure(reason, filename=None):
    """One simple window for a photo that failed calibration, so a batch
    still gets exactly one window per photo even when that photo can't
    be analyzed."""
    name_part = f"{filename}\n\n" if filename else ""
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.suptitle("Calibration failed", fontsize=12, fontweight='bold')
    ax.axis('off')
    ax.text(0.5, 0.5, f"{name_part}{reason}", ha='center', va='center', fontsize=11)
    plt.tight_layout()
    plt.show()


def display_comparison_table(results, filenames):
    """The +1 extra window shown only when more than one photo was
    analyzed - one row per photo plus an avg/std summary row. Reuses
    compute_photo_summary() so these numbers always match what each
    photo's own result window shows."""
    col_labels = ['Photo', 'Scale\n(mm/px)', 'Area\n(mm2)', 'Core area\n(mm2)',
                  'Recovered area\n(mm2)', 'Holes\n(count)',
                  'Circularity\n(unitless)', 'Optical density\nindex (unitless)',
                  'Combined optical\ndensity (mm2)', 'Volume\n(mm3)']
    RECOVERED_COL = 4
    cell_text = []
    numeric_rows = []
    scales = []
    overgrown_row_indices = []  # 1-based data row positions to color red

    for name, r in zip(filenames, results):
        if not r['success']:
            cell_text.append([name, 'FAILED', '-', '-', '-', '-', '-', '-', '-', '-'])
            continue

        scale_str = f"{r['mm_per_px']:.5f}"
        scales.append(r['mm_per_px'])

        summary = compute_photo_summary(r)
        if summary is None:
            cell_text.append([name, scale_str, '0', '0', '0', '0', '-', '-', '-', '-'])
            continue
        vol_str = f"{summary['total_volume']:.3f}" if summary['total_volume'] is not None else "TBD"
        name_str = name
        if summary['has_overgrown']:
            name_str = f"{name} !({summary['max_overgrowth_ratio']:.1f}x)"
            overgrown_row_indices.append(len(cell_text) + 1)  # +1: header is row 0
        elif summary['has_small']:
            name_str = f"{name} *"
        cell_text.append([
            name_str, scale_str, f"{summary['total_area']:.3f}",
            f"{summary['total_core']:.3f}", f"{summary['total_recovered']:.3f}",
            f"{summary['total_holes']}",
            f"{summary['avg_circularity']:.3f}", f"{summary['avg_optical_density']:.3f}",
            f"{summary['total_combined']:.3f}", vol_str,
        ])
        numeric_rows.append(summary)

    if numeric_rows or scales:
        areas = [s['total_area'] for s in numeric_rows]
        cores = [s['total_core'] for s in numeric_rows]
        recovered = [s['total_recovered'] for s in numeric_rows]
        holes = [s['total_holes'] for s in numeric_rows]
        circs = [s['avg_circularity'] for s in numeric_rows]
        odens = [s['avg_optical_density'] for s in numeric_rows]
        combined = [s['total_combined'] for s in numeric_rows]
        vols = [s['total_volume'] for s in numeric_rows if s['total_volume'] is not None]

        def stat_or_dash(fn, vals, nd=3):
            return f"{fn(vals):.{nd}f}" if vals else '-'

        cell_text.append([
            'AVG', stat_or_dash(np.mean, scales, 5), stat_or_dash(np.mean, areas),
            stat_or_dash(np.mean, cores), stat_or_dash(np.mean, recovered),
            stat_or_dash(np.mean, holes, 1), stat_or_dash(np.mean, circs),
            stat_or_dash(np.mean, odens), stat_or_dash(np.mean, combined),
            stat_or_dash(np.mean, vols) if vols else "TBD",
        ])
        cell_text.append([
            'STD', stat_or_dash(np.std, scales, 5), stat_or_dash(np.std, areas),
            stat_or_dash(np.std, cores), stat_or_dash(np.std, recovered),
            stat_or_dash(np.std, holes, 1), stat_or_dash(np.std, circs),
            stat_or_dash(np.std, odens), stat_or_dash(np.std, combined),
            stat_or_dash(np.std, vols) if vols else "TBD",
        ])

    fig, ax = plt.subplots(figsize=(max(11, 1.5 * len(filenames) + 5), 1.5 + 0.4 * len(cell_text)))
    fig.suptitle(f"Batch comparison (Backlight) - {len(filenames)} photo(s)", fontsize=12, fontweight='bold')
    ax.axis('off')

    table = ax.table(cellText=cell_text, colLabels=col_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.auto_set_column_width(col=list(range(len(col_labels))))
    table.scale(1, 1.8)
    last_data_row = len(cell_text)
    summary_rows_start = last_data_row - 1 if (numeric_rows or scales) else last_data_row + 1
    for (row, col), cell in table.get_celld().items():
        if row == 0 or row >= summary_rows_start:
            cell.set_text_props(fontweight='bold')
    # Overgrowth flag (req. 2): make the Recovered area cell visually
    # distinct for any photo with a flagged aggregate, same as the
    # per-photo table.
    for row_idx in overgrown_row_indices:
        table[row_idx, RECOVERED_COL].set_text_props(fontweight='bold', color='red')

    if any(s.get('has_small') for s in numeric_rows) or any(s.get('has_overgrown') for s in numeric_rows):
        caption_lines = []
        if any(s.get('has_small') for s in numeric_rows):
            caption_lines.append(f"* includes an aggregate below {CONFIDENT_AREA_MM2}mm2 - worth a visual check in that photo's result window")
        if any(s.get('has_overgrown') for s in numeric_rows):
            caption_lines.append("! includes an overgrowth-flagged aggregate - ratio shown is that photo's worst (max) recovered/core ratio; see per-photo window for the full breakdown")
        ax.text(0.02, 0.02, "\n".join(caption_lines),
                transform=ax.transAxes, fontsize=8, style='italic')

    plt.tight_layout()
    plt.show()


# =============================================
# MAIN
# =============================================

if __name__ == "__main__":
    print("=" * 60)
    print(f"PLASMA AGGREGATION ANALYSIS (BACKLIGHT)  v{VERSION}  [build: {BUILD_TAG}]")
    print(f"Counting floor: {MINIMUM_AREA_MM2}mm2  |  Confident floor (no '*' mark): {CONFIDENT_AREA_MM2}mm2")
    print(f"Master CSV log: {MASTER_CSV_PATH}")
    print("=" * 60)

    params = get_parameters()
    image_paths = params['image_paths']

    print(f"\n{len(image_paths)} image(s) selected")
    print(f"Reference length: {params['reference_mm']} mm")
    print(f"Calibration line color: {params['color_mode'].capitalize()}")
    if params['thickness_mm'] is not None:
        print(f"Aggregate thickness: {params['thickness_mm']} mm")
    else:
        print("Aggregate thickness: not provided (volume will show as TBD)")

    all_results = []
    filenames = []

    for path in image_paths:
        name = os.path.basename(path)
        filenames.append(name)
        print(f"\n--- {name} ---")

        image = cv2.imread(path)
        if image is None:
            print(f"ERROR: could not read image at '{path}' - skipping")
            result = {'success': False, 'reason': f"Could not read image file:\n{path}"}
            all_results.append(result)
            append_photo_to_master_csv(name, params, result)
            print(f"Logged to master CSV: {name}")
            display_failure(result['reason'], name)
            continue

        result = analyze_image(image, params['reference_mm'], params['color_mode'], params['thickness_mm'])
        all_results.append(result)

        if not result['success']:
            print(f"CALIBRATION FAILED: {result['reason']}")
            append_photo_to_master_csv(name, params, result)
            print(f"Logged to master CSV: {name}")
            display_failure(result['reason'], name)
            continue

        print(f"Calibration: {result['line_px']:.0f}px = {params['reference_mm']}mm "
              f"(1px = {result['mm_per_px']:.5f}mm)")
        print(f"Otsu threshold (core, flattened): {result['otsu_threshold']:.0f}")
        print(f"Loose threshold (hysteresis, raw, darker-than-background): {result['loose_threshold']:.0f}")
        hd = result['hsv_diagnostics']
        print(f"Adaptive HSV floors used: saturation>={hd['saturation_min']:.0f} "
              f"[99th pct S={hd['s_ref']:.0f}], value>={hd['value_min']:.0f} "
              f"[99th pct V={hd['v_ref']:.0f}]  <- compare across photos to check lighting robustness")
        print(f"Aggregates found: {len(result['measurements'])}")
        for m in result['measurements']:
            print(f"  #{m['id']}: core={m['core_area_mm2']:.4f}mm2  "
                  f"hysteresis-recovered={m['hysteresis_area_mm2']:.4f}mm2  "
                  f"total={m['true_area_mm2']:.4f}mm2")
        dd = result['detection_diagnostics']
        print(f"Otsu foreground coverage: {dd['raw_foreground_fraction']*100:.2f}% of frame RAW "
              f"(before morphology) -> {dd['foreground_fraction']*100:.2f}% AFTER morphology")
        if dd['rejected_top5']:
            print("Largest rejected candidate blob(s) (why they didn't count):")
            for area_mm2, reason in dd['rejected_top5']:
                print(f"    {area_mm2:.3f}mm2 - {reason}")
        elif len(result['measurements']) == 0:
            if dd['raw_foreground_fraction'] > 0:
                print("    Otsu DID flag some dark pixels, but morphology ate all of them - "
                      "the dark content found was thin/fine (e.g. traced outline or bubble rims), "
                      "not a solid blob. The real aggregate may have too little contrast to compete "
                      "with those thin dark marks for Otsu's threshold placement.")
            else:
                print("    Otsu found literally 0% dark pixels anywhere in the frame - "
                      "likely a real low-contrast issue in the photo, not a filtering issue.")

        display_results(result, filename=name)
        append_photo_to_master_csv(name, params, result)
        print(f"Logged to master CSV: {name}")

    if len(image_paths) > 1:
        print("\n" + "=" * 60)
        print("BATCH COMPARISON")
        print("=" * 60)
        display_comparison_table(all_results, filenames)

    print("\nAnalysis complete!")