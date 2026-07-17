import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import csv
from datetime import datetime

VERSION = "4.14"
BUILD_TAG = "cluster-flagging-redesign-plus-perf-and-area-fix"

# =============================================
# CHANGELOG (quick reference)
# =============================================
# NOTE: this file forked from V4_.py after V4.4 to add hysteresis edge
# recovery + the CSV master log (hence starting at 4.5); those earlier
# entries were never backfilled into this file's own changelog, so this
# is the first entry actually kept here.
#
# V4.6 - CHANGED: MINIMUM_AREA_MM2 lowered from 0.4 to 0.2 (lower-area-
#        floor), MIRRORING the same change made to B1 (backlight line).
#        WHY: a real aggregate was measured at ~0.24mm2, ~5 sigma below
#        background, consistently across 4 independent voltage captures
#        of the same sample - not a single-photo guess. The area floor
#        is a real-world physical-size cutoff shared by design between
#        this line and B1's (see B1.1's changelog), not something
#        specific to backlight vs reflected-light physics, so it's kept
#        in sync here even though the confirming photos were backlight
#        shots. CONFIDENT_AREA_MM2 stays at 1.0, so anything from 0.2 to
#        1.0mm2 still gets the existing "*" flag / orange outline -
#        same mechanism as before, just extended slightly lower, never
#        auto-counted as unambiguous.
#        ACCEPTED TRADEOFF, not hidden: more small dust specks may now
#        also cross this lower floor and get counted-with-a-flag.
#        NOT INDEPENDENTLY TESTED against a real reflected-light photo
#        with this specific change - the 4 confirming captures were all
#        backlight (B1) photos. Watch for this the next time a real
#        reflected-light batch is run.
#
# V4.7 - VISIBILITY IMPROVEMENT, MIRRORING B1.9. No detection/threshold/
#        growth/area logic touched (OVERGROWTH_RATIO, hysteresis growth,
#        and the area floors are all unchanged). PROMPTED BY V4.6/B1.8:
#        lowering the area floor to 0.2mm2 means small aggregates now
#        routinely sit close to the size where hysteresis-recovered edge
#        pixels are a meaningful fraction of the reported total, but the
#        core-vs-recovered breakdown (core_area_mm2 / hysteresis_area_mm2,
#        computed since V4.5/B1.5) only ever reached console/debug output -
#        no way to sanity-check a small, overgrowth-flagged result at a
#        glance without reading the terminal log alongside the table.
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
# V4.8 - CHANGED: MINIMUM_AREA_MM2 lowered from 0.2 to 0.14 (lower-area-
#        floor-2), MIRRORING the same change made to B1.10 (backlight
#        line), for the same reason V4.6 mirrored B1.8: the area floor is
#        a shared real-world physical-size cutoff, not something specific
#        to backlight vs reflected-light physics (see V4.6's changelog).
#        WHY: B1.9's per-candidate diagnostic logging was used to directly
#        measure a specific below-left blob next to the main wispy
#        aggregate in all 4 voltage shots (Acetic+BSA_3.8pH, 2.9-3.2V).
#        Measured area: 0.1774mm2 (2.9V), 0.1626mm2 (3.0V), 0.1627mm2
#        (3.1V), 0.1459mm2 (3.2V) - a real, physically-measured target, not
#        an estimate, at the same relative pixel location in every shot.
#        The floor is set to 0.14mm2, below all 4 measurements with
#        margin. CONFIDENT_AREA_MM2 stays at 1.0, so this blob still gets
#        the existing "*" flag / orange outline - counted, but never
#        presented as an unambiguous, no-need-to-double-check size.
#        NOT INDEPENDENTLY TESTED against a real reflected-light photo with
#        this specific change - same caveat as V4.6, the confirming
#        captures were all backlight (B1) photos. Watch for this the next
#        time a real reflected-light batch is run.
#
# V4.9 - TEMPORARY DEBUG ADDITION, PORTED from B1.py's B1.3 diagnostics. No
#        detection/threshold/growth/area logic touched - this only adds
#        visibility, motivated by real aggregates (confirmed real by eye in
#        the photo) being reported as 0 contours with no way to tell why.
#        classify_contours() now accepts collect_diagnostics (replacing the
#        old log_below_floor flag) and returns a "why was this rejected"
#        list (border-artifact vs. below-floor) from the core pass, same as
#        B1. analyze_image() now also measures raw Otsu foreground coverage
#        right after thresholding (before morphology), separately from the
#        existing post-morphology coverage, so a "0 aggregates" result can
#        be told apart as: Otsu found nothing at all vs. Otsu found
#        something morphology then erased vs. something survived to the
#        rejected-candidates list. Printed to console per photo, alongside
#        the largest rejected candidate blob(s) and their rejection reason.
#        Intended to be removed or reworked once the missing-aggregate bug
#        is diagnosed - not a permanent feature.
#
# V4.10 - TWO CHANGES, kept clearly separate:
#
#         (1) REAL, VALIDATED CHANGE: MINIMUM_AREA_MM2 lowered from 0.14 to
#         0.10 (lower-area-floor-3). CLOSES A GAP FOUND DURING DIAGNOSIS: the
#         two previous floor changes (V4.6's 0.4->0.2 and V4.8's 0.2->0.14)
#         were both justified using measurements that turned out to have been
#         run against B1.py (the backlight tool), not this file - confirmed
#         directly: running B1.py against the exact "Acetic+BSA_3.8pH_2.9V/
#         3.0V/3.1V/3.2V.JPG" reflected-light photos reproduces core-area
#         numbers (0.1840/0.1691/0.1692/0.1523mm2) matching the values V4.8's
#         own changelog cited (0.1774/0.1626/0.1627/0.1459mm2) almost exactly
#         - both V4.8 and this file itself already said as much ("NOT
#         INDEPENDENTLY TESTED against a real reflected-light photo... watch
#         for this"), but the gap had never actually been closed until now.
#         Separately, floor-lowering was tested directly for safety: swept
#         MINIMUM_AREA_MM2 through 0.4/0.2/0.14/0.05 against V4_report/2.9V.JPG
#         and 2.9V2.JPG (both currently-working reflected-light detections).
#         Result was strictly monotonic at every step - lowering the floor
#         only ever ADDED candidates, never removed one, and every already-
#         qualifying aggregate's measured area was bit-for-bit identical
#         across all four floor values. No mechanism found by which a lower
#         floor can break an existing detection. 0.10 is set below the
#         0.13mm2 highlight-confirmed blob found during the local-contrast
#         diagnostic work (real localized highlight, margin +23, 100% raw
#         Otsu coverage in its own bounding box - not noise), with this same
#         session's validation pass (see below) serving as the first real
#         reflected-light confirmation this floor has ever had.
#
#         (2) PROTOTYPE, EXPERIMENTAL, OPT-IN ONLY - NOT part of the default
#         active pipeline: a new, architecturally separate local-contrast
#         secondary seed path (ENABLE_LOCAL_CONTRAST_PROTOTYPE, default
#         False). Motivated by the session-2 finding that some real
#         aggregates never generate a single raw-Otsu foreground pixel
#         anywhere in their own footprint, because they lack a localized
#         bright highlight even though they're clearly visible by eye (best
#         evidence: the 2.9V-to-3.0V decay series, where raw Otsu coverage in
#         the aggregate's own bbox collapsed from 40-58% [detected] to 2.0%
#         [missed] to 0.0% [missed] over one voltage step). When enabled,
#         this path evaluates ONLY pixels the existing raw global-Otsu pass
#         left at zero (never a tweak to the existing global cutoff itself),
#         and flags a pixel only if it exceeds ITS OWN local-neighborhood
#         mean by both a z-score margin (LOCAL_CONTRAST_STD_MULTIPLIER) AND
#         an absolute minimum gray-level margin (LOCAL_CONTRAST_MIN_MARGIN -
#         guards against near-zero-std flat regions trivially qualifying).
#         Candidate pixels must then form a contiguous blob at least
#         LOCAL_CONTRAST_MIN_SEED_AREA_MM2 in size to become a seed at all -
#         single hot pixels or small noise clusters are rejected before they
#         ever reach the hysteresis/growth machinery, same safeguard
#         discipline as the rest of this file (area floors + explicit
#         minimum-contiguous-area gates, not "a lower bar everywhere").
#         Surviving candidates are unioned into the existing seed_mask and
#         flow through the SAME hysteresis growth, growth-distance-cap, and
#         final area-floor classification as any Otsu-found core - no
#         separate output/measurement code path. Kept behind the
#         ENABLE_LOCAL_CONTRAST_PROTOTYPE flag (default OFF) so every
#         existing detection is completely unaffected unless this is
#         explicitly turned on for testing. See this session's validation
#         comparison table for results - not yet reviewed/approved for the
#         default active path.
#
# V4.11 - PROTOTYPE, EXPERIMENTAL, OPT-IN ONLY - NOT part of the default
#         active pipeline: fragment clustering (ENABLE_FRAGMENT_CLUSTERING,
#         default False). A DIFFERENT bug from V4.10's local-contrast path -
#         that one targets aggregates with zero raw-Otsu coverage anywhere;
#         this one targets aggregates where global Otsu DOES find real
#         signal, but it lands in several small disconnected pieces instead
#         of one blob, so no single piece clears MINIMUM_AREA_MM2 alone.
#         CONFIRMED, not assumed: traced V4_report/3.0V.JPG directly - 2.02%
#         raw Otsu coverage in the aggregate's known bbox, essentially
#         unchanged after morphology (2.00%/2.01% - morphology was NOT the
#         cause, ruled out by direct before/after pixel counts), but split
#         across 13-15 disconnected components. The two largest were already
#         being logged individually as rejected candidates (0.0761mm2,
#         0.0511mm2), neither clearing the 0.10mm2 floor alone; combined
#         with the rest, total footprint (~0.19mm2) matches the aggregate's
#         independently-established real size (0.14-0.18mm2 from earlier
#         sessions).
#
#         Runs after Otsu + morphology, before classify_contours()'s
#         area-floor rejection - never touches the Otsu threshold or the
#         morphology step. Components already large enough to qualify alone,
#         and border-artifact-sized components, are left completely
#         untouched. Sub-floor components within FRAGMENT_CLUSTER_MAX_GAP_FRACTION
#         of each other (single-linkage, nearest-pixel distance - not
#         centroid/bbox distance, which would misjudge irregular fragment
#         shapes) are grouped; a group is only promoted to a seed candidate
#         if its combined real fragment area clears MINIMUM_AREA_MM2 AND its
#         convex-hull fill ratio clears FRAGMENT_CLUSTER_MIN_FILL_RATIO (the
#         chaining-failure safeguard - single-linkage clustering can bridge a
#         long sparse trail of unrelated specks if only distance is checked;
#         a real fragmented aggregate's hull is comparatively dense, an
#         accidental chain of scattered dust is not).
#
#         Gap threshold grounded in direct measurement, not a guess: the two
#         largest (already-logged) fragments in the validated 3.0V case sit
#         279px apart; a single-linkage MST needs a 196px max hop to connect
#         all 13-15 fragments into one group whose total area matches the
#         known real size. FRAGMENT_CLUSTER_MAX_GAP_FRACTION is set to
#         connect that full case (~200px at this photo's resolution).
#         CORRECTION: the V4.10-session Check-1 report described these gaps
#         as "tens to ~150px" - that was an eyeballed guess from the
#         fragments' bounding-box spread, not measured, and is superseded by
#         the numbers above.
#
#         The returned candidate mask is the FILLED CONVEX HULL of each
#         qualifying group, not just the original fragment pixels - this
#         necessarily includes "bridge" area between fragments that was
#         never actual raw foreground. CAUGHT DURING THIS SESSION, not
#         theoretical: the first working version reported the raw hull area
#         as the aggregate's true_area_mm2 - on the validated 3.0V case that
#         came out to 2.42mm2 against a real fragment-pixel area of 0.19mm2,
#         a ~12x inflation, silently wrong. FIXED: true_area_mm2 (and
#         circularity) are now overridden back to the real fragment-pixel-
#         count area for any aggregate whose seed includes fragment-cluster
#         pixels, with a printed NOTE disclosing the override - same "trust
#         the pixel count over the contour polygon, disclose the gap"
#         pattern already used for core/recovered area (V4.7). KNOWN
#         REMAINING GAP, not yet fixed: core_area_mm2/hysteresis_area_mm2
#         (and the overgrowth-ratio print) still derive from seed_mask,
#         which still contains the un-corrected hull - the existing V4.7
#         display clamp protects the on-screen Core/Recovered table columns,
#         but the raw internal core_area_mm2 field and any overgrowth
#         warning for a cluster-origin aggregate remain inflated. FILL_RATIO
#         ALSO HONESTLY DISCLOSED: 0.06 was set just under the one validated
#         case's own measured ratio (0.079), i.e. reverse-engineered from a
#         single confirmed case, not independently derived - tested clean
#         (zero false clusters) against all 21 real photos in this session's
#         validation set, but that is not an adversarial dust-field stress
#         test. See this session's validation comparison table for full
#         results, including the false-candidate and close-aggregate safety
#         checks - not yet reviewed/approved for the default active path.
#
# V4.12 - STILL PROTOTYPE, STILL DEFAULT OFF. Two follow-ups to V4.11, kept
#         separate:
#
#         (1) REAL FIX: core_area_mm2/hysteresis_area_mm2/overgrowth_ratio
#         were the "known remaining gap" flagged in V4.11 - fixed the same
#         way true_area_mm2 was: a bridge_only_mask (fragment-cluster hull
#         pixels minus the real fragment pixels) is now excluded from
#         core_area_px_count for any cluster-origin aggregate, with a
#         printed NOTE disclosing the correction. Traced all the way through
#         compute_photo_summary() to the actual CSV row, not just the
#         matplotlib table, per request: total_core/total_recovered in the
#         CSV export use display_core_mm2/display_recovered_mm2, which were
#         ALREADY protected by the existing V4.7 clamp even before this fix
#         (clamped against the already-corrected true_area_mm2) - so this
#         fix changes the raw internal core_area_mm2 field and any future
#         overgrowth warning, not what was already reaching the CSV for
#         core/recovered specifically. NEITHER core_area_mm2/hysteresis_
#         area_mm2/is_overgrown/overgrowth_ratio are themselves CSV columns
#         today (CSV_COLUMNS has no such column) - the concern about bad
#         numbers reaching the permanent log turned out to apply to
#         DIFFERENT fields instead, found while tracing this through:
#         avg_relative_intensity_index and total_combined_index (both real
#         CSV columns) come out as 0.0 for a cluster-origin aggregate,
#         because aggregate_mean is computed over agg_mask - which is still
#         the full hull shape, diluted by background-colored bridge pixels,
#         not just the real fragment pixels. ALSO FOUND: is_small (and
#         has_flagged_small_aggregate, a real CSV column) can read False for
#         a cluster-origin aggregate even when its corrected true size is
#         well under CONFIDENT_AREA_MM2, because small-flagging happens in
#         classify_contours() on the hull-inflated area, upstream of the
#         true_area_mm2 correction. NEITHER of these two newly-found gaps
#         was fixed this session - flagged, not fixed, since they were
#         outside what was asked.
#
#         (2) STRESS TEST, requested specifically because "no false
#         positives in 21 photos" was not the same as "confirmed safe."
#         It wasn't: a broader scan across 106 available photos (not just
#         the validated 21) found 17 with >=1 cluster candidate forming at
#         the current settings (gap ~200px, fill_ratio 0.06), most in
#         photos never checked before. Visually confirmed on Test3.JPG (a
#         real calibration-type photo with a textured fabric background,
#         the same general kind of photo this tool processes) - 9 separate
#         false candidates, all sitting on plain fabric texture nowhere near
#         the actual object in frame. A synthetic follow-up isolated the
#         mechanism: ROUND scattered dust specks (the originally-assumed
#         threat model) never triggered a false positive even at very loose
#         fill-ratio settings, because round dust rarely accumulates enough
#         combined area to clear MINIMUM_AREA_MM2 within one cluster. The
#         REAL failure mode is different: a few ELONGATED fragments (thin
#         scratches/fabric threads/fold lines, matching Test3.JPG's actual
#         shapes) can each carry enough area alone that just 2-3 of them
#         within gap range clear the floor easily. At the current fill_ratio
#         0.06, 3 random elongated fragments within a 150px radius produced
#         a false cluster in 6/15 synthetic trials (40%). Sweeping the fill-
#         ratio threshold against that same elongated-fragment case: still
#         failing at 0.10 (5/15), 0.15 (4/15), and 0.20 (1/15); clean (0/15)
#         starting at 0.25. CONCLUSION, not yet acted on: 0.06 is confirmed
#         unsafe against a real, relevant, already-observed failure mode -
#         FRAGMENT_CLUSTER_MIN_FILL_RATIO left AT 0.06 in this version
#         deliberately, since tightening it is a decision for review, not
#         something to change unilaterally mid-session. ENABLE_FRAGMENT_
#         CLUSTERING stays False throughout.
#
# V4.13 - FIXED (PROTOTYPE, gated behind ENABLE_FRAGMENT_CLUSTERING=False):
#         the two contaminated-CSV-column gaps flagged (not fixed) in V4.12.
#         (1) avg_relative_intensity_index/total_combined_index: aggregate_
#         mean is now recomputed over ONLY the real fragment pixels
#         (fragment_original_mask) for any cluster-origin aggregate, instead
#         of the full hull region - same disclosed-override pattern as
#         true_area_mm2. Verified on 3.0V: aggregate_mean corrected 128.1 ->
#         183.4, relative_intensity_index 0.0 -> 0.1903 (was reading exactly
#         0.0 before, diluted to background level by hull/bridge pixels).
#         3.0V2: 128.8 -> 174.5, intensity index 0.0 -> 0.1940. (2) is_small/
#         has_flagged_small_aggregate: now re-derived from the CORRECTED
#         true_area_mm2 against CONFIDENT_AREA_MM2 for any cluster-origin
#         aggregate, instead of trusting classify_contours()'s upstream
#         verdict on the pre-correction hull area. Verified on 3.0V/3.0V2:
#         is_small False -> True in both cases (0.19mm2/0.19mm2 true area is
#         correctly well under the 1.0mm2 confident floor - it was reading
#         False only because the hull area, ~2.4mm2/~1.6mm2, cleared that
#         floor). Both fixes confirmed inert with ENABLE_FRAGMENT_CLUSTERING
#         at its default False (cluster_bridge_px is always 0 with the
#         prototype off, so neither override branch executes).
#
#         ALSO: attempted to resolve the fill-ratio tension flagged in
#         V4.12 - 3.0V's own validated fill ratio (0.079) sits well inside
#         the range the V4.12 stress test showed was unsafe (failures didn't
#         fully stop until ~0.25), so any single fill-ratio threshold either
#         reopens the false-positive hole or defeats the one real case this
#         feature exists for. A fine-grained sweep (0.06 to 0.10 in 0.005
#         steps, 30 trials/step) confirmed there is NO safe value in that
#         range at all: the false-positive rate is FLAT at 27% (8/30) across
#         the entire 0.06-0.095 span, only dropping to 23% at 0.10 - 3.0V's
#         0.079 sits in the middle of the danger zone with zero margin in
#         either direction, not near an edge.
#
#         Searched for a second discriminating signal per request (num
#         fragments, mean fragment size, fragment density, spatial spread).
#         num_fragments looked promising at first: 3.0V/3.0V2 cluster from
#         13/16 fragments, while every false candidate found (Test3.JPG's 9
#         real false positives, all synthetic failures) clustered from only
#         2-3. A synthetic sweep up to 25 scattered elongated fragments never
#         produced a false candidate with >=13 fragments (0/280 trials) -
#         looked like a clean separator. It is NOT: (a) a REAL photo
#         (IMG_20260702_111826.jpg, flagged in V4.12) has one false candidate
#         clustering from 73 fragments at fill_ratio 0.075 - both well past
#         any num_fragments floor and inside 3.0V's own fill-ratio range;
#         (b) a broader synthetic sweep using denser dust fields (30-90
#         scattered elongated fragments, matching that real photo's scale)
#         regularly produced candidates with 11-30 fragments at fill_ratio
#         0.06-0.15 - 60-93% of trials, depending on density - which directly
#         overlaps 3.0V's signature (13 fragments, fill_ratio 0.079) with no
#         gap between them. mean_fragment_area_mm2 was checked as a third
#         candidate signal and also failed to separate: those same dense-dust
#         false candidates averaged 0.02-0.04mm2 per fragment, overlapping
#         3.0V's 0.0220mm2 and 3.0V2's 0.0177mm2 directly (dust field 70/250:
#         0.0204mm2, essentially identical to 3.0V).
#
#         CONCLUSION: no single geometric property tested (fill_ratio, num_
#         fragments, mean fragment area, or fragment density = num_fragments/
#         hull_area) separates a real dispersed aggregate from a dense
#         cluster of dust/texture fragments with any real margin - the two
#         cases can look geometrically identical. Per instruction not to
#         force a compromise or present an uncertain result as settled: NO
#         combined check was prototyped, since none of the tested signals
#         would have actually worked (prototyping a check already shown to
#         fail its own validation would misrepresent the finding). This is a
#         genuinely open problem for fragment-clustering as currently
#         designed, not a tuning gap - the mechanism as built may need a
#         non-geometric signal (e.g. requiring visual/manual confirmation of
#         cluster-origin candidates rather than auto-counting them) to be
#         safe to enable, or it may need to stay a prototype indefinitely.
#         FRAGMENT_CLUSTER_MIN_FILL_RATIO left at 0.06, ENABLE_FRAGMENT_
#         CLUSTERING left at False - both untouched, per instruction.
#
# V4.14 - REDESIGNED (supersedes the fully-automatic fragment-clustering of
#         V4.11-V4.13, which is retired, not deleted - cluster_fragments()
#         and its supporting code are unchanged, only how the result is USED
#         changed): since V4.13's stress test found no geometric property
#         (fill ratio, fragment count, mean fragment area) reliably separates
#         a real dispersed aggregate from a dense dust/fold-line field, this
#         version stops trying to auto-classify cluster-origin candidates as
#         real/not-real. Instead they are a distinct, always-flagged category
#         requiring manual visual confirmation: a magenta square marker (not
#         the existing orange "small" or red "overgrowth" colors) in the
#         result image, a "CLUSTER?" label, and EXCLUSION from every
#         automatic tally - the results table's TOTAL row, compute_photo_
#         summary()'s count/area/etc, and the CSV master log's confirmed
#         columns. Area/fragment-count/fill-ratio are still printed/logged
#         per candidate (console output + two new table columns) so a human
#         has the numbers needed to confirm or reject it by eye. ENABLE_
#         FRAGMENT_CLUSTERING renamed to ENABLE_CLUSTER_FLAGGING (the
#         behavior it gates changed enough to warrant a new name - it no
#         longer controls auto-counting, only whether candidates are found
#         and flagged at all) and now defaults to True: unlike the old
#         auto-counting version, there is no silent-false-positive risk left
#         to default off against, since nothing this produces is ever
#         auto-counted.
#
#         CSV DESIGN DECISION (was a choice, not forced): rather than fully
#         excluding cluster candidates from the CSV (audit trail lost) or
#         folding them into the existing confirmed columns (defeats the
#         point), added two new columns - unconfirmed_cluster_count and
#         unconfirmed_cluster_area_mm2 - that record them SEPARATELY.
#         aggregate_count/total_area_mm2/etc keep meaning exactly what they
#         meant before this feature existed.
#
#         PERFORMANCE BUG FOUND AND FIXED while validating (unrelated to
#         cluster-flagging, reproduces identically with the feature off):
#         _dilated_crop_for_label() built a cv2.dilate kernel sized
#         (2*cap_radius+1), where cap_radius scales with a blob's own
#         equivalent radius - for a real photo with a ~700px-radius
#         aggregate (IMG_3109.JPG) this produced a 2087x2087 kernel, and
#         cv2.dilate with an arbitrary-shaped kernel that large costs
#         proportional to kernel area, not image area. Confirmed via
#         cProfile: 1200s+ of a 1227s total analyze_image() call was inside
#         {dilate}, present identically whether ENABLE_CLUSTER_FLAGGING was
#         on or off. Replaced with cv2.distanceTransform (the exact same
#         Euclidean-circular region within cap_radius of the component,
#         computed in near-linear time regardless of cap_radius, not an
#         approximation) - confirmed ~45x speedup on the worst observed case
#         (1227s -> 27s) with byte-identical measurements on a previously-
#         validated case (3.0V unchanged at 0.1924mm2).
#
#         AREA-CORRUPTION BUG FOUND AND FIXED while validating (introduced in
#         V4.11, present through V4.13): the true_area_mm2/circularity
#         override for a cluster-origin aggregate computed its corrected area
#         from ONLY fragment_original_mask pixels (the sub-floor fragments
#         actually being clustered) - wrong whenever a cluster hull happens
#         to touch/overlap a SEPARATE, already-qualifying real aggregate,
#         since cleaned_for_core unions cluster_seed_mask into the whole-
#         image mask before contour-finding, so a hull that merely brushes a
#         nearby unrelated blob merges them into one contour. The old formula
#         then silently discarded that merged-in aggregate's real pixels
#         (not being fragment pixels themselves), reporting a drastically
#         undersized area. Confirmed on a real photo (IMG_20260706_102434.jpg):
#         a genuine 98.3mm2 aggregate collapsed to a reported 0.17mm2 purely
#         from touching a small, unrelated 7-fragment dust cluster's hull.
#         The SAME bug existed in the V4.12 aggregate_mean correction (used
#         for avg_relative_intensity_index/total_combined_index). Both fixed
#         to match the logic core_area_mm2 already used correctly: exclude
#         ONLY bridge_only_mask pixels (the hull-fill gaps that were never
#         real foreground), keep every other real foreground pixel regardless
#         of whether it came from the clustered fragments or a separate
#         merged-in blob. Verified inert on 3.0V/3.0V2 (no separate blob
#         nearby there, so old and new formulas agree exactly - unchanged at
#         0.1924mm2/0.1906mm2); verified fixed on IMG_20260706_102434.jpg
#         (the corrupted 0.17mm2/0.30mm2 candidates now correctly read
#         100.5mm2/11.7mm2, matching the real merged-in aggregates' true
#         size).
#
#         VALIDATION (95 photos, all available calibrated photos across both
#         data folders, with both fixes above applied): 91 photos have
#         BYTE-IDENTICAL confirmed (non-cluster) detections whether ENABLE_
#         CLUSTER_FLAGGING is on or off - count, area, everything. 4 photos
#         (IMG_20260702_111826.jpg, IMG_20260706_102434.jpg, IMG_3113.JPG,
#         Test3.JPG) show a confirmed-count/area change - in every case
#         because a cluster hull touched an already-qualifying real
#         aggregate and merged it into a flagged candidate, which is the
#         intended behavior (flag anything a cluster hull touches for
#         review), not a bug - confirmed by reconciliation: off-area
#         approximately equals on-confirmed-area + on-unconfirmed-area for
#         all 4 (e.g. IMG_3113.JPG: 5032.21 ~= 4651.16 + 382.83mm2; small
#         residual differences come from hole/bubble-exclusion shape changes
#         on the altered contour, not lost area). 12 of 95 photos produced
#         at least one cluster candidate. KNOWN, DISCLOSED, NOT FIXED this
#         version: when a cluster hull touches an already-confirmed real
#         aggregate, that aggregate's OWN classification flips to
#         unconfirmed too, even if e.g. 99% of the merged contour's area is
#         solid/confident and only a small fleck at the edge is the actual
#         dust - the area shown is now honest (fixed above), but a
#         previously-standalone confident detection can still lose its
#         "confirmed" status just for being spatially close to unrelated
#         noise. Whether this conservative-but-imprecise behavior is
#         acceptable, or whether merged contours should eventually be split
#         back into their real-aggregate and cluster-candidate parts, is left
#         for review - flagged, not decided unilaterally.
# =============================================

# --- Otsu thresholding ---
BLUR_SIZE_FRACTION  = 0.0043
MORPH_SIZE_FRACTION = 0.0009

# --- Illumination flattening ---
ILLUMINATION_KERNEL_FRACTION = 0.5
BORDER_ARTIFACT_AREA_FRACTION = 0.9

# --- Calibration line color detection (HSV) ---
HUE_CENTERS = {'RED': 0, 'GREEN': 60, 'BLUE': 120}  # OpenCV hue scale is 0-180
HUE_TOLERANCE  = 15
SATURATION_MIN = 80
VALUE_MIN      = 40

# --- Calibration line shape filter ---
MIN_ASPECT_RATIO         = 3.0
MIN_LINE_LENGTH_FRACTION = 0.05

# --- Aggregate noise filter ---
MINIMUM_AREA_MM2 = 0.10
CONFIDENT_AREA_MM2 = 1.0
BORDERLINE_AREA_MM2 = 0.05

# --- Local-contrast secondary seed path (V4.10, PROTOTYPE/EXPERIMENTAL) ---
# Default OFF - existing detections are completely unaffected unless this is
# explicitly enabled. See V4.10 changelog for the full rationale.
ENABLE_LOCAL_CONTRAST_PROTOTYPE = False
LOCAL_CONTRAST_KERNEL_FRACTION = 0.06   # local-neighborhood size, smaller
                                          # than the global illumination blur
                                          # (0.5) so it captures "brighter
                                          # than its own surroundings" rather
                                          # than re-deriving the global trend
LOCAL_CONTRAST_STD_MULTIPLIER = 3.0     # stricter than the hysteresis loose
                                          # threshold's 2.0 - this path is
                                          # more noise-prone than a global
                                          # cutoff, so it needs a higher bar
LOCAL_CONTRAST_MIN_MARGIN = 4           # absolute gray-level floor on top of
                                          # the z-score test, so a near-flat
                                          # patch with tiny std can't qualify
                                          # on noise alone
LOCAL_CONTRAST_MIN_SEED_AREA_MM2 = 0.10  # a contiguous-area gate applied
                                          # BEFORE a candidate becomes a seed
                                          # at all, independent of the final
                                          # area-floor classification -
                                          # single hot pixels/small noise
                                          # clusters never reach hysteresis

# --- Fragment clustering (V4.11, PROTOTYPE/EXPERIMENTAL) ---
# Default OFF - existing detections are completely unaffected unless this is
# explicitly enabled. See V4.11 changelog. MOTIVATED BY a directly-measured
# case (V4_report/3.0V.JPG): a real, Otsu-crossing aggregate (2.02% raw
# coverage, confirmed surviving morphology intact) was scattered across 13
# disconnected connected-components instead of one blob, none individually
# clearing MINIMUM_AREA_MM2 alone even though their combined footprint
# (~0.19mm2) matches the aggregate's established real size. Gap threshold
# below is grounded in DIRECT MEASUREMENT on that case, not a guess: the two
# largest fragments (already individually logged as rejected candidates,
# 0.0761mm2 and 0.0511mm2) sit 279px apart; a single-linkage spanning tree
# needs a 196px max hop to connect all 13 fragments into one group whose
# total area (~0.19mm2) matches the known real aggregate size. NOTE: an
# earlier informal estimate (Check 1 report) described these gaps as "tens to
# ~150px" - that was an eyeballed guess from the fragments' bounding-box
# spread, not a real measurement, and is superseded by the numbers here.
# RENAMED (V4.14) from ENABLE_FRAGMENT_CLUSTERING: the behavior this flag
# controls changed. It no longer auto-counts a cluster candidate as a
# confirmed aggregate - see the V4.14 changelog entry for why (documented
# geometric indistinguishability from dust, found via stress test in V4.13).
# Cluster candidates are now always flagged for manual visual confirmation
# and excluded from every automatic tally (table TOTAL row, CSV export).
# That removes the silent-false-positive risk the old auto-counting version
# had, so this defaults to True - there's no longer a downside to leaving it
# on other than an extra marker to glance at in photos that have one.
ENABLE_CLUSTER_FLAGGING = True
FRAGMENT_CLUSTER_MAX_GAP_FRACTION = 0.058  # ~200px at 3456px width - matches
                                             # the measured 196px MST distance
                                             # for the validated real case
FRAGMENT_CLUSTER_MIN_FILL_RATIO = 0.06      # combined fragment pixel area /
                                             # convex-hull area of the group -
                                             # guards against a threshold that
                                             # is small in isolation but still
                                             # chains together a long, sparse
                                             # trail of unrelated specks
                                             # (classic single-linkage
                                             # "chaining" failure mode).
                                             # HONEST NOTE on how this number
                                             # was set: the validated 3.0V
                                             # case's own fill ratio measured
                                             # 0.079 (13 fragments, hull
                                             # ~12x the real pixel area) -
                                             # 0.06 is set just under that
                                             # with a small margin, i.e.
                                             # reverse-engineered from the one
                                             # confirmed real case, NOT
                                             # independently derived from a
                                             # dust/noise distribution. Tested
                                             # clean (zero false clusters)
                                             # against 21 real photos incl.
                                             # known vignette/dust-heavy
                                             # frames, but that is not the
                                             # same as an adversarial worst-
                                             # case dust field - see this
                                             # session's validation report
                                             # for the explicit caveat.

# =============================================
# CSV MASTER LOG
# =============================================
# V4.7 FIX: MASTER_CSV_PATH used to be derived from os.path.dirname(__file__)
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
MASTER_CSV_FILENAME = "plasma_analysis_master_log.csv"
MASTER_CSV_DIR = r"C:\Users\66950\Desktop\Projects in github\Plasma\data"
os.makedirs(MASTER_CSV_DIR, exist_ok=True)
MASTER_CSV_PATH = os.path.join(MASTER_CSV_DIR, MASTER_CSV_FILENAME)

CSV_COLUMNS = [
    "run_timestamp", "filename", "version", "build_tag",
    "calibration_status", "calibration_failure_reason",
    "reference_mm", "calibration_color", "mm_per_px", "background_mean",
    "aggregate_count", "total_area_mm2", "total_holes", "avg_circularity",
    "avg_relative_intensity_index", "total_combined_index",
    "total_volume_mm3", "has_flagged_small_aggregate",
    # V4.14: cluster-origin candidates (fragmented signal that only clears
    # the size floor when combined - see is_cluster_origin comment in
    # analyze_image) are NEVER folded into aggregate_count/total_area_mm2/
    # etc above - those columns stay exactly what they were before this
    # feature existed, so no permanent-log number can be silently corrupted
    # by an unconfirmed candidate. These two columns record them SEPARATELY
    # instead of dropping them from the log entirely, so the master log
    # still has an audit trail of "a candidate existed here and was left for
    # manual review" without it ever being mistaken for a confirmed count.
    "unconfirmed_cluster_count", "unconfirmed_cluster_area_mm2",
]


def append_photo_to_master_csv(filename, params, result):
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
    else:
        row["calibration_status"] = "success"
        row["mm_per_px"] = f'{result["mm_per_px"]:.5f}'
        row["background_mean"] = f'{result["background_mean"]:.1f}'

        summary = compute_photo_summary(result)
        if summary is None:
            row["aggregate_count"] = 0
            row["unconfirmed_cluster_count"] = 0
        else:
            row["aggregate_count"] = summary["count"]
            row["total_area_mm2"] = f'{summary["total_area"]:.3f}'
            row["total_holes"] = summary["total_holes"]
            row["avg_circularity"] = f'{summary["avg_circularity"]:.3f}'
            row["avg_relative_intensity_index"] = f'{summary["avg_intensity_idx"]:.3f}'
            row["total_combined_index"] = f'{summary["total_combined"]:.3f}'
            row["total_volume_mm3"] = (
                f'{summary["total_volume"]:.3f}' if summary["total_volume"] is not None else "TBD"
            )
            row["has_flagged_small_aggregate"] = "TRUE" if summary["has_small"] else "FALSE"
            # V4.14: recorded separately from aggregate_count/total_area_mm2
            # above - see the CSV_COLUMNS comment for why these are never
            # merged into the confirmed totals.
            row["unconfirmed_cluster_count"] = summary["unconfirmed_count"]
            row["unconfirmed_cluster_area_mm2"] = f'{summary["unconfirmed_total_area"]:.3f}'

    with open(MASTER_CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if file_is_new:
            writer.writeheader()
        writer.writerow(row)

# --- Hysteresis edge recovery ---
LOOSE_THRESHOLD_STD_MULTIPLIER = 2.0
GROWTH_MAX_RADIUS_MULTIPLIER = 1.5

# --- Bubble exclusion (Hough circle detection) ---
BUBBLE_MIN_RADIUS_FRACTION = 0.006
BUBBLE_MAX_RADIUS_FRACTION = 0.06
BUBBLE_DETECTION_MAX_DIMENSION = 1000
BUBBLE_CORE_OVERLAP_MAX = 0.3

OVERGROWTH_RATIO = 0.5

# BGR draw colors, just for the calibration overlay visualization
DRAW_COLORS = {'RED': (0, 0, 255), 'GREEN': (0, 255, 0), 'BLUE': (255, 0, 0)}
HYSTERESIS_COLOR_BGR = (255, 255, 0)  # cyan tint for grown-in pixels
OVERGROWN_COLOR_BGR = (0, 0, 255)     # red outline for flagged/suspicious growth
# V4.14: distinct from all four colors already in use (green=confident,
# orange=small/low-confidence, red=overgrowth flag, cyan=hysteresis-grown
# fill) - magenta was already the convention this file's own diagnostic
# scripts used for candidate bboxes, kept consistent here.
CLUSTER_UNCONFIRMED_COLOR_BGR = (255, 0, 255)  # magenta - requires visual confirmation


# =============================================
# ILLUMINATION FLATTENING
# =============================================

def make_odd(n):
    n = int(n)
    return n if n % 2 == 1 else n + 1


def classify_contours(binary_mask, minimum_area_px, confident_area_px,
                       borderline_area_px, border_artifact_area_px,
                       mm2_per_px2, collect_diagnostics=False):
    """PORTED from B1.py (B1.3/B1.9 diagnostics): when collect_diagnostics is
    True, also returns a "why was this rejected" list (border-artifact vs
    below-floor), computed from the core pass only - unaffected by
    hysteresis growth. Debug-only addition, see V4.9 changelog."""
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


# Safety cap: if a photo's cleaned mask has more sub-floor fragments than
# this, skip fragment clustering entirely for that photo rather than pay an
# O(n^2) cost on what's already a very noisy frame - see cluster_fragments().
FRAGMENT_CLUSTER_MAX_FRAGMENTS = 300


def cluster_fragments(cleaned, minimum_area_px, border_artifact_area_px,
                       gap_px, min_fill_ratio, mm2_per_px2):
    """PROTOTYPE/EXPERIMENTAL (V4.11) - see changelog. Groups individually-
    sub-floor connected components in `cleaned` (the post-morphology mask)
    that sit within gap_px of each other, when their COMBINED area would
    clear MINIMUM_AREA_MM2 even though none does alone. Runs AFTER Otsu +
    morphology, BEFORE classify_contours()'s area-floor rejection - never
    touches the global Otsu threshold or the morphology step itself.

    Components already big enough to qualify alone, and border-artifact-
    sized components, are left completely untouched (existing path,
    unchanged). Single-linkage grouping by nearest-pixel distance (not
    centroid or bbox distance - correct for irregular fragment shapes).
    Each resulting group must ALSO pass a convex-hull fill-ratio sanity
    check (min_fill_ratio) - single-linkage clustering is prone to
    "chaining" a long sparse trail of unrelated specks together at a gap
    threshold that looks safe for any one pair in isolation; a real
    fragmented aggregate's hull is comparatively dense, an accidental chain
    of scattered dust is not.

    Returns (cluster_seed_mask, fragment_original_mask, candidate_info).
    cluster_seed_mask is the FILLED CONVEX HULL of each qualifying group,
    not just the original fragment pixels - this necessarily includes some
    "bridge" area between fragments that was never actual raw foreground
    (unavoidable: a single contour needs a connected shape). Measured
    directly on the validated 3.0V case: hull area came out ~12x the real
    fragment-pixel area (fill ratio 0.079) - far too large a discrepancy to
    let the hull stand in as the aggregate's reported size. fragment_
    original_mask is the SAME group's pixels with NO hull-fill, i.e. only
    real Otsu-crossing pixels - the caller uses this to override
    true_area_mm2 for cluster-origin aggregates back to something honest,
    the same "pixel-count vs contour-polygon area can differ, trust the
    pixel count, disclose the discrepancy" pattern already used for core/
    hysteresis area (see V4.7 changelog). candidate_info reports both areas
    for every group so the inflation is visible, not hidden."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)

    sub_floor_labels = []
    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area > border_artifact_area_px or area > minimum_area_px:
            continue  # already qualifies alone, or a border artifact - untouched
        sub_floor_labels.append(lbl)

    if len(sub_floor_labels) < 2:
        return np.zeros(cleaned.shape, dtype=np.uint8), np.zeros(cleaned.shape, dtype=np.uint8), []
    if len(sub_floor_labels) > FRAGMENT_CLUSTER_MAX_FRAGMENTS:
        print(f"  [PROTOTYPE] fragment clustering skipped: {len(sub_floor_labels)} sub-floor "
              f"fragments exceeds the {FRAGMENT_CLUSTER_MAX_FRAGMENTS}-fragment safety cap "
              f"(frame too noisy to cluster safely/cheaply)")
        return np.zeros(cleaned.shape, dtype=np.uint8), np.zeros(cleaned.shape, dtype=np.uint8), []

    bboxes = {lbl: stats[lbl, cv2.CC_STAT_LEFT:cv2.CC_STAT_LEFT + 4] for lbl in sub_floor_labels}
    parent = {lbl: lbl for lbl in sub_floor_labels}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    img_h, img_w = cleaned.shape
    for i in range(len(sub_floor_labels)):
        li = sub_floor_labels[i]
        lx, ly, lw, lh = bboxes[li]
        for j in range(i + 1, len(sub_floor_labels)):
            lj = sub_floor_labels[j]
            jx, jy, jw, jh = bboxes[lj]
            # cheap bounding-box pre-filter before any pixel-level distance work
            bbox_gap_x = max(0, max(lx, jx) - min(lx + lw, jx + jw))
            bbox_gap_y = max(0, max(ly, jy) - min(ly + lh, jy + jh))
            if (bbox_gap_x ** 2 + bbox_gap_y ** 2) ** 0.5 > gap_px:
                continue
            # exact nearest-pixel distance, computed on a small local crop
            # (not the full image) - the two fragments plus gap_px padding
            x0 = max(0, min(lx, jx) - gap_px)
            y0 = max(0, min(ly, jy) - gap_px)
            x1 = min(img_w, max(lx + lw, jx + jw) + gap_px)
            y1 = min(img_h, max(ly + lh, jy + jh) + gap_px)
            crop_labels = labels[y0:y1, x0:x1]
            mask_i = (crop_labels == li).astype(np.uint8)
            mask_j = crop_labels == lj
            if not mask_i.any() or not mask_j.any():
                continue
            dist = cv2.distanceTransform(1 - mask_i, cv2.DIST_L2, 5)
            if dist[mask_j].min() <= gap_px:
                union(li, lj)

    groups = {}
    for lbl in sub_floor_labels:
        groups.setdefault(find(lbl), []).append(lbl)

    cluster_seed_mask = np.zeros(cleaned.shape, dtype=np.uint8)
    # Separate from cluster_seed_mask on purpose: the ORIGINAL fragment
    # pixels only, with no hull-fill bridge area added. cluster_seed_mask
    # necessarily includes bridge pixels between fragments that were never
    # real foreground (unavoidable - a single contour needs a connected
    # shape) - but that bridge area must NOT be counted as measured
    # aggregate area, or a sparse, dispersed real signal (fill ratio ~0.08
    # measured on the validated 3.0V case) gets reported at 10x+ its real
    # size. The caller uses this mask to override true_area_mm2 for
    # cluster-origin aggregates back to the real fragment pixel count,
    # exactly the same "pixel-count area vs contour-polygon area can differ,
    # trust the pixel count, disclose the discrepancy" pattern already used
    # for core/hysteresis area (see V4.7 changelog).
    fragment_original_mask = np.zeros(cleaned.shape, dtype=np.uint8)
    candidate_info = []
    for members in groups.values():
        if len(members) < 2:
            continue  # isolated fragment, nothing to cluster - falls through
                       # to the normal below-floor rejection, unchanged
        total_area_px = sum(int(stats[m, cv2.CC_STAT_AREA]) for m in members)
        if total_area_px <= minimum_area_px:
            continue  # still doesn't clear the floor even combined

        member_mask = np.isin(labels, members).astype(np.uint8)
        pts = cv2.findNonZero(member_mask)
        hull = cv2.convexHull(pts)
        hull_area_px = cv2.contourArea(hull)
        fill_ratio = total_area_px / max(hull_area_px, 1)
        if fill_ratio < min_fill_ratio:
            continue  # sanity check failed - looks like a chained/bridged
                       # trail of unrelated specks, not one real object

        group_mask = np.zeros(cleaned.shape, dtype=np.uint8)
        cv2.fillPoly(group_mask, [hull], 255)
        cluster_seed_mask = cv2.bitwise_or(cluster_seed_mask, group_mask)
        fragment_original_mask[member_mask == 1] = 255

        hx, hy, hw, hh = cv2.boundingRect(hull)
        candidate_info.append({
            'num_fragments': len(members),
            'fragment_area_mm2': total_area_px * mm2_per_px2,
            'hull_area_mm2': hull_area_px * mm2_per_px2,
            'fill_ratio': fill_ratio,
            'bbox': (int(hx), int(hy), int(hw), int(hh)),
        })
    return cluster_seed_mask, fragment_original_mask, candidate_info


def find_local_contrast_seeds(blurred, raw_binary, line_exclusion_mask, morph_size, mm2_per_px2):
    """PROTOTYPE/EXPERIMENTAL (V4.10) - see changelog. Finds candidate seed
    blobs in regions the existing raw global-Otsu pass (raw_binary) never
    flagged at all, by checking whether a pixel exceeds its OWN local-
    neighborhood mean rather than one frame-wide cutoff. Architecturally
    separate from the global Otsu step - never modifies or reads its
    threshold value, only its output mask (to restrict itself to Otsu's
    blind spots).

    Returns (seed_candidate_mask, candidate_info) where candidate_info is a
    list of dicts (bbox, area_mm2, mean_z, mean_margin) for every surviving
    candidate blob, for diagnostics/reporting - this function makes no
    decisions about counting/area-floor classification, that still happens
    downstream in the normal classify_contours() pass once these candidates
    are unioned into seed_mask."""
    img_h, img_w = blurred.shape
    kernel_size = make_odd(LOCAL_CONTRAST_KERNEL_FRACTION * min(img_h, img_w))
    blurred_f = blurred.astype(np.float32)
    local_mean = cv2.GaussianBlur(blurred_f, (kernel_size, kernel_size), 0)
    local_sqmean = cv2.GaussianBlur(blurred_f * blurred_f, (kernel_size, kernel_size), 0)
    local_var = np.clip(local_sqmean - local_mean * local_mean, 0, None)
    local_std = np.sqrt(local_var)

    margin = blurred_f - local_mean
    z_score = margin / np.maximum(local_std, 1e-3)
    candidate_mask = ((z_score >= LOCAL_CONTRAST_STD_MULTIPLIER) &
                       (margin >= LOCAL_CONTRAST_MIN_MARGIN)).astype(np.uint8) * 255

    # Only evaluate Otsu's blind spots - pixels the raw (pre-morphology)
    # global threshold already flagged are left entirely to the existing
    # path, never re-decided here.
    candidate_mask[raw_binary == 255] = 0
    candidate_mask[line_exclusion_mask == 255] = 0

    # Same noise-removal kernel as the main pipeline, so a single hot pixel
    # can't survive into a "blob" on its own.
    kernel = np.ones((morph_size, morph_size), np.uint8)
    candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_OPEN, kernel)

    min_seed_area_px = LOCAL_CONTRAST_MIN_SEED_AREA_MM2 / mm2_per_px2
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate_mask, connectivity=8)
    seed_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    candidate_info = []
    for lbl in range(1, num_labels):
        area_px = stats[lbl, cv2.CC_STAT_AREA]
        if area_px < min_seed_area_px:
            continue
        blob_mask = labels == lbl
        seed_mask[blob_mask] = 255
        candidate_info.append({
            'bbox': (int(stats[lbl, cv2.CC_STAT_LEFT]), int(stats[lbl, cv2.CC_STAT_TOP]),
                     int(stats[lbl, cv2.CC_STAT_WIDTH]), int(stats[lbl, cv2.CC_STAT_HEIGHT])),
            'area_mm2': float(area_px * mm2_per_px2),
            'mean_z': float(z_score[blob_mask].mean()),
            'mean_margin': float(margin[blob_mask].mean()),
        })
    return seed_mask, candidate_info


def detect_bubble_mask(gray, seed_mask, line_exclusion_mask):
    img_h, img_w = gray.shape
    long_dim = max(img_h, img_w)
    scale = min(1.0, BUBBLE_DETECTION_MAX_DIMENSION / long_dim)
    small_gray = cv2.resize(gray, (max(1, int(img_w * scale)), max(1, int(img_h * scale))),
                             interpolation=cv2.INTER_AREA) if scale < 1.0 else gray

    short_dim = min(small_gray.shape)
    min_r = max(3, int(BUBBLE_MIN_RADIUS_FRACTION * short_dim))
    max_r = max(min_r + 1, int(BUBBLE_MAX_RADIUS_FRACTION * short_dim))

    blurred_for_circles = cv2.medianBlur(small_gray, 5)
    circles = cv2.HoughCircles(
        blurred_for_circles, cv2.HOUGH_GRADIENT, dp=1.5, minDist=min_r,
        param1=80, param2=30, minRadius=min_r, maxRadius=max_r
    )

    bubble_mask = np.zeros(gray.shape, dtype=np.uint8)
    if circles is None:
        return bubble_mask

    for cx, cy, r in circles[0]:
        cx, cy, r = cx / scale, cy / scale, r / scale
        circle_mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.circle(circle_mask, (int(round(cx)), int(round(cy))), int(round(r)), 255, -1)
        circle_area = np.count_nonzero(circle_mask == 255)
        if circle_area == 0:
            continue
        core_overlap = np.count_nonzero((circle_mask == 255) & (seed_mask == 255))
        if (core_overlap / circle_area) <= BUBBLE_CORE_OVERLAP_MAX:
            bubble_mask[circle_mask == 255] = 255

    bubble_mask[line_exclusion_mask == 255] = 0
    return bubble_mask


def _dilated_crop_for_label(labels, label, stats, seed_mask_shape):
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
    # PERF FIX (V4.14): replaced cv2.dilate with a (2*cap_radius+1)-sized
    # circular kernel - found, while investigating why some dense photos
    # took 20-60+ minutes, to be catastrophically slow for large aggregates.
    # A single ~700px-equivalent-radius blob (real case: IMG_3109.JPG) drove
    # cap_radius to ~1043px, i.e. a 2087x2087 structuring element - cv2.dilate
    # with an arbitrary-shaped kernel that large is proportional to kernel
    # area, not image area, so cost exploded. This bug is UNRELATED to
    # cluster-flagging (V4.11-V4.14) - it's in the pre-existing hysteresis
    # growth-radius cap and reproduces identically with ENABLE_CLUSTER_
    # FLAGGING off; confirmed via profiling (cProfile showed 1200s+ of a
    # 1227s total run inside {dilate} with the flag OFF). cv2.distanceTransform
    # computes the exact same thing - the Euclidean-circular region within
    # cap_radius of component_crop - in time that does not blow up with
    # cap_radius, since it's a near-linear-time algorithm over the crop
    # regardless of radius. Mathematically equivalent to the original (the
    # old kernel was also circular, same width/height), not an approximation.
    dist = cv2.distanceTransform(255 - component_crop, cv2.DIST_L2, 5)
    dilated_crop = np.where(dist <= cap_radius, 255, 0).astype(np.uint8)
    return y0, y1, x0, x1, dilated_crop


def build_growth_distance_cap(seed_mask):
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


def flatten_illumination(gray):
    h, w = gray.shape
    kernel_size = make_odd(min(h, w) * ILLUMINATION_KERNEL_FRACTION)
    illumination = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    diff = gray.astype(np.int16) - illumination.astype(np.int16)
    return np.clip(diff + 128, 0, 255).astype(np.uint8)


# =============================================
# CALIBRATION LINE DETECTION
# =============================================

def get_color_mask(hsv_image, color_mode):
    h = hsv_image[:, :, 0].astype(int)
    s = hsv_image[:, :, 1].astype(int)
    v = hsv_image[:, :, 2].astype(int)

    center = HUE_CENTERS[color_mode]

    if color_mode == 'RED':
        # red wraps around the 0/180 boundary on OpenCV's hue scale
        hue_mask = (h <= HUE_TOLERANCE) | (h >= 180 - HUE_TOLERANCE)
    else:
        hue_mask = np.abs(h - center) <= HUE_TOLERANCE

    return hue_mask & (s >= SATURATION_MIN) & (v >= VALUE_MIN)


def find_calibration_line(color_mask):
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
        if aspect_ratio >= MIN_ASPECT_RATIO and length_px >= min_length_px:
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
# =============================================

def get_parameters_gui():
    import tkinter as tk
    from tkinter import filedialog, ttk

    result = {}
    selected_paths = []
    root = tk.Tk()
    root.title("Plasma Aggregation Analysis - Setup")
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
            color_mode = 'RED'

        thickness_text = thickness_var.get().strip()
        thickness_mm = None
        if thickness_text != "":
            try:
                t = float(thickness_text)
                if t > 0:
                    thickness_mm = t
            except ValueError:
                pass

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
    # ---- Calibration ----
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    color_mask = get_color_mask(hsv, color_mode)
    line = find_calibration_line(color_mask)

    if line is None:
        return {'success': False, 'reason': (
            f"No {color_mode.lower()} calibration line could be confirmed.\n"
            f"Possible causes: the line isn't in frame, lighting is too poor,\n"
            f"or the wrong color was selected for this photo."
        )}

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

    # ---- Grayscale + illumination flattening + blur ----
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_h, img_w = gray.shape
    blur_size = max(3, make_odd(BLUR_SIZE_FRACTION * min(img_h, img_w)))
    morph_size = max(2, int(round(MORPH_SIZE_FRACTION * min(img_h, img_w))))
    flattened = flatten_illumination(gray)
    blurred = cv2.GaussianBlur(flattened, (blur_size, blur_size), 0)

    # ---- Calibration line exclusion mask (used twice: threshold + final result) ----
    line_exclusion_mask = np.zeros(gray.shape, dtype=np.uint8)
    (lcx, lcy), (lw, lh), langle = line['rect']
    padded_rect = ((lcx, lcy), (lw + 20, lh + 20), langle)
    line_box = np.int32(cv2.boxPoints(padded_rect))
    cv2.fillPoly(line_exclusion_mask, [line_box], 255)

    # ---- Otsu threshold, computed without the calibration line's pixels ----
    otsu_threshold, _ = cv2.threshold(
        blurred[line_exclusion_mask == 0], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    _, binary = cv2.threshold(blurred, otsu_threshold, 255, cv2.THRESH_BINARY)
    raw_foreground_fraction = float(np.count_nonzero(binary)) / (img_h * img_w)

    # ---- Noise removal ----
    kernel = np.ones((morph_size, morph_size), np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    # ---- Exclude the calibration line's own region from the final result ----
    line_center_point = (int(lcx), int(lcy))
    pre_exclusion_contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in pre_exclusion_contours:
        if cv2.pointPolygonTest(c, line_center_point, False) >= 0:
            cv2.drawContours(cleaned, [c], -1, 0, -1)
    cleaned[line_exclusion_mask == 255] = 0

    # ---- Core contours + holes (the trusted, flattened-Otsu anchor) ----
    minimum_area_px = MINIMUM_AREA_MM2 / mm2_per_px2
    confident_area_px = CONFIDENT_AREA_MM2 / mm2_per_px2
    borderline_area_px = BORDERLINE_AREA_MM2 / mm2_per_px2
    border_artifact_area_px = BORDER_ARTIFACT_AREA_FRACTION * img_h * img_w
    foreground_fraction = float(np.count_nonzero(cleaned)) / (img_h * img_w)

    # ---- CLUSTER FLAGGING (V4.11, redesigned V4.14): fragment clustering ----
    # Default ON (ENABLE_CLUSTER_FLAGGING=True) - when off, cleaned_for_core
    # is just `cleaned` unchanged and everything downstream is byte-identical
    # to pre-V4.11 behavior. Runs on `cleaned` (post-morphology, pre-floor) -
    # never touches the Otsu threshold or the morphology step. See V4.14
    # changelog and cluster_fragments() docstring. IMPORTANT: a cluster
    # candidate found here still becomes a normal contour/measurement further
    # down (so its area/fragment-count/fill-ratio can be reported), but is
    # flagged via is_cluster_origin and excluded from every automatic tally -
    # see the is_cluster_origin block below and compute_photo_summary().
    fragment_cluster_diagnostics = {'enabled': ENABLE_CLUSTER_FLAGGING, 'candidates': []}
    cleaned_for_core = cleaned
    cluster_seed_mask = np.zeros(gray.shape, dtype=np.uint8)
    fragment_original_mask = np.zeros(gray.shape, dtype=np.uint8)
    if ENABLE_CLUSTER_FLAGGING:
        gap_px = int(round(FRAGMENT_CLUSTER_MAX_GAP_FRACTION * min(img_h, img_w)))
        cluster_seed_mask, fragment_original_mask, cluster_candidates = cluster_fragments(
            cleaned, minimum_area_px, border_artifact_area_px,
            gap_px, FRAGMENT_CLUSTER_MIN_FILL_RATIO, mm2_per_px2)
        fragment_cluster_diagnostics['candidates'] = cluster_candidates
        if np.any(cluster_seed_mask):
            cleaned_for_core = cv2.bitwise_or(cleaned, cluster_seed_mask)
            for c in cluster_candidates:
                print(f"  [PROTOTYPE] fragment cluster: {c['num_fragments']} fragments -> "
                      f"real={c['fragment_area_mm2']:.4f}mm2 hull={c['hull_area_mm2']:.4f}mm2 "
                      f"fill_ratio={c['fill_ratio']:.2f} bbox={c['bbox']}")

    (core_contours, core_hierarchy, core_outer_contours,
     core_hole_contours, core_small_indices, rejected) = classify_contours(
        cleaned_for_core, minimum_area_px, confident_area_px, borderline_area_px,
        border_artifact_area_px, mm2_per_px2, collect_diagnostics=True)

    # DEBUG-ONLY (V4.9, temporary): raw-vs-post-morphology Otsu foreground
    # coverage and the largest rejected candidate blob(s), PORTED from
    # B1.py's B1.3 diagnostics. Added to investigate real aggregates being
    # reported as 0 contours - see V4.9 changelog.
    detection_diagnostics = {
        'raw_foreground_fraction': raw_foreground_fraction,
        'foreground_fraction': foreground_fraction,
        'rejected_top5': rejected[:5],
    }

    # ---- Hysteresis edge recovery ----
    qualifying_core_idx = {i for _, i in core_outer_contours}
    seed_mask = cleaned_for_core.copy()
    if core_hierarchy is not None:
        for i, contour in enumerate(core_contours):
            parent_idx = core_hierarchy[0][i][3]
            if parent_idx == -1 and i not in qualifying_core_idx:
                cv2.drawContours(seed_mask, [contour], -1, 0, -1)

    # ---- PROTOTYPE/EXPERIMENTAL (V4.10): local-contrast secondary seed path ----
    # Default OFF (ENABLE_LOCAL_CONTRAST_PROTOTYPE=False) - when off, this
    # block is a no-op and seed_mask/detection_diagnostics are byte-identical
    # to pre-V4.10 behavior. Only ever ADDS seed pixels in regions the raw
    # global-Otsu pass (binary) left untouched - never removes or reinterprets
    # anything the existing pipeline already found. See V4.10 changelog.
    local_contrast_diagnostics = {'enabled': ENABLE_LOCAL_CONTRAST_PROTOTYPE, 'candidates': []}
    local_contrast_seed_mask = np.zeros(gray.shape, dtype=np.uint8)
    if ENABLE_LOCAL_CONTRAST_PROTOTYPE:
        local_contrast_seed_mask, local_contrast_candidates = find_local_contrast_seeds(
            blurred, binary, line_exclusion_mask, morph_size, mm2_per_px2)
        local_contrast_diagnostics['candidates'] = local_contrast_candidates
        if np.any(local_contrast_seed_mask):
            seed_mask = cv2.bitwise_or(seed_mask, local_contrast_seed_mask)
            print(f"  [PROTOTYPE] local-contrast path added {len(local_contrast_candidates)} "
                  f"seed candidate(s): " +
                  ", ".join(f"{c['area_mm2']:.4f}mm2(z={c['mean_z']:.1f})" for c in local_contrast_candidates))

    # PROTOTYPE (V4.11) core-area correction, part 2: bridge_only_mask marks
    # exactly the hull-fill pixels added by fragment clustering that were
    # NEVER real Otsu foreground (cluster_seed_mask minus the real fragment
    # pixels). true_area_mm2 was already corrected to exclude this - without
    # also excluding it from core_area_px_count below, that field (and the
    # overgrowth ratio derived from it) would stay wrong even after the
    # true_area_mm2 fix, and so would compute_photo_summary()'s has_overgrown/
    # max_overgrowth_ratio, which are NOT currently written to the CSV
    # (CSV_COLUMNS has no overgrowth column today) but total_core/
    # total_recovered ARE computed from these same fields via display_core_mm2/
    # display_recovered_mm2 - those are already protected by the existing
    # V4.7 clamp against true_area_mm2, but the raw core_area_mm2 field itself
    # was still wrong before this fix, disclosed via the printed NOTE below.
    bridge_only_mask = (cluster_seed_mask == 255) & (fragment_original_mask == 0)

    core_hole_mask = np.zeros(gray.shape, dtype=np.uint8)
    for hole_contour, parent_idx in core_hole_contours:
        if parent_idx in qualifying_core_idx:
            cv2.drawContours(core_hole_mask, [hole_contour], -1, 255, -1)

    raw_background_pixels = gray[(seed_mask == 0) & (line_exclusion_mask == 0)]
    if raw_background_pixels.size > 0:
        raw_background_mean = float(raw_background_pixels.mean())
        raw_background_std = float(raw_background_pixels.std())
    else:
        raw_background_mean, raw_background_std = float(gray.mean()), float(gray.std())
    loose_threshold = raw_background_mean + LOOSE_THRESHOLD_STD_MULTIPLIER * raw_background_std
    print(f"  Loose threshold (hysteresis, raw): {loose_threshold:.1f}  "
          f"(background mean {raw_background_mean:.1f} + {LOOSE_THRESHOLD_STD_MULTIPLIER} "
          f"x std {raw_background_std:.1f})")

    loose_mask = np.where(gray > loose_threshold, 255, 0).astype(np.uint8)
    loose_mask[line_exclusion_mask == 255] = 0
    loose_mask[core_hole_mask == 255] = 0

    # ---- Bubble exclusion: growth must never cross into a detected bubble ----
    bubble_mask = detect_bubble_mask(gray, seed_mask, line_exclusion_mask)
    bubble_pixel_count = int(np.count_nonzero(bubble_mask == 255))
    if bubble_pixel_count > 0:
        print(f"  Bubble exclusion triggered: {bubble_pixel_count}px marked as "
              f"bubble, excluded from hysteresis growth")
    loose_mask[bubble_mask == 255] = 0

    # ---- Growth distance cap ----
    growth_allowed_mask = build_growth_distance_cap(seed_mask)
    loose_mask[growth_allowed_mask == 0] = 0

    union_mask = cv2.bitwise_or(seed_mask, loose_mask)
    num_labels, labels = cv2.connectedComponents(union_mask, connectivity=8)
    seed_labels = set(np.unique(labels[seed_mask == 255])) - {0}
    if seed_labels:
        grown_mask = np.where(np.isin(labels, list(seed_labels)), 255, 0).astype(np.uint8)
    else:
        grown_mask = seed_mask.copy()

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

    # ---- Background mean intensity (for the relative intensity index) ----
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

        agg_mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(agg_mask, [contour], -1, 255, -1)
        for hc in my_holes:
            cv2.drawContours(agg_mask, [hc], -1, 0, -1)
        agg_pixels = gray[agg_mask == 255]
        aggregate_mean = float(agg_pixels.mean()) if agg_pixels.size > 0 else 0.0

        # PROTOTYPE (V4.12, FIXED V4.14) intensity-index fix: aggregate_mean
        # above is averaged over the full hull region (agg_mask), which for a
        # cluster-origin candidate includes non-real bridge pixels sitting on
        # plain background between fragments - that dilutes aggregate_mean
        # toward background_mean, which was driving relative_intensity_index
        # (and therefore combined_index) to ~0.0 even for a genuinely bright
        # aggregate. Same disclosed-override pattern as true_area_mm2 above:
        # for a cluster-origin aggregate, recompute aggregate_mean over all
        # real foreground pixels (excluding ONLY bridge_only_mask), not just
        # the hull.
        #
        # V4.14 BUG FIX: same mistake as true_area_mm2 above - this used to
        # average ONLY fragment_original_mask pixels, which silently dropped
        # any separate real aggregate's pixels that got merged into the same
        # contour via a touching cluster hull (see true_area_mm2 fix comment
        # for the concrete real-photo case). Fixed to match: exclude only
        # bridge_only_mask, keep every other real foreground pixel.
        cluster_bridge_px_for_mean = int(np.count_nonzero((agg_mask == 255) & bridge_only_mask))
        if cluster_bridge_px_for_mean > 0:
            real_pixels = gray[(agg_mask == 255) & (~bridge_only_mask)]
            if real_pixels.size > 0:
                corrected_mean = float(real_pixels.mean())
                print(f"  NOTE: aggregate #{i + 1} aggregate_mean corrected for fragment-cluster "
                      f"bridge pixels - was {aggregate_mean:.1f} (diluted by hull/bridge "
                      f"background pixels), now {corrected_mean:.1f} (real foreground pixels only).")
                aggregate_mean = corrected_mean

        # PROTOTYPE (V4.11, FIXED V4.14) fragment-cluster area correction:
        # true_area_mm2/true_area_px above are cv2.contourArea on the FILLED
        # CONVEX HULL for a cluster-origin candidate, not the real foreground
        # footprint - measured directly on the validated 3.0V case, that hull
        # came out ~12x the real fragment pixel area (fill ratio 0.079).
        # Overridden here back to the real, honest pixel-count area for any
        # aggregate whose seed includes fragment-cluster pixels - same
        # "pixel-count vs contour-polygon area can differ, trust the pixel
        # count, disclose it" pattern already used for core/recovered area
        # below (V4.7). circularity is recomputed too, using the corrected
        # area against the hull's unchanged perimeter - a low circularity
        # here is accurate and expected for a genuinely dispersed cluster,
        # not a bug.
        #
        # V4.14 BUG FIX: this used to count ONLY fragment_original_mask
        # pixels (the sub-floor fragments actually being clustered), which
        # is wrong whenever the cluster hull happens to touch/overlap a
        # SEPARATE, already-qualifying real aggregate - cleaned_for_core
        # unions cluster_seed_mask into the whole-image mask before contour-
        # finding, so a hull that merely brushes a nearby unrelated blob
        # merges them into one contour. The old formula then silently
        # discarded that nearby aggregate's real pixels (not being fragment
        # pixels themselves), reporting a drastically undersized area - a
        # real photo (IMG_20260706_102434.jpg) showed a genuine 98.3mm2
        # aggregate collapse to 0.17mm2 this way when a small unrelated dust
        # cluster's hull happened to touch it. Fixed to match the logic
        # core_area_mm2 already used further below: exclude ONLY bridge_only_
        # mask pixels (the hull-fill gaps that were never real foreground),
        # keep every other real foreground pixel regardless of whether it
        # came from the clustered fragments or a separate merged-in blob.
        # Verified inert on the validated 3.0V/3.0V2 cases (no separate blob
        # nearby there, so the two formulas agree exactly - same 0.1924mm2/
        # 0.1906mm2 as before this fix).
        cluster_bridge_px = int(np.count_nonzero((agg_mask == 255) & bridge_only_mask))
        if cluster_bridge_px > 0:
            corrected_area_px = int(np.count_nonzero((agg_mask == 255) & (~bridge_only_mask)))
            corrected_area_mm2 = corrected_area_px * mm2_per_px2
            print(f"  NOTE: aggregate #{i + 1} includes a fragment-cluster region - "
                  f"contour/hull-based area ({true_area_mm2:.4f}mm2) overridden to the "
                  f"real foreground pixel-count area ({corrected_area_mm2:.4f}mm2); "
                  f"{cluster_bridge_px}px ({cluster_bridge_px * mm2_per_px2:.4f}mm2) of "
                  f"hull/bridge area excluded as not real foreground.")
            true_area_px = corrected_area_px
            true_area_mm2 = corrected_area_mm2
            circularity = (4 * np.pi * true_area_px / (perimeter_px ** 2)
                           if perimeter_px > 0 else 0)

        denom = max(255.0 - background_mean, 1e-6)
        relative_intensity_index = max(0.0, (aggregate_mean - background_mean) / denom)
        combined_index = true_area_mm2 * relative_intensity_index

        volume_mm3 = true_area_mm2 * thickness_mm if thickness_mm is not None else None

        # PROTOTYPE diagnostics only (V4.10/V4.11): which seed source(s) this
        # aggregate came from - doesn't affect area/classification, purely
        # for validating the local-contrast and fragment-clustering paths
        # against known cases.
        has_local_contrast_seed = bool(np.any((agg_mask == 255) & (local_contrast_seed_mask == 255)))
        has_fragment_cluster_seed = bool(np.any((agg_mask == 255) & (cluster_seed_mask == 255)))
        has_otsu_seed = bool(np.any((agg_mask == 255) & (seed_mask == 255) &
                                     (local_contrast_seed_mask == 0) & (cluster_seed_mask == 0)))
        origins = []
        if has_otsu_seed:
            origins.append('otsu')
        if has_local_contrast_seed:
            origins.append('local_contrast')
        if has_fragment_cluster_seed:
            origins.append('fragment_cluster')
        seed_origin = '+'.join(origins) if origins else 'otsu'

        # REDESIGNED (V4.14): is_cluster_origin marks this aggregate as
        # requiring manual visual confirmation before being counted - see the
        # V4.14 changelog. This supersedes the V4.11-V4.13 fully-automatic
        # version, which auto-counted fragment-cluster aggregates the same as
        # any other; that was retired because a stress test (V4.13 changelog)
        # found no geometric property (fill ratio, fragment count, mean
        # fragment area) reliably separates a real dispersed aggregate from a
        # dense dust/fold-line field - some real false-positive photos land
        # in the exact same numeric range as the validated real cases. cluster_
        # num_fragments/cluster_fill_ratio are looked up from the matching
        # entry in fragment_cluster_diagnostics['candidates'] (by bbox-center
        # containment) so a human reviewing the output has the numbers needed
        # to make the call - these are None for a non-cluster-origin aggregate.
        is_cluster_origin = has_fragment_cluster_seed
        cluster_num_fragments = None
        cluster_fill_ratio = None
        if is_cluster_origin:
            for cand in fragment_cluster_diagnostics['candidates']:
                cx_, cy_, cw_, ch_ = cand['bbox']
                ccx, ccy = cx_ + cw_ // 2, cy_ + ch_ // 2
                if 0 <= ccy < agg_mask.shape[0] and 0 <= ccx < agg_mask.shape[1] and agg_mask[ccy, ccx] == 255:
                    cluster_num_fragments = cand['num_fragments']
                    cluster_fill_ratio = cand['fill_ratio']
                    break

        # PROTOTYPE (V4.11) core-area correction, part 2 (see bridge_only_mask
        # comment above): exclude fragment-cluster bridge pixels from the
        # core pixel count - they were never real Otsu foreground, just
        # hull-fill needed to make the cluster a single contour. hysteresis_
        # area_px_count is untouched: bridge pixels were always inside
        # seed_mask (never counted as hysteresis-grown), so that field was
        # never wrong in the first place - only core_area_px_count was.
        core_bridge_px = int(np.count_nonzero((agg_mask == 255) & bridge_only_mask))
        core_area_px_count = int(np.count_nonzero((agg_mask == 255) & (seed_mask == 255) &
                                                    (~bridge_only_mask)))
        hysteresis_area_px_count = int(np.count_nonzero((agg_mask == 255) & (seed_mask == 0)))
        core_area_mm2 = core_area_px_count * mm2_per_px2
        hysteresis_area_mm2 = hysteresis_area_px_count * mm2_per_px2
        if core_bridge_px > 0:
            print(f"  NOTE: aggregate #{i + 1} core_area_mm2 corrected for fragment-cluster "
                  f"bridge pixels - {core_bridge_px}px ({core_bridge_px * mm2_per_px2:.4f}mm2) "
                  f"of hull/bridge area excluded from the core pixel count (same correction "
                  f"already applied to true_area_mm2 above).")

        is_overgrown = hysteresis_area_mm2 > OVERGROWTH_RATIO * max(core_area_mm2, 1e-9)
        overgrowth_ratio = hysteresis_area_mm2 / max(core_area_mm2, 1e-9)
        if is_overgrown:
            print(f"  WARNING: aggregate #{i + 1} hysteresis-recovered area "
                  f"({hysteresis_area_mm2:.4f}mm2) exceeds {OVERGROWTH_RATIO * 100:.0f}% "
                  f"of its core area ({core_area_mm2:.4f}mm2) - flagged as suspicious growth "
                  f"(recovered/core = {overgrowth_ratio:.2f}x)")

        # V4.7 reporting addition: core/recovered breakdown for the visible
        # tables (V4.4/B1.5 already computed core_area_mm2/hysteresis_area_mm2,
        # but only ever printed them to console). Both are raster PIXEL
        # COUNTS (partition of agg_mask by seed_mask), while true_area_mm2 is
        # a cv2.contourArea (Green's-theorem polygon) measurement - the same
        # measure used everywhere else in this file for area floors/
        # classification, so true_area_mm2 must stay the authoritative
        # "Area" column. Pixel-count area and contour-polygon area are not
        # identical measures (boundary-pixel effects, most visible on small
        # aggregates near the area floor - exactly the case this change was
        # requested for), so core_area_mm2 + hysteresis_area_mm2 is NOT
        # guaranteed to equal true_area_mm2 exactly. Checked, not assumed:
        # display_core/display_recovered below are DERIVED so they always
        # reconcile to true_area_mm2 by construction (display_recovered =
        # Area - display_core), instead of showing two independently-
        # measured numbers that could silently fail to add up in the table a
        # user is trusting at a glance. is_overgrown/overgrowth_ratio above
        # are left driven by the original raw pixel-count hysteresis_area_mm2
        # - unchanged detection behavior, per spec.
        display_core_mm2 = min(core_area_mm2, true_area_mm2)
        display_recovered_mm2 = true_area_mm2 - display_core_mm2

        # PROTOTYPE (V4.12) is_small fix: contour_idx in small_indices below
        # reflects classify_contours()'s verdict on the PRE-correction hull
        # area (it runs upstream of the true_area_mm2 override above), so a
        # cluster-origin aggregate whose corrected true_area_mm2 is actually
        # well under CONFIDENT_AREA_MM2 was reading is_small=False simply
        # because its inflated hull area cleared the confident floor. Same
        # disclosed-override pattern: re-derive is_small from the corrected,
        # authoritative true_area_mm2 instead of trusting the stale upstream
        # verdict for any cluster-origin aggregate.
        is_small_verdict = contour_idx in small_indices
        if cluster_bridge_px > 0:
            corrected_is_small = true_area_mm2 < CONFIDENT_AREA_MM2
            if corrected_is_small != is_small_verdict:
                print(f"  NOTE: aggregate #{i + 1} is_small corrected for fragment-cluster "
                      f"hull inflation - was {is_small_verdict} (based on pre-correction hull "
                      f"area), now {corrected_is_small} (based on corrected true_area_mm2="
                      f"{true_area_mm2:.4f}mm2 vs {CONFIDENT_AREA_MM2}mm2 confident floor).")
            is_small_verdict = corrected_is_small
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
            'is_small': is_small_verdict,
            'is_overgrown': is_overgrown,
            'is_cluster_origin': is_cluster_origin,
            'cluster_num_fragments': cluster_num_fragments,
            'cluster_fill_ratio': cluster_fill_ratio,
            'seed_origin': seed_origin,
            'center_x': center_x, 'center_y': center_y,
            'bbox_x': x, 'bbox_y': y, 'bbox_w': w, 'bbox_h': h,
            'background_mean': background_mean,
            'aggregate_mean': aggregate_mean,
            'relative_intensity_index': relative_intensity_index,
            'combined_index': combined_index,
            'volume_mm3': volume_mm3,
        })

    # ---- Clean result image ----
    result_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    final_mask = np.zeros(gray.shape, dtype=np.uint8)
    for contour, _ in outer_contours:
        cv2.drawContours(final_mask, [contour], -1, 255, -1)
    for hc, _ in hole_contours:
        cv2.drawContours(final_mask, [hc], -1, 0, -1)
    hysteresis_mask = np.where((final_mask == 255) & (seed_mask == 0), 255, 0).astype(np.uint8)

    if np.any(hysteresis_mask):
        result_image[hysteresis_mask == 255] = HYSTERESIS_COLOR_BGR

    # V4.14: cluster-origin candidates get their own category, drawn/labeled
    # distinctly and taking priority over the small/overgrown flags below -
    # see is_cluster_origin comment above for why (they require manual visual
    # confirmation, not automatic small/overgrowth classification).
    cluster_indices = {m['contour_idx'] for m in measurements if m['is_cluster_origin']}
    overgrown_indices = {m['contour_idx'] for m in measurements
                          if m['is_overgrown'] and m['contour_idx'] not in cluster_indices}
    confident_contours = [c for c, i in outer_contours
                           if i not in small_indices and i not in overgrown_indices and i not in cluster_indices]
    small_contours = [c for c, i in outer_contours
                       if i in small_indices and i not in overgrown_indices and i not in cluster_indices]
    overgrown_contours = [c for c, i in outer_contours if i in overgrown_indices]
    cluster_contours = [c for c, i in outer_contours if i in cluster_indices]
    cv2.drawContours(result_image, confident_contours, -1, (0, 255, 0), 2)
    cv2.drawContours(result_image, small_contours, -1, (0, 165, 255), 2)
    cv2.drawContours(result_image, overgrown_contours, -1, OVERGROWN_COLOR_BGR, 4)
    cv2.drawContours(result_image, cluster_contours, -1, CLUSTER_UNCONFIRMED_COLOR_BGR, 5)
    cv2.drawContours(result_image, [hc for hc, _ in hole_contours], -1, (0, 0, 255), 2)
    for m in measurements:
        if m['is_cluster_origin']:
            dot_color = CLUSTER_UNCONFIRMED_COLOR_BGR
            label = f"#{m['id']} CLUSTER?"
            # Distinct marker shape (filled square), not just a color swap -
            # a colorblind viewer or a black-and-white printout should still
            # be able to tell this apart from the small/overgrown dots.
            cx_, cy_ = m['center_x'], m['center_y']
            cv2.rectangle(result_image, (cx_ - 7, cy_ - 7), (cx_ + 7, cy_ + 7), dot_color, -1)
        elif m['is_overgrown']:
            dot_color = OVERGROWN_COLOR_BGR
            label = f"#{m['id']}!"
            cv2.circle(result_image, (m['center_x'], m['center_y']), 6, dot_color, -1)
        elif m['is_small']:
            dot_color = (0, 165, 255)
            label = f"#{m['id']}*"
            cv2.circle(result_image, (m['center_x'], m['center_y']), 6, dot_color, -1)
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
        'bubble_pixel_count': bubble_pixel_count,
        'measurements': measurements,
        'background_mean': background_mean,
        'thickness_mm': thickness_mm,
        'detection_diagnostics': detection_diagnostics,
        'local_contrast_diagnostics': local_contrast_diagnostics,
        'fragment_cluster_diagnostics': fragment_cluster_diagnostics,
    }


# =============================================
# DISPLAY — single window, 2x2 grid
# =============================================

def compute_photo_summary(r):
    all_measurements = r['measurements']
    if not all_measurements:
        return None
    # V4.14: cluster-origin candidates require manual visual confirmation
    # (see is_cluster_origin comment in analyze_image) and are EXCLUDED from
    # every automatic tally below - 'confirmed' drives count/area/etc exactly
    # like before this feature existed. unconfirmed_* fields report the
    # cluster candidates separately so they're visible without being counted.
    measurements = [m for m in all_measurements if not m['is_cluster_origin']]
    unconfirmed = [m for m in all_measurements if m['is_cluster_origin']]
    total_volume = sum(m['volume_mm3'] for m in measurements) if r['thickness_mm'] is not None else None
    overgrown = [m for m in measurements if m['is_overgrown']]
    return {
        'count': len(measurements),
        'total_area': sum(m['true_area_mm2'] for m in measurements),
        'total_core': sum(m['display_core_mm2'] for m in measurements),
        'total_recovered': sum(m['display_recovered_mm2'] for m in measurements),
        'total_holes': sum(m['num_holes'] for m in measurements),
        'avg_circularity': (sum(m['circularity'] for m in measurements) / len(measurements)
                             if measurements else 0.0),
        'avg_intensity_idx': (sum(m['relative_intensity_index'] for m in measurements) / len(measurements)
                               if measurements else 0.0),
        'total_combined': sum(m['combined_index'] for m in measurements),
        'total_volume': total_volume,
        'has_small': any(m['is_small'] for m in measurements),
        'has_overgrown': len(overgrown) > 0,
        'max_overgrowth_ratio': max((m['overgrowth_ratio'] for m in overgrown), default=None),
        'unconfirmed_count': len(unconfirmed),
        'unconfirmed_total_area': sum(m['true_area_mm2'] for m in unconfirmed),
        'has_unconfirmed_cluster': len(unconfirmed) > 0,
    }


def display_results(r, filename=None):
    measurements = r['measurements']
    name_part = f"{filename}  |  " if filename else ""

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(
        f"Plasma Aggregation Analysis v{VERSION} [{BUILD_TAG}]  |  {name_part}"
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

        col_labels = ['#', 'Scale\n(mm/px)', 'Area\n(mm2)', 'Core area\n(mm2)',
                      'Recovered area\n(mm2)', 'Holes\n(count)',
                      'Circularity\n(unitless)', 'Relative intensity\nindex (unitless)',
                      'Combined index\n(mm2)', 'Volume\n(mm3)',
                      'Cluster\nfragments', 'Cluster\nfill ratio']
        RECOVERED_COL = 4
        scale_str = f"{r['mm_per_px']:.5f}"
        any_small = any(m['is_small'] for m in measurements)
        any_overgrown = any(m['is_overgrown'] for m in measurements)
        any_cluster = any(m['is_cluster_origin'] for m in measurements)
        cluster_row_indices = []
        cell_text = []
        for m in measurements:
            vol_str = f"{m['volume_mm3']:.3f}" if m['volume_mm3'] is not None else "TBD"
            # V4.14: cluster-origin candidates get their own distinct id
            # marker ("CLUSTER?"), which takes priority over the small/
            # overgrown markers - see is_cluster_origin comment in
            # analyze_image for why these are a separate category, always
            # requiring a human look regardless of size/overgrowth status.
            if m['is_cluster_origin']:
                id_str = f"{m['id']} CLUSTER?"
                cluster_row_indices.append(len(cell_text) + 1)
            elif m['is_overgrown']:
                id_str = f"{m['id']}! ({m['overgrowth_ratio']:.1f}x)"
            elif m['is_small']:
                id_str = f"{m['id']}*"
            else:
                id_str = f"{m['id']}"
            frag_str = str(m['cluster_num_fragments']) if m['cluster_num_fragments'] is not None else "-"
            fill_str = f"{m['cluster_fill_ratio']:.3f}" if m['cluster_fill_ratio'] is not None else "-"
            cell_text.append([
                id_str, scale_str, f"{m['true_area_mm2']:.3f}",
                f"{m['display_core_mm2']:.3f}", f"{m['display_recovered_mm2']:.3f}",
                f"{m['num_holes']}",
                f"{m['circularity']:.3f}", f"{m['relative_intensity_index']:.3f}",
                f"{m['combined_index']:.3f}", vol_str,
                frag_str, fill_str,
            ])
        total_vol_str = f"{summary['total_volume']:.3f}" if summary['total_volume'] is not None else "TBD"
        total_label = 'TOTAL' if not any_cluster else 'TOTAL (confirmed only)'
        cell_text.append([
            total_label, scale_str, f"{summary['total_area']:.3f}",
            f"{summary['total_core']:.3f}", f"{summary['total_recovered']:.3f}",
            f"{summary['total_holes']}",
            f"{summary['avg_circularity']:.3f}", f"{summary['avg_intensity_idx']:.3f}",
            f"{summary['total_combined']:.3f}", total_vol_str,
            "-", "-",
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
        for row_idx, m in enumerate(measurements, start=1):
            if m['is_overgrown'] and not m['is_cluster_origin']:
                table[row_idx, RECOVERED_COL].set_text_props(fontweight='bold', color='red')
        for row_idx in cluster_row_indices:
            for col_idx in range(len(col_labels)):
                table[row_idx, col_idx].set_text_props(fontweight='bold', color='magenta')
        caption_lines = []
        if any_cluster:
            caption_lines.append(
                "CLUSTER? = CLUSTER - UNCONFIRMED, requires visual check (magenta square marker in the "
                "result image): a real, Otsu-crossing but fragmented signal that only clears the size "
                "floor when its scattered pieces are combined. Testing found no reliable automatic way "
                "to tell this apart from a dense field of dust/fold-line specks, so it is NEVER counted "
                "automatically - excluded from the TOTAL row and the CSV master log. Area/fragment-count/"
                "fill-ratio are still reported above so a human has the numbers needed to confirm or "
                "reject it by eye.")
        if any_small:
            caption_lines.append(f"* below {CONFIDENT_AREA_MM2}mm2 (shown in orange in the result image) - worth a visual check")
        if any_overgrown:
            caption_lines.append(f"! overgrowth flag: recovered area exceeds {OVERGROWTH_RATIO * 100:.0f}% of core area (red outline in the result image) - ratio shown in parentheses")
        if caption_lines:
            axes[1, 1].text(
                0.02, 0.88, "\n".join(caption_lines),
                transform=axes[1, 1].transAxes, fontsize=7, style='italic', verticalalignment='top'
            )
    else:
        axes[1, 1].text(0.02, 0.9, "No aggregates detected.\nTry a different photo or check lighting.",
                         transform=axes[1, 1].transAxes, fontsize=10, verticalalignment='top')

    plt.tight_layout()
    plt.show()


def display_failure(reason, filename=None):
    name_part = f"{filename}\n\n" if filename else ""
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.suptitle("Calibration failed", fontsize=12, fontweight='bold')
    ax.axis('off')
    ax.text(0.5, 0.5, f"{name_part}{reason}", ha='center', va='center', fontsize=11)
    plt.tight_layout()
    plt.show()


def display_comparison_table(results, filenames):
    col_labels = ['Photo', 'Scale\n(mm/px)', 'Area\n(mm2)', 'Core area\n(mm2)',
                  'Recovered area\n(mm2)', 'Holes\n(count)',
                  'Circularity\n(unitless)', 'Relative intensity\nindex (unitless)',
                  'Combined index\n(mm2)', 'Volume\n(mm3)']
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
            name_str += f" !({summary['max_overgrowth_ratio']:.1f}x)"
            overgrown_row_indices.append(len(cell_text) + 1)  # +1: header is row 0
        elif summary['has_small']:
            name_str += "*"
        # V4.14: cluster-origin candidates are excluded from this row's
        # confirmed totals (same as the per-photo table) - flagged here so a
        # count of 0 in a batch view doesn't silently hide a candidate that
        # still needs a visual look in that photo's own result window.
        if summary['has_unconfirmed_cluster']:
            name_str += f" [{summary['unconfirmed_count']} CLUSTER?]"
        cell_text.append([
            name_str, scale_str, f"{summary['total_area']:.3f}",
            f"{summary['total_core']:.3f}", f"{summary['total_recovered']:.3f}",
            f"{summary['total_holes']}",
            f"{summary['avg_circularity']:.3f}", f"{summary['avg_intensity_idx']:.3f}",
            f"{summary['total_combined']:.3f}", vol_str,
        ])
        numeric_rows.append(summary)

    if numeric_rows or scales:
        areas = [s['total_area'] for s in numeric_rows]
        cores = [s['total_core'] for s in numeric_rows]
        recovered = [s['total_recovered'] for s in numeric_rows]
        holes = [s['total_holes'] for s in numeric_rows]
        circs = [s['avg_circularity'] for s in numeric_rows]
        intens = [s['avg_intensity_idx'] for s in numeric_rows]
        combined = [s['total_combined'] for s in numeric_rows]
        vols = [s['total_volume'] for s in numeric_rows if s['total_volume'] is not None]

        def stat_or_dash(fn, vals, nd=3):
            return f"{fn(vals):.{nd}f}" if vals else '-'

        cell_text.append([
            'AVG', stat_or_dash(np.mean, scales, 5), stat_or_dash(np.mean, areas),
            stat_or_dash(np.mean, cores), stat_or_dash(np.mean, recovered),
            stat_or_dash(np.mean, holes, 1), stat_or_dash(np.mean, circs),
            stat_or_dash(np.mean, intens), stat_or_dash(np.mean, combined),
            stat_or_dash(np.mean, vols) if vols else "TBD",
        ])
        cell_text.append([
            'STD', stat_or_dash(np.std, scales, 5), stat_or_dash(np.std, areas),
            stat_or_dash(np.std, cores), stat_or_dash(np.std, recovered),
            stat_or_dash(np.std, holes, 1), stat_or_dash(np.std, circs),
            stat_or_dash(np.std, intens), stat_or_dash(np.std, combined),
            stat_or_dash(np.std, vols) if vols else "TBD",
        ])

    fig, ax = plt.subplots(figsize=(max(11, 1.5 * len(filenames) + 5), 1.5 + 0.4 * len(cell_text)))
    fig.suptitle(f"Batch comparison - {len(filenames)} photo(s)", fontsize=12, fontweight='bold')
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
    for row_idx in overgrown_row_indices:
        table[row_idx, RECOVERED_COL].set_text_props(fontweight='bold', color='red')

    caption_lines = []
    if any(s.get('has_unconfirmed_cluster') for s in numeric_rows):
        caption_lines.append("[N CLUSTER?] = N cluster-origin candidate(s) found but NOT counted in this "
                              "row's totals - fragmented signal that only clears the size floor when "
                              "combined, requires a visual check in that photo's own result window "
                              "(magenta marker there)")
    if any(s.get('has_small') for s in numeric_rows):
        caption_lines.append(f"* includes an aggregate below {CONFIDENT_AREA_MM2}mm2 - worth a visual check in that photo's result window")
    if any(s.get('has_overgrown') for s in numeric_rows):
        caption_lines.append("! includes an overgrowth-flagged aggregate - ratio shown is that photo's worst (max) recovered/core ratio; see per-photo window for the full breakdown")
    if caption_lines:
        ax.text(0.02, 0.02, "\n".join(caption_lines),
                transform=ax.transAxes, fontsize=8, style='italic')

    plt.tight_layout()
    plt.show()


# =============================================
# MAIN
# =============================================

if __name__ == "__main__":
    print("=" * 60)
    print(f"PLASMA AGGREGATION ANALYSIS  v{VERSION}  [build: {BUILD_TAG}]")
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
            display_failure(result['reason'], name)
            continue

        result = analyze_image(image, params['reference_mm'], params['color_mode'], params['thickness_mm'])
        all_results.append(result)

        if not result['success']:
            print(f"CALIBRATION FAILED: {result['reason']}")
            append_photo_to_master_csv(name, params, result)
            display_failure(result['reason'], name)
            continue

        print(f"Calibration: {result['line_px']:.0f}px = {params['reference_mm']}mm "
              f"(1px = {result['mm_per_px']:.5f}mm)")
        print(f"Otsu threshold (core, flattened): {result['otsu_threshold']:.0f}")
        print(f"Loose threshold (hysteresis, raw): {result['loose_threshold']:.0f}")
        confirmed_count = sum(1 for m in result['measurements'] if not m['is_cluster_origin'])
        cluster_count = sum(1 for m in result['measurements'] if m['is_cluster_origin'])
        print(f"Aggregates found: {confirmed_count} confirmed"
              + (f" + {cluster_count} unconfirmed cluster candidate(s)" if cluster_count else ""))
        for m in result['measurements']:
            if m['is_cluster_origin']:
                print(f"  #{m['id']} [CLUSTER? - requires visual check, NOT counted in totals/CSV]: "
                      f"total={m['true_area_mm2']:.4f}mm2  "
                      f"fragments={m['cluster_num_fragments']}  "
                      f"fill_ratio={m['cluster_fill_ratio']:.3f}")
            else:
                print(f"  #{m['id']}: core={m['core_area_mm2']:.4f}mm2  "
                      f"hysteresis-recovered={m['hysteresis_area_mm2']:.4f}mm2  "
                      f"total={m['true_area_mm2']:.4f}mm2")

        # DEBUG-ONLY (V4.9, temporary), PORTED from B1.py: reports Otsu
        # foreground coverage raw vs. post-morphology, and why the largest
        # rejected candidate blob(s) were thrown out - see V4.9 changelog.
        dd = result['detection_diagnostics']
        print(f"Otsu foreground coverage: {dd['raw_foreground_fraction']*100:.2f}% of frame RAW "
              f"(before morphology) -> {dd['foreground_fraction']*100:.2f}% AFTER morphology")
        if dd['rejected_top5']:
            print("Largest rejected candidate blob(s) (why they didn't count):")
            for area_mm2, reason in dd['rejected_top5']:
                print(f"    {area_mm2:.3f}mm2 - {reason}")
        elif len(result['measurements']) == 0:
            if dd['raw_foreground_fraction'] > 0:
                print("    Otsu DID flag some pixels, but morphology ate all of them - "
                      "the content found was thin/fine (e.g. traced outline or bubble rims), "
                      "not a solid blob. The real aggregate may have too little contrast to compete "
                      "with those thin marks for Otsu's threshold placement.")
            else:
                print("    Otsu found literally 0% foreground pixels anywhere in the frame - "
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
