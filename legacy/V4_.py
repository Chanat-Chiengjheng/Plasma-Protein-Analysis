import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

VERSION = "4.4"
BUILD_TAG = "otsu-line-exclusion-fix"  # change this string every time this
                                                 # file changes, so it's obvious at a
                                                 # glance (printed every run, no need
                                                 # to compare wording by eye) whether
                                                 # the file actually running is the
                                                 # one just downloaded

# =============================================
# CHANGELOG (quick reference)
# =============================================
# V4.0 - first version
# V4.1 - FIXED: real red calibration line was being rejected
#        CHANGED: line shape check loosened, length must be 3x thickness
#        (was 5x)
# V4.2 - FIXED: green-background photo produced one giant wrong "aggregate"
#        CHANGED: added a correction for uneven lighting before
#        thresholding (see "illumination flattening" below for why)
#        WATCH OUT: on the older blue photo, this also shifted the
#        measured area from 52.2 to 44.5 mm2 (about 15% smaller) - same
#        aggregate found, slightly stricter on faint edges. See full
#        notes below if you want the reasoning.
# V4.3 - ADDED: batch analysis. The popup now accepts multiple images at
#        once. Each photo still gets its own result window, same as
#        before - and if more than one photo was selected, one extra
#        comparison-table window appears at the end. Running with a
#        single photo behaves exactly like V4.2: no comparison window,
#        no behavior change.
#        CHANGED: the info panel (bottom-right of each result window)
#        is now a real rendered table instead of a block of monospace
#        text - same numbers, easier to read.
#        ASSUMPTION: reference length, calibration line's color, and
#        thickness are shared across the whole batch (one popup, one set
#        of answers, applied to every photo). Flag this if a batch ever
#        needs different values per photo.
# V4.4 - CHANGED (real-batch feedback): both tables now include the
#        calibration scale (mm/px) for each photo, since it can
#        legitimately vary shot to shot even with the same reference
#        length entered. Every column header is now a full word with its
#        unit shown - no more "Circ." / "Int. idx" / unitless "Volume".
#
#        missed_aggregate_fix (same V4.4, not bumped to V4.5 per request):
#        a real batch photo showed a small, thin aggregate getting
#        completely missed - no contour at all, despite being clearly
#        visible in the original image.
#        ROUND 1 (kernel resolution scaling): BLUR_SIZE and MORPH_SIZE
#        were fixed pixel counts, made resolution-relative instead. This
#        was a real inconsistency worth fixing, but testing against the
#        actual photo afterward showed it was NOT the cause of this bug -
#        the small aggregate still got missed.
#        ROUND 2 (confirmed cause, area floor): added a console print for
#        any contour rejected for being too small, which confirmed the
#        real cause - the small aggregate (0.49-0.57mm2 in two test
#        photos) was getting cut by MINIMUM_AREA_MM2, not eroded by
#        morphology. Area alone can't fix this cleanly though: ordinary
#        dust specks in the same photos measured up to 0.50mm2 too -
#        nearly identical size to the real aggregate.
#        ROUND 3 (shape as tie-breaker, tried and reverted): the one
#        confirmed real aggregate had lower circularity (0.13, more
#        fibrous/irregular) than sampled dust specks (0.18-0.60, more
#        compact). Tried auto-including anything below the area floor
#        that was elongated enough. Tested against a second real photo
#        and it backfired - 4 dust specks were irregular enough to pass
#        too, including one the exact same size (0.498mm2) as the
#        confirmed real aggregate. Reverted: nothing below MINIMUM_AREA_MM2
#        is auto-included. FINAL STATE: any contour in the 0.1-1.0mm2
#        range is fully reported (area, circularity, pixel location) but
#        never auto-counted, so a real small aggregate in that range can
#        be found and manually verified against the photo, rather than
#        the code guessing and risking false positives elsewhere. This
#        remains a known limitation, not a solved problem - confirmed by
#        testing two different approaches against real photos, not
#        theorized.
#        ROUND 4 (solidity tried, also failed; floor lowered instead):
#        tried solidity (area / convex hull area) as a different shape
#        tie-breaker - a winding fiber should have lower solidity than a
#        compact dust speck even if both have low circularity. Tested
#        against the same real photos: the confirmed real aggregate's
#        solidity (0.525) sat inside the dust specks' range (0.463-0.976),
#        not outside it. Two different shape properties have now failed
#        to separate them. Decision: stop searching for a perfect
#        automatic separator and make the trade-off explicit instead.
#        MINIMUM_AREA_MM2 lowered from 1.0 to 0.4 (below the smallest
#        confirmed real value, 0.4909mm2) - this now counts real small
#        aggregates, accepting that the one dust speck that measured
#        0.498mm2 in testing would count too (roughly 1 in 21 sampled
#        dust specks crossed this new floor). CONFIDENT_AREA_MM2 (1.0)
#        kept as the boundary for what still counts as unambiguous -
#        anything between the new floor and this gets a visible "*" in
#        both tables and is drawn in orange instead of green in the
#        result image, so it's never confused with a confident detection
#        and stays easy to spot-check by eye.
#
#        otsu_line_bias_fix (same V4.4, found testing a real batch):
#        a photo with a real but tiny aggregate signal relative to the
#        calibration line was finding 0 aggregates despite a real one
#        being clearly visible. Confirmed by testing: Otsu was computing
#        its threshold using the WHOLE flattened image, including the
#        calibration line's pixels. With the line's dark pixels in the
#        mix, Otsu picked 89 - below the background's own value (125-132
#        after flattening) - so the entire background read as foreground,
#        got caught by the border-artifact guard, and the real tiny
#        signal was discarded along with it. Excluding the line's pixels
#        from the threshold computation (not just from the final result)
#        shifted Otsu to 137 on the same photo - correctly above
#        background. FOUND WHILE TESTING THIS: the line's color-detected
#        region (used for the exclusion box) badly undershot its actual
#        grayscale brightness footprint - a translucent marker can have a
#        bright "halo" well beyond where it reads as saturated red. With
#        the corrected (higher) threshold, that halo crossed into
#        foreground and got counted as an 18.6mm2 fake aggregate - the
#        line itself, not a real one. Fixed by excluding whatever
#        connected blob in the actual thresholded result touches the
#        line's known center, however large its real footprint turns out
#        to be, instead of trusting a fixed geometric padding guess.
#        Regression-tested against every previously-working photo (no
#        change in any of their numbers) plus this one (0 -> 1 correct
#        aggregate, line no longer miscounted).
# =============================================

# =============================================
# PLASMA PROTEIN AGGREGATION ANALYSIS TOOL
# Version 4.0 — Reproducible / No Code Editing Required
# =============================================
# SCOPE: reflected-light setup only (current lab setup).
# The backlight version is a separate future script.
#
# WHAT CHANGED FROM V3 (for the report — why we did / didn't do things):
#
#   Display
#     - V3: two pop-up windows, six panels, summary text crammed onto
#       the result image.
#     - V4: one window, 2x2 grid (original / calibration / clean result /
#       dedicated text-only info panel). Grayscale/blurred/binary steps
#       still happen internally, just aren't shown.
#
#   Aggregate threshold
#     - V3: BRIGHT_PERCENTILE, manually retuned almost every run because
#       the right percentile depends on how much area actually aggregated.
#     - V4: Otsu's method (cv2.THRESH_OTSU). Data-driven from the image's
#       own brightness histogram instead of assuming a fixed area fraction.
#     - Otsu's known failure mode is a histogram with more than two
#       brightness populations (e.g. a bright solution-edge reflection
#       sitting above the aggregate). That's the reason V1's original Otsu
#       attempt was dropped.
#     - Standard procedure going forward: use a larger solution so the
#       bright edge sits physically separated from the aggregate, then
#       paint over that edge so it doesn't compete as a third brightness
#       population. This is a procedure fix, not a code fix — no special
#       masking logic was added for the painted-over region, because
#       painted ink reads dark and naturally falls into the background
#       side of Otsu's threshold on its own.
#     - Tested directly against a real photo (an older, smaller-solution
#       sample, a harder case than what the new procedure produces) —
#       the aggregate was detected as one clean piece, unaffected by the
#       painted-over patches sitting nearby. Not yet seen on a photo from
#       the new larger-solution procedure specifically, but that setup
#       only increases the safety margin versus the case already tested.
#
#   Calibration line detection
#     - V3: 9 fixed BGR channel thresholds (3 colors x 3 channels).
#     - V4: HSV hue-band detection. Hue is far less sensitive to overall
#       brightness than raw BGR values, and one hue-band check now serves
#       all three calibration line colors instead of three separately
#       tuned constant sets.
#
#   Calibration line shape filter
#     - V3: MIN_LINE_LENGTH_PX / MIN_THICKNESS_PX, fixed pixel values that
#       silently break if camera distance/zoom changes.
#     - V4: a length-to-thickness aspect ratio (it must look like a line,
#       not a blob) plus a length-relative-to-image-diagonal floor. Scales
#       naturally with zoom. If nothing clears both checks, V4 reports a
#       clear calibration failure instead of guessing a wrong scale
#       (V3's old fallback-to-601px behavior is intentionally removed).
#
#   Noise filters
#     - BLUR_SIZE / MORPH_SIZE: kept as fixed internal constants, same
#       values as V3. Not exposed — but MORPH_SIZE is flagged below for a
#       validation test, since morphological closing can erase a real
#       hole that's narrower than its kernel before hole-subtraction ever
#       sees it. Worth checking against a known image with a real small
#       hole before fully trusting hole_area_px.
#     - MINIMUM_AREA_MM2: now a fixed constant instead of a per-shot tuned
#       value. This is contingent on the Task 3 fixture and lighting work
#       actually reducing noise at the source — if real images still show
#       more noise than this constant can absorb, that's a sign Task 3
#       needs more work, not a reason to make this tunable again.
#
#   New measurements
#     - Volume: optional aggregate thickness (mm) -> area x thickness.
#       Left as "TBD" if no thickness is given. Never errors.
#     - Intensity index (Phase 4 / professor's request): two of the five
#       suggested parameters were implemented —
#         relative_intensity_index = (aggregate_mean - background_mean)
#                                     / (255 - background_mean)
#         combined_index = true_area_mm2 * relative_intensity_index
#       Both are self-correcting against shot-to-shot lighting drift,
#       since they're computed relative to the background of that same
#       photo, and combined_index directly answers the professor's stated
#       example (equal area, different intensity, should not score equal).
#       NOT implemented, and why:
#         - raw mean grayscale intensity: an absolute number, vulnerable
#           to the same lighting-drift problem the relative index fixes.
#         - integrated intensity (sum instead of mean): mathematically
#           almost the same as area x raw mean, so it's redundant with
#           combined_index without adding new information.
#         - relative darkness index: for reflected light this is just
#           1 - relative_intensity_index, the same number flipped. It
#           should get real meaning once the backlight version exists,
#           where "light blocked passing through" is a genuinely
#           different physical measurement.
#
#   New: illumination flattening (uneven lighting across the frame)
#     - A real green-background photo exposed a failure mode Otsu can't
#       handle on its own: if lighting brightness varies noticeably across
#       the frame, a single global threshold can land inside the
#       background's own brightness range, misclassifying the brighter
#       part of the background as foreground. This isn't a green-vs-blue
#       issue specifically — checked directly, the aggregate was still
#       genuinely brighter than background on average; the background
#       itself just weren't uniformly lit (a 41-point grayscale spread
#       within "background" alone in the failing photo).
#     - Fix: estimate the slow-varying lighting trend with a large
#       Gaussian blur (big enough to mostly ignore the aggregate/line as
#       small local features) and subtract it out before thresholding.
#       This compares each pixel to its own local expected background
#       instead of one global brightness level, so a lighting gradient
#       stops mattering. Checked against the already-working blue photo:
#       still finds the same single aggregate in the same place, but the
#       measured area shifted from 52.2mm2 to 44.5mm2 (about 15% smaller).
#       That's not nothing — it's because comparing each pixel to its own
#       local neighborhood is a stricter test for faint, translucent
#       fringe pixels than one global threshold was. The aggregate's core
#       is unaffected; its faint outer edge is where the difference comes
#       from. Worth deciding whether that trade-off (fixes the lighting
#       bug, costs a somewhat more conservative boundary on faint edges)
#       is acceptable, rather than assuming it's free.
#     - The original (non-flattened) grayscale image is still what's used
#       for the intensity-index and volume calculations afterward — the
#       flattening is only used to decide where the aggregate's boundary
#       is, not to replace the real measured brightness values once that
#       boundary is known.
#
#   New: full-frame border-artifact guard
#     - A defensive check, found necessary while validating the fix
#       above: a few stray pixels right at the image edge could
#       occasionally make cv2.findContours trace the entire image frame
#       as a spurious "outer contour". Any contour whose bounding box
#       covers almost the entire image is now excluded before it ever
#       reaches the aggregate list.
#
#   Parameter input
#     - V3: hardcoded constants at the top of the script.
#     - V4: a popup form (tkinter) is tried first — image file (browse
#       button), reference length, calibration line's color (dropdown,
#       so no typos are possible), and optional aggregate thickness.
#       If the popup can't launch for any reason, the script automatically
#       falls back to the same four questions asked one at a time in the
#       terminal, with case/whitespace-insensitive matching on the color
#       answer and a re-prompt instead of a crash on invalid input.
# =============================================


# =============================================
# INTERNAL CONSTANTS — fixed, not shown to the user.
# Only change these here if testing on real images shows a problem.
# =============================================

# --- Otsu thresholding (replaces BRIGHT_PERCENTILE) ---
# BLUR_SIZE and MORPH_SIZE used to be fixed pixel counts (15 and 3). A real
# batch photo showed a small, thin aggregate getting completely missed -
# no contour at all, despite being clearly visible and bright in the
# original image. A fixed-pixel morphological kernel doesn't scale with
# image resolution the way the calibration-line filters already do, so a
# thin/small real feature is more vulnerable to being eroded away on some
# resolutions than others. Both are now a fraction of image size instead,
# matching the same fix already applied to the calibration-line shape
# filter. NOT YET CONFIRMED against the actual photo that showed the
# problem - only the cropped result screenshot was available, not the
# original file - so this addresses the most likely structural cause, but
# testing against the real photo is the next step to confirm it.
BLUR_SIZE_FRACTION  = 0.0043  # ~15px at the ~3456px-wide photos tested so far
MORPH_SIZE_FRACTION = 0.0009  # ~3px at the same reference resolution
# NOTE: see "Noise filters" comment above re: morphological closing + small real holes.

# --- Illumination flattening (corrects uneven lighting before thresholding) ---
ILLUMINATION_KERNEL_FRACTION = 0.5   # fraction of the shorter image dimension
                                       # used as the blur kernel for estimating
                                       # the lighting trend. Large enough to stay
                                       # mostly clear of the aggregate's own
                                       # footprint; see "illumination flattening"
                                       # comment above.
BORDER_ARTIFACT_AREA_FRACTION = 0.9   # a contour covering more than this
                                       # fraction of the whole image is treated
                                       # as a border artifact, not a real aggregate

# --- Calibration line color detection (HSV, replaces 9 BGR thresholds) ---
HUE_CENTERS = {'RED': 0, 'GREEN': 60, 'BLUE': 120}  # OpenCV hue scale is 0-180
HUE_TOLERANCE  = 15
SATURATION_MIN = 80
VALUE_MIN      = 40

# --- Calibration line shape filter (ratio-based, replaces fixed px cutoffs) ---
MIN_ASPECT_RATIO         = 3.0   # length must be at least 3x the thickness
                                   # (lowered from an initial guess of 5.0 after
                                   # a real photo measured 3.91 — the painted
                                   # line is a thick capsule, not a thin stripe)
MIN_LINE_LENGTH_FRACTION = 0.05   # line must span at least 5% of the image diagonal

# --- Aggregate noise filter ---
# Two shape-based attempts to auto-separate real small aggregates from dust
# specks were tried and both failed real-photo testing: circularity let
# dust through as false positives (including one the exact same size as a
# confirmed real aggregate); solidity put the real aggregate's value
# inside the dust speck range, not outside it. The actual numbers: the
# smallest confirmed real aggregate measured 0.4909mm2; the largest sampled
# dust speck measured 0.498mm2 - 0.007mm2 apart. No area threshold can
# separate those two values. Decision: the floor is lowered below both
# confirmed real values, accepting that occasionally a dust speck in that
# same narrow range (roughly 1 in 21 sampled) may get counted too. Nothing
# in the lowered range is silently smoothed over - it's marked with "*" in
# both tables (see display_results / display_comparison_table) specifically
# so it's never confused with a confident, unambiguous detection.
MINIMUM_AREA_MM2 = 0.4    # the real counting floor now
CONFIDENT_AREA_MM2 = 1.0  # above this, no "*" mark - this was the old floor,
                           # kept as the boundary for what still counts as
                           # an unambiguous, no-need-to-double-check size
BORDERLINE_AREA_MM2 = 0.05  # true noise floor - nothing below this is even
                              # reported, let alone counted (single/few-pixel
                              # JPEG artifacts, not real candidates either way)

# BGR draw colors, just for the calibration overlay visualization
DRAW_COLORS = {'RED': (0, 0, 255), 'GREEN': (0, 255, 0), 'BLUE': (255, 0, 0)}


# =============================================
# ILLUMINATION FLATTENING (corrects uneven lighting before thresholding)
# =============================================

def make_odd(n):
    """Gaussian blur kernels must be odd-sized."""
    n = int(n)
    return n if n % 2 == 1 else n + 1


def flatten_illumination(gray):
    """Removes a slow-varying lighting gradient so Otsu compares each pixel
    to its own local expected background instead of one global brightness
    level. See the 'illumination flattening' comment at the top of this
    file for why this was added and what it was tested against."""
    h, w = gray.shape
    kernel_size = make_odd(min(h, w) * ILLUMINATION_KERNEL_FRACTION)
    illumination = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    diff = gray.astype(np.int16) - illumination.astype(np.int16)
    return np.clip(diff + 128, 0, 255).astype(np.uint8)


# =============================================
# CALIBRATION LINE DETECTION
# =============================================

def get_color_mask(hsv_image, color_mode):
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

    return hue_mask & (s >= SATURATION_MIN) & (v >= VALUE_MIN)


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
    selected_paths = []  # holds the real full paths; image_path_var only holds a display string
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
    """Run the full V4 analysis on a loaded BGR image. Returns a dict
    with everything display_results() needs, or a failure reason."""

    # ---- Calibration (HSV + ratio-based shape filter) ----
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

    # ---- Calibration line exclusion mask (computed early - used twice) ----
    # Confirmed by testing, not just theory: if the painted line happens to be
    # brighter than a dark background, Otsu can classify the line itself as
    # foreground, and its area is easily large enough to pass MINIMUM_AREA_MM2.
    # The area filter alone is not a reliable enough safety net on its own, so
    # the line's region is explicitly carved out before aggregate contours are
    # ever found, regardless of its brightness or area.
    # SEPARATE, also-confirmed issue: the line's dark pixels can also skew
    # which threshold Otsu picks in the first place, when the real aggregate
    # signal is small/sparse relative to the line. Tested directly: on one
    # real photo, Otsu picked 89 with the line included, 137 with it
    # excluded - 137 correctly sits above the background's own brightness,
    # 89 did not, and that's the difference between finding the real
    # aggregate and the whole frame being misread as one giant false
    # blob (caught and discarded by the border-artifact guard, but taking
    # the real signal down with it). So the line is now excluded from the
    # pixels Otsu sees, not just from the final contour result.
    line_exclusion_mask = np.zeros(gray.shape, dtype=np.uint8)
    (lcx, lcy), (lw, lh), langle = line['rect']
    padded_rect = ((lcx, lcy), (lw + 20, lh + 20), langle)  # small safety margin
    line_box = np.int32(cv2.boxPoints(padded_rect))
    cv2.fillPoly(line_exclusion_mask, [line_box], 255)

    # ---- Otsu threshold, computed without the calibration line's pixels ----
    otsu_threshold, _ = cv2.threshold(
        blurred[line_exclusion_mask == 0], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    _, binary = cv2.threshold(blurred, otsu_threshold, 255, cv2.THRESH_BINARY)

    # ---- Noise removal ----
    kernel = np.ones((morph_size, morph_size), np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    # ---- Exclude the calibration line's own region from the final result ----
    # The padded geometric box (line_exclusion_mask) is only an approximation
    # from the HSV-detected core - confirmed by testing that it isn't enough
    # on its own: a translucent marker can have a brightness "halo" well
    # beyond its color-detected extent (one real photo: color-detected box
    # was 463x1080px, but the halo's actual bright footprint was 1069x1520px
    # - the line itself got counted as an 18.6mm2 fake "aggregate" because
    # the fixed padding fell far short). Instead, whatever connected blob in
    # the actual thresholded result touches the line's known center gets
    # excluded entirely, however large its real footprint turns out to be -
    # this adapts to the real halo instead of guessing a margin.
    line_center_point = (int(lcx), int(lcy))
    pre_exclusion_contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in pre_exclusion_contours:
        if cv2.pointPolygonTest(c, line_center_point, False) >= 0:
            cv2.drawContours(cleaned, [c], -1, 0, -1)
    cleaned[line_exclusion_mask == 255] = 0  # geometric box too, as a floor

    # ---- Contours + holes ----
    contours, hierarchy = cv2.findContours(cleaned, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    minimum_area_px = MINIMUM_AREA_MM2 / mm2_per_px2
    confident_area_px = CONFIDENT_AREA_MM2 / mm2_per_px2
    borderline_area_px = BORDERLINE_AREA_MM2 / mm2_per_px2

    outer_contours = []
    hole_contours = []
    small_indices = set()  # counted, but below the old confident floor - gets a "*" mark
    border_artifact_area_px = BORDER_ARTIFACT_AREA_FRACTION * img_h * img_w
    if hierarchy is not None:
        for i, contour in enumerate(contours):
            parent_idx = hierarchy[0][i][3]
            area_px = cv2.contourArea(contour)
            if parent_idx == -1:
                if area_px > border_artifact_area_px:
                    pass  # border artifact, never counted
                elif area_px > minimum_area_px:
                    outer_contours.append((contour, i))
                    if area_px <= confident_area_px:
                        small_indices.add(i)
                elif area_px > borderline_area_px:
                    # Logged even though excluded - if this still looks like
                    # it should have counted, that's the number to check
                    # against the photo.
                    area_mm2 = area_px * mm2_per_px2
                    print(f"  (below the counting floor: area={area_mm2:.4f}mm2, "
                          f"floor is {MINIMUM_AREA_MM2}mm2)")
            else:
                hole_contours.append((contour, parent_idx))

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

        # true-area mask for this aggregate (outer minus its own holes)
        agg_mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(agg_mask, [contour], -1, 255, -1)
        for hc in my_holes:
            cv2.drawContours(agg_mask, [hc], -1, 0, -1)
        agg_pixels = gray[agg_mask == 255]
        aggregate_mean = float(agg_pixels.mean()) if agg_pixels.size > 0 else 0.0

        denom = max(255.0 - background_mean, 1e-6)
        relative_intensity_index = max(0.0, (aggregate_mean - background_mean) / denom)
        combined_index = true_area_mm2 * relative_intensity_index

        volume_mm3 = true_area_mm2 * thickness_mm if thickness_mm is not None else None

        measurements.append({
            'id': i + 1,
            'true_area_px': true_area_px,
            'true_area_mm2': true_area_mm2,
            'num_holes': len(my_holes),
            'perimeter_mm': perimeter_mm,
            'circularity': circularity,
            'is_small': contour_idx in small_indices,
            'center_x': center_x, 'center_y': center_y,
            'bbox_x': x, 'bbox_y': y, 'bbox_w': w, 'bbox_h': h,
            'background_mean': background_mean,
            'aggregate_mean': aggregate_mean,
            'relative_intensity_index': relative_intensity_index,
            'combined_index': combined_index,
            'volume_mm3': volume_mm3,
        })

    # ---- Clean result image (no text block baked into it) ----
    # Small aggregates (below CONFIDENT_AREA_MM2) are drawn in orange instead
    # of green, so they're visibly flagged as "smaller, worth a second look"
    # right in the image itself, not just as a number in a table.
    result_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    confident_contours = [c for c, i in outer_contours if i not in small_indices]
    small_contours = [c for c, i in outer_contours if i in small_indices]
    cv2.drawContours(result_image, confident_contours, -1, (0, 255, 0), 2)
    cv2.drawContours(result_image, small_contours, -1, (0, 165, 255), 2)
    cv2.drawContours(result_image, [hc for hc, _ in hole_contours], -1, (0, 0, 255), 2)
    for m in measurements:
        dot_color = (0, 165, 255) if m['is_small'] else (0, 255, 255)
        label = f"#{m['id']}*" if m['is_small'] else f"#{m['id']}"
        cv2.circle(result_image, (m['center_x'], m['center_y']), 6, dot_color, -1)
        cv2.putText(result_image, label, (m['bbox_x'], m['bbox_y'] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, dot_color, 2)

    return {
        'success': True,
        'image': image, 'cal_result': cal_result, 'result_image': result_image,
        'color_mode': color_mode, 'reference_mm': reference_mm,
        'line_px': line_px, 'mm_per_px': mm_per_px,
        'otsu_threshold': otsu_threshold,
        'measurements': measurements,
        'background_mean': background_mean,
        'thickness_mm': thickness_mm,
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
    return {
        'count': len(measurements),
        'total_area': sum(m['true_area_mm2'] for m in measurements),
        'total_holes': sum(m['num_holes'] for m in measurements),
        'avg_circularity': sum(m['circularity'] for m in measurements) / len(measurements),
        'avg_intensity_idx': sum(m['relative_intensity_index'] for m in measurements) / len(measurements),
        'total_combined': sum(m['combined_index'] for m in measurements),
        'total_volume': total_volume,
        'has_small': any(m['is_small'] for m in measurements),
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

        col_labels = ['#', 'Scale\n(mm/px)', 'Area\n(mm2)', 'Holes\n(count)',
                      'Circularity\n(unitless)', 'Relative intensity\nindex (unitless)',
                      'Combined index\n(mm2)', 'Volume\n(mm3)']
        scale_str = f"{r['mm_per_px']:.5f}"
        any_small = any(m['is_small'] for m in measurements)
        cell_text = []
        for m in measurements:
            vol_str = f"{m['volume_mm3']:.3f}" if m['volume_mm3'] is not None else "TBD"
            id_str = f"{m['id']}*" if m['is_small'] else f"{m['id']}"
            cell_text.append([
                id_str, scale_str, f"{m['true_area_mm2']:.3f}", f"{m['num_holes']}",
                f"{m['circularity']:.3f}", f"{m['relative_intensity_index']:.3f}",
                f"{m['combined_index']:.3f}", vol_str,
            ])
        total_vol_str = f"{summary['total_volume']:.3f}" if summary['total_volume'] is not None else "TBD"
        cell_text.append([
            'TOTAL', scale_str, f"{summary['total_area']:.3f}", f"{summary['total_holes']}",
            f"{summary['avg_circularity']:.3f}", f"{summary['avg_intensity_idx']:.3f}",
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
        if any_small:
            axes[1, 1].text(
                0.02, 0.88, f"* below {CONFIDENT_AREA_MM2}mm2 (shown in orange in the result image) - worth a visual check",
                transform=axes[1, 1].transAxes, fontsize=8, style='italic', verticalalignment='top'
            )
    else:
        axes[1, 1].text(0.02, 0.9, "No aggregates detected.\nTry a different photo or check lighting.",
                         transform=axes[1, 1].transAxes, fontsize=10, verticalalignment='top')

    plt.tight_layout()
    plt.show()


# =============================================
# MAIN
# =============================================

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
    col_labels = ['Photo', 'Scale\n(mm/px)', 'Area\n(mm2)', 'Holes\n(count)',
                  'Circularity\n(unitless)', 'Relative intensity\nindex (unitless)',
                  'Combined index\n(mm2)', 'Volume\n(mm3)']
    cell_text = []
    numeric_rows = []   # photos with at least 1 aggregate - for area/circ/etc. averages
    scales = []          # every successfully-calibrated photo - scale is valid even at 0 aggregates

    for name, r in zip(filenames, results):
        if not r['success']:
            cell_text.append([name, 'FAILED', '-', '-', '-', '-', '-', '-'])
            continue

        scale_str = f"{r['mm_per_px']:.5f}"
        scales.append(r['mm_per_px'])

        summary = compute_photo_summary(r)
        if summary is None:
            cell_text.append([name, scale_str, '0', '0', '-', '-', '-', '-'])
            continue
        vol_str = f"{summary['total_volume']:.3f}" if summary['total_volume'] is not None else "TBD"
        name_str = f"{name}*" if summary['has_small'] else name
        cell_text.append([
            name_str, scale_str, f"{summary['total_area']:.3f}", f"{summary['total_holes']}",
            f"{summary['avg_circularity']:.3f}", f"{summary['avg_intensity_idx']:.3f}",
            f"{summary['total_combined']:.3f}", vol_str,
        ])
        numeric_rows.append(summary)

    if numeric_rows or scales:
        areas = [s['total_area'] for s in numeric_rows]
        holes = [s['total_holes'] for s in numeric_rows]
        circs = [s['avg_circularity'] for s in numeric_rows]
        intens = [s['avg_intensity_idx'] for s in numeric_rows]
        combined = [s['total_combined'] for s in numeric_rows]
        vols = [s['total_volume'] for s in numeric_rows if s['total_volume'] is not None]

        def stat_or_dash(fn, vals, nd=3):
            return f"{fn(vals):.{nd}f}" if vals else '-'

        cell_text.append([
            'AVG', stat_or_dash(np.mean, scales, 5), stat_or_dash(np.mean, areas),
            stat_or_dash(np.mean, holes, 1), stat_or_dash(np.mean, circs),
            stat_or_dash(np.mean, intens), stat_or_dash(np.mean, combined),
            stat_or_dash(np.mean, vols) if vols else "TBD",
        ])
        cell_text.append([
            'STD', stat_or_dash(np.std, scales, 5), stat_or_dash(np.std, areas),
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

    if any(s.get('has_small') for s in numeric_rows):
        ax.text(0.02, 0.02, f"* includes an aggregate below {CONFIDENT_AREA_MM2}mm2 - worth a visual check in that photo's result window",
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
            display_failure(result['reason'], name)
            continue

        result = analyze_image(image, params['reference_mm'], params['color_mode'], params['thickness_mm'])
        all_results.append(result)

        if not result['success']:
            print(f"CALIBRATION FAILED: {result['reason']}")
            display_failure(result['reason'], name)
            continue

        print(f"Calibration: {result['line_px']:.0f}px = {params['reference_mm']}mm "
              f"(1px = {result['mm_per_px']:.5f}mm)")
        print(f"Otsu threshold: {result['otsu_threshold']:.0f}")
        print(f"Aggregates found: {len(result['measurements'])}")

        display_results(result, filename=name)

    if len(image_paths) > 1:
        print("\n" + "=" * 60)
        print("BATCH COMPARISON")
        print("=" * 60)
        display_comparison_table(all_results, filenames)

    print("\nAnalysis complete!")