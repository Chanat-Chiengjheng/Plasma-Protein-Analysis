# ARCHIVED: superseded by ../src/pipeline.py. Kept for iteration history; may reference paths relative to the old repo layout.
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

VERSION = "4.2"

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
BLUR_SIZE  = 15   # Gaussian blur kernel before thresholding (must be odd)
MORPH_SIZE = 3    # noise-removal kernel for morphological open/close
# NOTE: see "Noise filters" comment above re: MORPH_SIZE + small real holes.

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
MINIMUM_AREA_MM2 = 1.0  # fixed now; see "Noise filters" comment above

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
    root = tk.Tk()
    root.title("Plasma Aggregation Analysis - Setup")
    root.geometry("440x440")

    image_path_var = tk.StringVar()
    reference_var  = tk.StringVar()
    color_var      = tk.StringVar(value="Red")
    thickness_var  = tk.StringVar()
    error_var      = tk.StringVar()

    def browse_file():
        path = filedialog.askopenfilename(
            title="Select sample image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.JPG *.JPEG *.PNG"),
                       ("All files", "*.*")]
        )
        if path:
            image_path_var.set(path)

    def on_run():
        path = image_path_var.get().strip()
        if not path or not os.path.isfile(path):
            error_var.set("Please select a valid image file.")
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

        result['image_path']  = path
        result['reference_mm'] = reference_mm
        result['color_mode']  = color_mode
        result['thickness_mm'] = thickness_mm
        root.destroy()

    pad = {'padx': 16, 'pady': (10, 2)}

    tk.Label(root, text="Image file").pack(anchor='w', **pad)
    file_frame = tk.Frame(root)
    file_frame.pack(fill='x', padx=16)
    tk.Entry(file_frame, textvariable=image_path_var, state='readonly').pack(
        side='left', fill='x', expand=True)
    tk.Button(file_frame, text="Browse...", command=browse_file).pack(
        side='left', padx=(6, 0))

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
        image_path = input("Enter path to image file: ").strip().strip('"\'')
        if os.path.isfile(image_path):
            break
        print(f"  File not found: '{image_path}' - try again.\n")

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
        'image_path': image_path,
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

    # ---- Grayscale + illumination flattening + blur + Otsu threshold ----
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flattened = flatten_illumination(gray)
    blurred = cv2.GaussianBlur(flattened, (BLUR_SIZE, BLUR_SIZE), 0)
    otsu_threshold, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # ---- Noise removal ----
    kernel = np.ones((MORPH_SIZE, MORPH_SIZE), np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    # ---- Exclude the calibration line's own region ----
    # Confirmed by testing, not just theory: if the painted line happens to be
    # brighter than a dark background, Otsu can classify the line itself as
    # foreground, and its area is easily large enough to pass MINIMUM_AREA_MM2.
    # The area filter alone is not a reliable enough safety net on its own, so
    # the line's region is explicitly carved out before aggregate contours are
    # ever found, regardless of its brightness or area.
    line_exclusion_mask = np.zeros(gray.shape, dtype=np.uint8)
    (lcx, lcy), (lw, lh), langle = line['rect']
    padded_rect = ((lcx, lcy), (lw + 20, lh + 20), langle)  # small safety margin
    line_box = np.int32(cv2.boxPoints(padded_rect))
    cv2.fillPoly(line_exclusion_mask, [line_box], 255)
    cleaned[line_exclusion_mask == 255] = 0

    # ---- Contours + holes ----
    contours, hierarchy = cv2.findContours(cleaned, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    minimum_area_px = MINIMUM_AREA_MM2 / mm2_per_px2

    outer_contours = []
    hole_contours = []
    img_h, img_w = gray.shape
    border_artifact_area_px = BORDER_ARTIFACT_AREA_FRACTION * img_h * img_w
    if hierarchy is not None:
        for i, contour in enumerate(contours):
            parent_idx = hierarchy[0][i][3]
            area_px = cv2.contourArea(contour)
            if parent_idx == -1:
                if minimum_area_px < area_px <= border_artifact_area_px:
                    outer_contours.append((contour, i))
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
            'center_x': center_x, 'center_y': center_y,
            'bbox_x': x, 'bbox_y': y, 'bbox_w': w, 'bbox_h': h,
            'background_mean': background_mean,
            'aggregate_mean': aggregate_mean,
            'relative_intensity_index': relative_intensity_index,
            'combined_index': combined_index,
            'volume_mm3': volume_mm3,
        })

    # ---- Clean result image (no text block baked into it) ----
    result_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(result_image, [c for c, _ in outer_contours], -1, (0, 255, 0), 2)
    cv2.drawContours(result_image, [hc for hc, _ in hole_contours], -1, (0, 0, 255), 2)
    for m in measurements:
        cv2.circle(result_image, (m['center_x'], m['center_y']), 6, (0, 255, 255), -1)
        cv2.putText(result_image, f"#{m['id']}", (m['bbox_x'], m['bbox_y'] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

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

def display_results(r):
    measurements = r['measurements']

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    fig.suptitle(
        f"Plasma Aggregation Analysis v4  |  "
        f"{r['color_mode'].capitalize()} line: {r['line_px']:.0f}px = {r['reference_mm']}mm  |  "
        f"Otsu threshold: {r['otsu_threshold']:.0f}",
        fontsize=12, fontweight='bold'
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
    info_lines = ["SUMMARY", "-" * 34]
    if measurements:
        total_area = sum(m['true_area_mm2'] for m in measurements)
        avg_circ = sum(m['circularity'] for m in measurements) / len(measurements)
        if r['thickness_mm'] is not None:
            total_volume_str = f"{sum(m['volume_mm3'] for m in measurements):.4f} mm3"
        else:
            total_volume_str = "TBD (no thickness given)"

        info_lines += [
            f"Aggregates found:   {len(measurements)}",
            f"Total area:         {total_area:.4f} mm2",
            f"Total volume:       {total_volume_str}",
            f"Avg circularity:    {avg_circ:.3f}",
            f"Background mean:    {r['background_mean']:.1f}",
            "-" * 34,
        ]
        for m in measurements:
            vol_str = f"{m['volume_mm3']:.4f}" if m['volume_mm3'] is not None else "TBD"
            info_lines.append(f"#{m['id']}  area={m['true_area_mm2']:.3f}mm2  holes={m['num_holes']}")
            info_lines.append(
                f"    intensity_idx={m['relative_intensity_index']:.3f}  "
                f"combined={m['combined_index']:.3f}  volume={vol_str}"
            )
    else:
        info_lines += ["No aggregates detected.", "Try a different photo or check lighting."]

    axes[1, 1].text(0.02, 0.98, "\n".join(info_lines),
                     transform=axes[1, 1].transAxes, fontsize=9,
                     verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    plt.show()


# =============================================
# MAIN
# =============================================

if __name__ == "__main__":
    print("=" * 60)
    print(f"PLASMA AGGREGATION ANALYSIS  v{VERSION}")
    print("=" * 60)

    params = get_parameters()

    image = cv2.imread(params['image_path'])
    if image is None:
        print(f"ERROR: could not read image at '{params['image_path']}'")
        sys.exit(1)

    print(f"\nImage loaded: {image.shape[1]}w x {image.shape[0]}h px")
    print(f"Reference length: {params['reference_mm']} mm")
    print(f"Calibration line color: {params['color_mode'].capitalize()}")
    if params['thickness_mm'] is not None:
        print(f"Aggregate thickness: {params['thickness_mm']} mm")
    else:
        print("Aggregate thickness: not provided (volume will show as TBD)")

    result = analyze_image(image, params['reference_mm'], params['color_mode'], params['thickness_mm'])

    if not result['success']:
        print("\n" + "=" * 60)
        print("CALIBRATION FAILED")
        print("=" * 60)
        print(result['reason'])
        sys.exit(1)

    print(f"\nCalibration: {result['line_px']:.0f}px = {params['reference_mm']}mm "
          f"(1px = {result['mm_per_px']:.5f}mm)")
    print(f"Otsu threshold: {result['otsu_threshold']:.0f}")
    print(f"Aggregates found: {len(result['measurements'])}")

    display_results(result)

    print("\nAnalysis complete!")