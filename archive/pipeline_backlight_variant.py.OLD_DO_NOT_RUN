import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

VERSION = "B1.2"
BUILD_TAG = "batch-enabled"

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
SATURATION_MIN = 80
VALUE_MIN      = 40

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
MINIMUM_AREA_MM2   = 0.4
CONFIDENT_AREA_MM2 = 1.0

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
    level. Direction-agnostic - works the same whether the foreground ends
    up being the bright side or the dark side of the resulting threshold."""
    h, w = gray.shape
    kernel_size = make_odd(min(h, w) * ILLUMINATION_KERNEL_FRACTION)
    illumination = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    diff = gray.astype(np.int16) - illumination.astype(np.int16)
    return np.clip(diff + 128, 0, 255).astype(np.uint8)


# =============================================
# CALIBRATION LINE DETECTION (identical to reflected-light V4.2)
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

    # ---- Otsu threshold, computed without the calibration line's pixels ----
    # THRESH_BINARY_INV (not THRESH_BINARY): backlight aggregate is DARK on
    # a bright background, so the dark side of the cut is foreground here.
    otsu_threshold, _ = cv2.threshold(
        blurred[line_exclusion_mask == 0], 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    _, binary = cv2.threshold(blurred, otsu_threshold, 255, cv2.THRESH_BINARY_INV)

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

    # ---- Contours + holes ----
    contours, hierarchy = cv2.findContours(cleaned, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    minimum_area_px = MINIMUM_AREA_MM2 / mm2_per_px2
    confident_area_px = CONFIDENT_AREA_MM2 / mm2_per_px2

    outer_contours = []
    hole_contours = []
    small_indices = set()  # counted, but below the confident floor - gets a "*" mark
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
            else:
                hole_contours.append((contour, parent_idx))

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
        'avg_optical_density': sum(m['optical_density_index'] for m in measurements) / len(measurements),
        'total_combined': sum(m['combined_optical_density'] for m in measurements),
        'total_volume': total_volume,
        'has_small': any(m['is_small'] for m in measurements),
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
        col_labels = ['#', 'Scale\n(mm/px)', 'Area\n(mm2)', 'Holes\n(count)',
                      'Circularity\n(unitless)', 'Optical density\nindex (unitless)',
                      'Combined optical\ndensity (mm2)', 'Volume\n(mm3)']
        scale_str = f"{r['mm_per_px']:.5f}"
        cell_text = []
        for m in measurements:
            vol_str = f"{m['volume_mm3']:.3f}" if m['volume_mm3'] is not None else "TBD"
            id_str = f"{m['id']}*" if m['is_small'] else f"{m['id']}"
            cell_text.append([
                id_str, scale_str, f"{m['true_area_mm2']:.3f}", f"{m['num_holes']}",
                f"{m['circularity']:.3f}", f"{m['optical_density_index']:.3f}",
                f"{m['combined_optical_density']:.3f}", vol_str,
            ])
        total_vol_str = f"{summary['total_volume']:.3f}" if summary['total_volume'] is not None else "TBD"
        cell_text.append([
            'TOTAL', scale_str, f"{summary['total_area']:.3f}", f"{summary['total_holes']}",
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
                  'Circularity\n(unitless)', 'Optical density\nindex (unitless)',
                  'Combined optical\ndensity (mm2)', 'Volume\n(mm3)']
    cell_text = []
    numeric_rows = []
    scales = []

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
        cell_text.append([
            name, scale_str, f"{summary['total_area']:.3f}", f"{summary['total_holes']}",
            f"{summary['avg_circularity']:.3f}", f"{summary['avg_optical_density']:.3f}",
            f"{summary['total_combined']:.3f}", vol_str,
        ])
        numeric_rows.append(summary)

    if numeric_rows or scales:
        areas = [s['total_area'] for s in numeric_rows]
        holes = [s['total_holes'] for s in numeric_rows]
        circs = [s['avg_circularity'] for s in numeric_rows]
        odens = [s['avg_optical_density'] for s in numeric_rows]
        combined = [s['total_combined'] for s in numeric_rows]
        vols = [s['total_volume'] for s in numeric_rows if s['total_volume'] is not None]

        def stat_or_dash(fn, vals, nd=3):
            return f"{fn(vals):.{nd}f}" if vals else '-'

        cell_text.append([
            'AVG', stat_or_dash(np.mean, scales, 5), stat_or_dash(np.mean, areas),
            stat_or_dash(np.mean, holes, 1), stat_or_dash(np.mean, circs),
            stat_or_dash(np.mean, odens), stat_or_dash(np.mean, combined),
            stat_or_dash(np.mean, vols) if vols else "TBD",
        ])
        cell_text.append([
            'STD', stat_or_dash(np.std, scales, 5), stat_or_dash(np.std, areas),
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

    plt.tight_layout()
    plt.show()


# =============================================
# MAIN
# =============================================

if __name__ == "__main__":
    print("=" * 60)
    print(f"PLASMA AGGREGATION ANALYSIS (BACKLIGHT)  v{VERSION}  [build: {BUILD_TAG}]")
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