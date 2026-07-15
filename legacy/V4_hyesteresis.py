import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import csv
from datetime import datetime

VERSION = "4.8"
BUILD_TAG = "lower-area-floor-2"

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
MINIMUM_AREA_MM2 = 0.14
CONFIDENT_AREA_MM2 = 1.0
BORDERLINE_AREA_MM2 = 0.05

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


# =============================================
# ILLUMINATION FLATTENING
# =============================================

def make_odd(n):
    n = int(n)
    return n if n % 2 == 1 else n + 1


def classify_contours(binary_mask, minimum_area_px, confident_area_px,
                       borderline_area_px, border_artifact_area_px,
                       mm2_per_px2, log_below_floor=False):
    contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    outer_contours = []
    hole_contours = []
    small_indices = set()
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
                elif log_below_floor and area_px > borderline_area_px:
                    area_mm2 = area_px * mm2_per_px2
                    print(f"  (below the counting floor: area={area_mm2:.4f}mm2, "
                          f"floor is {MINIMUM_AREA_MM2}mm2)")
            else:
                hole_contours.append((contour, parent_idx))
    return contours, hierarchy, outer_contours, hole_contours, small_indices


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
    kernel_size = 2 * cap_radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated_crop = cv2.dilate(component_crop, kernel)
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

    (core_contours, core_hierarchy, core_outer_contours,
     core_hole_contours, core_small_indices) = classify_contours(
        cleaned, minimum_area_px, confident_area_px, borderline_area_px,
        border_artifact_area_px, mm2_per_px2, log_below_floor=True)

    # ---- Hysteresis edge recovery ----
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
     hole_contours, small_indices) = classify_contours(
        grown_mask, minimum_area_px, confident_area_px, borderline_area_px,
        border_artifact_area_px, mm2_per_px2, log_below_floor=False)

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

        denom = max(255.0 - background_mean, 1e-6)
        relative_intensity_index = max(0.0, (aggregate_mean - background_mean) / denom)
        combined_index = true_area_mm2 * relative_intensity_index

        volume_mm3 = true_area_mm2 * thickness_mm if thickness_mm is not None else None

        core_area_px_count = int(np.count_nonzero((agg_mask == 255) & (seed_mask == 255)))
        hysteresis_area_px_count = int(np.count_nonzero((agg_mask == 255) & (seed_mask == 0)))
        core_area_mm2 = core_area_px_count * mm2_per_px2
        hysteresis_area_mm2 = hysteresis_area_px_count * mm2_per_px2

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
        'bubble_pixel_count': bubble_pixel_count,
        'measurements': measurements,
        'background_mean': background_mean,
        'thickness_mm': thickness_mm,
    }


# =============================================
# DISPLAY — single window, 2x2 grid
# =============================================

def compute_photo_summary(r):
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
        'avg_intensity_idx': sum(m['relative_intensity_index'] for m in measurements) / len(measurements),
        'total_combined': sum(m['combined_index'] for m in measurements),
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
                      'Combined index\n(mm2)', 'Volume\n(mm3)']
        RECOVERED_COL = 4
        scale_str = f"{r['mm_per_px']:.5f}"
        any_small = any(m['is_small'] for m in measurements)
        any_overgrown = any(m['is_overgrown'] for m in measurements)
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
                f"{m['circularity']:.3f}", f"{m['relative_intensity_index']:.3f}",
                f"{m['combined_index']:.3f}", vol_str,
            ])
        total_vol_str = f"{summary['total_volume']:.3f}" if summary['total_volume'] is not None else "TBD"
        cell_text.append([
            'TOTAL', scale_str, f"{summary['total_area']:.3f}",
            f"{summary['total_core']:.3f}", f"{summary['total_recovered']:.3f}",
            f"{summary['total_holes']}",
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
        if summary['has_overgrown']:
            name_str = f"{name} !({summary['max_overgrowth_ratio']:.1f}x)"
            overgrown_row_indices.append(len(cell_text) + 1)  # +1: header is row 0
        elif summary['has_small']:
            name_str = f"{name}*"
        else:
            name_str = name
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
        print(f"Aggregates found: {len(result['measurements'])}")
        for m in result['measurements']:
            print(f"  #{m['id']}: core={m['core_area_mm2']:.4f}mm2  "
                  f"hysteresis-recovered={m['hysteresis_area_mm2']:.4f}mm2  "
                  f"total={m['true_area_mm2']:.4f}mm2")

        display_results(result, filename=name)
        append_photo_to_master_csv(name, params, result)
        print(f"Logged to master CSV: {name}")

    if len(image_paths) > 1:
        print("\n" + "=" * 60)
        print("BATCH COMPARISON")
        print("=" * 60)
        display_comparison_table(all_results, filenames)

    print("\nAnalysis complete!")
