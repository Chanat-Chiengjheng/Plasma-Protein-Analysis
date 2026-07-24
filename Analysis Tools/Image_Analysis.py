import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import csv
from datetime import datetime

VERSION = "5.0"
BUILD_TAG = "reflected-backlight-merge"



MINIMUM_AREA_MM2 = {'REFLECTED': 0.10, 'BACKLIGHT': 0.14}
MAX_ASPECT_RATIO = {'REFLECTED': None, 'BACKLIGHT': 6.0}

BLUR_SIZE_FRACTION  = 0.0043
MORPH_SIZE_FRACTION = 0.0009

ILLUMINATION_KERNEL_FRACTION = 0.5
BORDER_ARTIFACT_AREA_FRACTION = 0.9

HUE_CENTERS = {'RED': 0, 'GREEN': 60, 'BLUE': 120}
HUE_TOLERANCE  = 15
SATURATION_MIN = 80
VALUE_MIN      = 40
RELATIVE_SATURATION_FRACTION = 0.35
RELATIVE_VALUE_FRACTION      = 0.35
ABSOLUTE_SATURATION_FLOOR    = 30
ABSOLUTE_VALUE_FLOOR         = 15

MIN_ASPECT_RATIO         = 3.0
MIN_LINE_LENGTH_FRACTION = 0.05

CONFIDENT_AREA_MM2 = 1.0
BORDERLINE_AREA_MM2 = 0.05

ENABLE_LOCAL_CONTRAST_PROTOTYPE = False
LOCAL_CONTRAST_KERNEL_FRACTION = 0.06
LOCAL_CONTRAST_STD_MULTIPLIER = 3.0
LOCAL_CONTRAST_MIN_MARGIN = 4
LOCAL_CONTRAST_MIN_SEED_AREA_MM2 = 0.10

ENABLE_CLUSTER_FLAGGING = True
FRAGMENT_CLUSTER_MAX_GAP_FRACTION = 0.058
FRAGMENT_CLUSTER_MIN_FILL_RATIO = 0.06
FRAGMENT_CLUSTER_MAX_FRAGMENTS = 300

LOOSE_THRESHOLD_STD_MULTIPLIER = 2.0
GROWTH_MAX_RADIUS_MULTIPLIER = 1.5

ENABLE_BUBBLE_EXCLUSION = False
BUBBLE_MIN_RADIUS_FRACTION = 0.006
BUBBLE_MAX_RADIUS_FRACTION = 0.06
BUBBLE_DETECTION_MAX_DIMENSION = 1000
BUBBLE_CORE_OVERLAP_MAX = 0.3

OVERGROWTH_RATIO = 0.5

DRAW_COLORS = {'RED': (0, 0, 255), 'GREEN': (0, 255, 0), 'BLUE': (255, 0, 0)}
HYSTERESIS_COLOR_BGR = (255, 255, 0)
OVERGROWN_COLOR_BGR = (0, 0, 255)
CLUSTER_UNCONFIRMED_COLOR_BGR = (255, 0, 255)


MASTER_CSV_FILENAME = "plasma_analysis_master_log_v5.csv"
MASTER_CSV_DIR = r"C:\Users\66950\Desktop\Projects in github\Plasma\data"
os.makedirs(MASTER_CSV_DIR, exist_ok=True)
MASTER_CSV_PATH = os.path.join(MASTER_CSV_DIR, MASTER_CSV_FILENAME)

CSV_COLUMNS = [
    "run_timestamp", "filename", "version", "build_tag", "mode",
    "calibration_status", "calibration_failure_reason",
    "reference_mm", "calibration_color", "mm_per_px", "background_mean",
    "aggregate_count", "total_area_mm2", "total_holes", "avg_circularity",
    "avg_relative_intensity_index", "total_combined_index",
    "avg_optical_density_index", "total_combined_optical_density",
    "total_volume_mm3", "has_flagged_small_aggregate",
    "unconfirmed_cluster_count", "unconfirmed_cluster_area_mm2",
    "hsv_saturation_min", "hsv_value_min", "hsv_s_ref_p99", "hsv_v_ref_p99",
]


def append_photo_to_master_csv(filename, params, result):
    file_is_new = (not os.path.exists(MASTER_CSV_PATH)) or os.path.getsize(MASTER_CSV_PATH) == 0

    row = {col: "" for col in CSV_COLUMNS}
    row["run_timestamp"] = datetime.now().isoformat(timespec="seconds")
    row["filename"] = filename
    row["version"] = VERSION
    row["build_tag"] = BUILD_TAG
    row["mode"] = params.get("mode", "")
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
            row["aggregate_count"] = 0
            row["unconfirmed_cluster_count"] = 0
        else:
            row["aggregate_count"] = summary["count"]
            row["total_area_mm2"] = f'{summary["total_area"]:.3f}'
            row["total_holes"] = summary["total_holes"]
            row["avg_circularity"] = f'{summary["avg_circularity"]:.3f}'
            if result["mode"] == "REFLECTED":
                row["avg_relative_intensity_index"] = f'{summary["avg_headline"]:.3f}'
                row["total_combined_index"] = f'{summary["total_combined"]:.3f}'
            else:
                row["avg_optical_density_index"] = f'{summary["avg_headline"]:.3f}'
                row["total_combined_optical_density"] = f'{summary["total_combined"]:.3f}'
            row["total_volume_mm3"] = (
                f'{summary["total_volume"]:.3f}' if summary["total_volume"] is not None else "TBD"
            )
            row["has_flagged_small_aggregate"] = "TRUE" if summary["has_small"] else "FALSE"
            row["unconfirmed_cluster_count"] = summary["unconfirmed_count"]
            row["unconfirmed_cluster_area_mm2"] = f'{summary["unconfirmed_total_area"]:.3f}'

    with open(MASTER_CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if file_is_new:
            writer.writeheader()
        writer.writerow(row)



def make_odd(n):
    n = int(n)
    return n if n % 2 == 1 else n + 1


def classify_contours(binary_mask, minimum_area_px, confident_area_px,
                       borderline_area_px, border_artifact_area_px,
                       mm2_per_px2, collect_diagnostics=False):
    """[shared] Runs findContours + the area-floor/border-artifact rules on
    a binary mask. When collect_diagnostics is True, also returns a "why
    was this rejected" list (border-artifact vs below-floor), computed from
    the core pass only - unaffected by hysteresis growth."""
    contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    outer_contours = []
    hole_contours = []
    small_indices = set()
    rejected = []
    minimum_area_mm2 = minimum_area_px * mm2_per_px2
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
                        rejected.append((area_mm2, f'below minimum floor ({minimum_area_mm2}mm2)'))
                else:
                    outer_contours.append((contour, i))
                    if area_px <= confident_area_px:
                        small_indices.add(i)
            else:
                hole_contours.append((contour, parent_idx))
    rejected.sort(key=lambda x: x[0], reverse=True)
    return contours, hierarchy, outer_contours, hole_contours, small_indices, rejected


def cluster_fragments(cleaned, minimum_area_px, border_artifact_area_px,
                       gap_px, min_fill_ratio, mm2_per_px2):
    """[shared as of V5.0] Groups individually-sub-floor connected
    components in `cleaned` (the post-morphology mask) that sit within
    gap_px of each other, when their COMBINED area would clear the area
    floor even though none does alone. Runs AFTER Otsu + morphology, BEFORE
    classify_contours()'s area-floor rejection - never touches the global
    Otsu threshold or the morphology step itself. Polarity-agnostic: only
    ever operates on `cleaned`, which is already the mode-appropriate
    foreground mask by the time this runs (THRESH_BINARY vs
    THRESH_BINARY_INV was already resolved upstream). See V4.14's changelog
    in legacy/V4_hyesteresis.py for the full history of this mechanism's
    design (why hull-fill bridge area must be excluded from the reported
    area, why cluster candidates are flagged rather than auto-counted,
    etc.) - unchanged here, just newly available to backlight mode too.

    Returns (cluster_seed_mask, fragment_original_mask, candidate_info)."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)

    sub_floor_labels = []
    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area > border_artifact_area_px or area > minimum_area_px:
            continue
        sub_floor_labels.append(lbl)

    if len(sub_floor_labels) < 2:
        return np.zeros(cleaned.shape, dtype=np.uint8), np.zeros(cleaned.shape, dtype=np.uint8), []
    if len(sub_floor_labels) > FRAGMENT_CLUSTER_MAX_FRAGMENTS:
        print(f"  [cluster-flagging] skipped: {len(sub_floor_labels)} sub-floor "
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
            bbox_gap_x = max(0, max(lx, jx) - min(lx + lw, jx + jw))
            bbox_gap_y = max(0, max(ly, jy) - min(ly + lh, jy + jh))
            if (bbox_gap_x ** 2 + bbox_gap_y ** 2) ** 0.5 > gap_px:
                continue
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
    fragment_original_mask = np.zeros(cleaned.shape, dtype=np.uint8)
    candidate_info = []
    for members in groups.values():
        if len(members) < 2:
            continue
        total_area_px = sum(int(stats[m, cv2.CC_STAT_AREA]) for m in members)
        if total_area_px <= minimum_area_px:
            continue

        member_mask = np.isin(labels, members).astype(np.uint8)
        pts = cv2.findNonZero(member_mask)
        hull = cv2.convexHull(pts)
        hull_area_px = cv2.contourArea(hull)
        fill_ratio = total_area_px / max(hull_area_px, 1)
        if fill_ratio < min_fill_ratio:
            continue

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
    """[reflected only] PROTOTYPE/EXPERIMENTAL (V4.10) - see V5.0 changelog
    for why this is gated to reflected mode only. Finds candidate seed
    blobs in regions the existing raw global-Otsu pass (raw_binary) never
    flagged at all, by checking whether a pixel exceeds its OWN local-
    neighborhood mean rather than one frame-wide cutoff."""
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

    candidate_mask[raw_binary == 255] = 0
    candidate_mask[line_exclusion_mask == 255] = 0

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
    """[shared] Hough-circle-based bubble detector. Polarity-agnostic: only
    looks for round shapes in `gray` and checks their overlap against
    seed_mask, regardless of whether the aggregate itself is the bright or
    dark side of the frame. Opt-in via ENABLE_BUBBLE_EXCLUSION (default
    False for both modes as of V5.0) - see V5.0 changelog."""
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
    """[shared] Local (bounding-box-limited) growth-cap region for one
    connected-component label. Uses cv2.distanceTransform (V4.14 perf fix -
    see V5.0 changelog) rather than cv2.dilate with a giant kernel;
    mathematically equivalent, confirmed ~45x faster on the worst case seen
    (a ~700px-equivalent-radius aggregate) - a pure speedup for backlight
    too, since this function was polarity-agnostic already."""
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
    dist = cv2.distanceTransform(255 - component_crop, cv2.DIST_L2, 5)
    dilated_crop = np.where(dist <= cap_radius, 255, 0).astype(np.uint8)
    return y0, y1, x0, x1, dilated_crop


def build_growth_distance_cap(seed_mask):
    """[shared] Caps how far hysteresis growth can spread from each core
    blob, AND stops growth from bridging two separate confirmed aggregates
    together (contested no-man's-land pixels are excluded from growth for
    both)."""
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
    """[shared] Removes a slow-varying lighting gradient so Otsu compares
    each pixel to its own local expected background instead of one global
    brightness level. Direction-agnostic - works the same whether the
    foreground ends up being the bright side or the dark side of the
    resulting threshold.

    exclusion_mask (optional): region(s) - e.g. the calibration line/bar -
    to blank out with the surrounding background level before estimating
    the illumination trend, so a large solid excluded object can't drag the
    "expected local background" down/up over a wide halo around it (see
    B1.4's changelog in legacy/B1.py for the real-photo case this fixed).
    Reflected mode calls this WITHOUT an exclusion mask (preserving V4.14's
    exact behavior byte-for-byte - see V5.0 changelog for why this wasn't
    changed as part of the merge); backlight mode calls it WITH the
    calibration line's exclusion mask, same as B1.10."""
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



def compute_adaptive_hsv_floors(hsv_image):
    """[shared as of V5.0, ported from B1.3] Derive this photo's own
    saturation/value floors instead of using a fixed constant across all
    lighting conditions - see V5.0 changelog for why this now applies to
    reflected mode too."""
    s = hsv_image[:, :, 1].astype(int)
    v = hsv_image[:, :, 2].astype(int)
    s_ref = float(np.percentile(s, 99))
    v_ref = float(np.percentile(v, 99))
    saturation_min = max(ABSOLUTE_SATURATION_FLOOR, RELATIVE_SATURATION_FRACTION * s_ref)
    value_min = max(ABSOLUTE_VALUE_FLOOR, RELATIVE_VALUE_FRACTION * v_ref)
    return saturation_min, value_min, s_ref, v_ref


def get_color_mask(hsv_image, color_mode, saturation_min, value_min):
    h = hsv_image[:, :, 0].astype(int)
    s = hsv_image[:, :, 1].astype(int)
    v = hsv_image[:, :, 2].astype(int)

    center = HUE_CENTERS[color_mode]

    if color_mode == 'RED':
        hue_mask = (h <= HUE_TOLERANCE) | (h >= 180 - HUE_TOLERANCE)
    else:
        hue_mask = np.abs(h - center) <= HUE_TOLERANCE

    return hue_mask & (s >= saturation_min) & (v >= value_min)


def find_calibration_line(color_mask, max_aspect_ratio=None):
    """[shared] Find the painted reference line by shape (length-to-
    thickness ratio + a relative length floor). max_aspect_ratio (backlight
    only, see V5.0 changelog): guards against the transparent-boundary
    chromatic-fringe artifact - None means no ceiling (reflected mode,
    preserving V4.14's exact behavior)."""
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
        if aspect_ratio < MIN_ASPECT_RATIO:
            continue
        if max_aspect_ratio is not None and aspect_ratio > max_aspect_ratio:
            continue
        if length_px >= min_length_px:
            candidates.append({
                'rect': rect,
                'length_px': length_px,
                'thickness_px': thickness_px,
                'center': (cx, cy),
            })

    if not candidates:
        return None
    return max(candidates, key=lambda c: c['length_px'])



def get_parameters_gui():
    import tkinter as tk
    from tkinter import filedialog, ttk

    result = {}
    selected_paths = []
    root = tk.Tk()
    root.title("Plasma Aggregation Analysis - Setup")
    root.geometry("440x480")

    image_path_var = tk.StringVar()
    mode_var       = tk.StringVar(value="Reflected Light")
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

        mode = 'BACKLIGHT' if mode_var.get().strip() == 'Backlight' else 'REFLECTED'

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
        result['mode'] = mode
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

    tk.Label(root, text="Lighting setup").pack(anchor='w', **pad)
    ttk.Combobox(root, textvariable=mode_var, values=["Reflected Light", "Backlight"],
                 state="readonly").pack(fill='x', padx=16)

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
        mode_input = input(
            "Enter lighting setup [Reflected/Backlight] (default Reflected): "
        ).strip().upper()
        if mode_input == "":
            mode = "REFLECTED"
            break
        elif mode_input in ("REFLECTED", "BACKLIGHT"):
            mode = mode_input
            break
        print("  Please enter Reflected or Backlight.\n")

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
        'mode': mode,
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



def analyze_image(image, reference_mm, color_mode, thickness_mm, mode):
    """mode: 'REFLECTED' or 'BACKLIGHT'. See the V5.0 changelog for exactly
    which behavior is mode-gated vs shared."""
    minimum_area_mm2_for_mode = MINIMUM_AREA_MM2[mode]
    max_aspect_ratio_for_mode = MAX_ASPECT_RATIO[mode]

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _adaptive_saturation_min, _adaptive_value_min, s_ref, v_ref = compute_adaptive_hsv_floors(hsv)
    if mode == 'BACKLIGHT':
        saturation_min, value_min = _adaptive_saturation_min, _adaptive_value_min
    else:
        saturation_min, value_min = SATURATION_MIN, VALUE_MIN
    color_mask = get_color_mask(hsv, color_mode, saturation_min, value_min)
    line = find_calibration_line(color_mask, max_aspect_ratio_for_mode)
    hsv_diagnostics = {
        'saturation_min': saturation_min, 'value_min': value_min,
        's_ref': s_ref, 'v_ref': v_ref,
    }

    if line is None:
        floor_kind = "adaptive" if mode == 'BACKLIGHT' else "fixed"
        return {'success': False, 'mode': mode, 'reason': (
            f"No {color_mode.lower()} calibration line could be confirmed.\n"
            f"Possible causes: the line isn't in frame, lighting is too poor,\n"
            f"or the wrong color was selected for this photo.\n"
            f"(This photo's {floor_kind} floors: saturation>={saturation_min:.0f} "
            f"[99th pct S={s_ref:.0f}], value>={value_min:.0f} [99th pct V={v_ref:.0f}])"
        ), 'hsv_diagnostics': hsv_diagnostics}

    line_px = line['length_px']
    mm_per_px = reference_mm / line_px
    mm2_per_px2 = mm_per_px ** 2

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

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_h, img_w = gray.shape
    blur_size = max(3, make_odd(BLUR_SIZE_FRACTION * min(img_h, img_w)))
    morph_size = max(2, int(round(MORPH_SIZE_FRACTION * min(img_h, img_w))))

    line_exclusion_mask = np.zeros(gray.shape, dtype=np.uint8)
    (lcx, lcy), (lw, lh), langle = line['rect']
    padded_rect = ((lcx, lcy), (lw + 20, lh + 20), langle)
    line_box = np.int32(cv2.boxPoints(padded_rect))
    cv2.fillPoly(line_exclusion_mask, [line_box], 255)

    if mode == 'BACKLIGHT':
        flattened = flatten_illumination(gray, line_exclusion_mask)
    else:
        flattened = flatten_illumination(gray)
    blurred = cv2.GaussianBlur(flattened, (blur_size, blur_size), 0)

    if mode == 'BACKLIGHT':
        otsu_threshold, _ = cv2.threshold(
            blurred[line_exclusion_mask == 0], 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        _, binary = cv2.threshold(blurred, otsu_threshold, 255, cv2.THRESH_BINARY_INV)
    else:
        otsu_threshold, _ = cv2.threshold(
            blurred[line_exclusion_mask == 0], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        _, binary = cv2.threshold(blurred, otsu_threshold, 255, cv2.THRESH_BINARY)
    raw_foreground_fraction = float(np.count_nonzero(binary)) / (img_h * img_w)

    kernel = np.ones((morph_size, morph_size), np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    line_center_point = (int(lcx), int(lcy))
    pre_exclusion_contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in pre_exclusion_contours:
        if cv2.pointPolygonTest(c, line_center_point, False) >= 0:
            cv2.drawContours(cleaned, [c], -1, 0, -1)
    cleaned[line_exclusion_mask == 255] = 0

    minimum_area_px = minimum_area_mm2_for_mode / mm2_per_px2
    confident_area_px = CONFIDENT_AREA_MM2 / mm2_per_px2
    borderline_area_px = BORDERLINE_AREA_MM2 / mm2_per_px2
    border_artifact_area_px = BORDER_ARTIFACT_AREA_FRACTION * img_h * img_w
    foreground_fraction = float(np.count_nonzero(cleaned)) / (img_h * img_w)

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
                print(f"  [cluster-flagging] candidate: {c['num_fragments']} fragments -> "
                      f"real={c['fragment_area_mm2']:.4f}mm2 hull={c['hull_area_mm2']:.4f}mm2 "
                      f"fill_ratio={c['fill_ratio']:.2f} bbox={c['bbox']}")

    (core_contours, core_hierarchy, core_outer_contours,
     core_hole_contours, core_small_indices, rejected) = classify_contours(
        cleaned_for_core, minimum_area_px, confident_area_px, borderline_area_px,
        border_artifact_area_px, mm2_per_px2, collect_diagnostics=True)

    detection_diagnostics = {
        'raw_foreground_fraction': raw_foreground_fraction,
        'foreground_fraction': foreground_fraction,
        'rejected_top5': rejected[:5],
    }

    qualifying_core_idx = {i for _, i in core_outer_contours}
    seed_mask = cleaned_for_core.copy()
    if core_hierarchy is not None:
        for i, contour in enumerate(core_contours):
            parent_idx = core_hierarchy[0][i][3]
            if parent_idx == -1 and i not in qualifying_core_idx:
                cv2.drawContours(seed_mask, [contour], -1, 0, -1)

    local_contrast_diagnostics = {'enabled': ENABLE_LOCAL_CONTRAST_PROTOTYPE and mode == 'REFLECTED',
                                   'candidates': []}
    local_contrast_seed_mask = np.zeros(gray.shape, dtype=np.uint8)
    if ENABLE_LOCAL_CONTRAST_PROTOTYPE and mode == 'REFLECTED':
        local_contrast_seed_mask, local_contrast_candidates = find_local_contrast_seeds(
            blurred, binary, line_exclusion_mask, morph_size, mm2_per_px2)
        local_contrast_diagnostics['candidates'] = local_contrast_candidates
        if np.any(local_contrast_seed_mask):
            seed_mask = cv2.bitwise_or(seed_mask, local_contrast_seed_mask)
            print(f"  [PROTOTYPE] local-contrast path added {len(local_contrast_candidates)} "
                  f"seed candidate(s): " +
                  ", ".join(f"{c['area_mm2']:.4f}mm2(z={c['mean_z']:.1f})" for c in local_contrast_candidates))

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

    if mode == 'BACKLIGHT':
        loose_threshold = raw_background_mean - LOOSE_THRESHOLD_STD_MULTIPLIER * raw_background_std
        print(f"  Loose threshold (hysteresis, raw, darker-than-background): {loose_threshold:.1f}  "
              f"(background mean {raw_background_mean:.1f} - {LOOSE_THRESHOLD_STD_MULTIPLIER} "
              f"x std {raw_background_std:.1f})")
        loose_mask = np.where(gray < loose_threshold, 255, 0).astype(np.uint8)
    else:
        loose_threshold = raw_background_mean + LOOSE_THRESHOLD_STD_MULTIPLIER * raw_background_std
        print(f"  Loose threshold (hysteresis, raw): {loose_threshold:.1f}  "
              f"(background mean {raw_background_mean:.1f} + {LOOSE_THRESHOLD_STD_MULTIPLIER} "
              f"x std {raw_background_std:.1f})")
        loose_mask = np.where(gray > loose_threshold, 255, 0).astype(np.uint8)

    loose_mask[line_exclusion_mask == 255] = 0
    loose_mask[core_hole_mask == 255] = 0

    bubble_pixel_count = 0
    if ENABLE_BUBBLE_EXCLUSION:
        bubble_mask = detect_bubble_mask(gray, seed_mask, line_exclusion_mask)
        bubble_pixel_count = int(np.count_nonzero(bubble_mask == 255))
        if bubble_pixel_count > 0:
            print(f"  Bubble exclusion triggered: {bubble_pixel_count}px marked as "
                  f"bubble, excluded from hysteresis growth")
        loose_mask[bubble_mask == 255] = 0

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

    (contours, hierarchy, outer_contours,
     hole_contours, small_indices, _) = classify_contours(
        grown_mask, minimum_area_px, confident_area_px, borderline_area_px,
        border_artifact_area_px, mm2_per_px2, collect_diagnostics=False)

    full_mask = np.zeros(gray.shape, dtype=np.uint8)
    for contour, _ in outer_contours:
        cv2.drawContours(full_mask, [contour], -1, 255, -1)
    background_pixels = gray[full_mask == 0]
    background_mean = float(background_pixels.mean()) if background_pixels.size > 0 else 0.0

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

        cluster_bridge_px_for_mean = int(np.count_nonzero((agg_mask == 255) & bridge_only_mask))
        if cluster_bridge_px_for_mean > 0:
            real_pixels = gray[(agg_mask == 255) & (~bridge_only_mask)]
            if real_pixels.size > 0:
                corrected_mean = float(real_pixels.mean())
                print(f"  NOTE: aggregate #{i + 1} aggregate_mean corrected for fragment-cluster "
                      f"bridge pixels - was {aggregate_mean:.1f} (diluted by hull/bridge "
                      f"background pixels), now {corrected_mean:.1f} (real foreground pixels only).")
                aggregate_mean = corrected_mean

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

        if mode == 'BACKLIGHT':
            safe_background = max(background_mean, 1e-6)
            relative_transmittance = min(1.0, max(1e-6, aggregate_mean / safe_background))
            optical_density_index = -np.log10(relative_transmittance)
            combined_optical_density = true_area_mm2 * optical_density_index
            relative_intensity_index = None
            combined_index = None
        else:
            denom = max(255.0 - background_mean, 1e-6)
            relative_intensity_index = max(0.0, (aggregate_mean - background_mean) / denom)
            combined_index = true_area_mm2 * relative_intensity_index
            relative_transmittance = None
            optical_density_index = None
            combined_optical_density = None

        volume_mm3 = true_area_mm2 * thickness_mm if thickness_mm is not None else None

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

        core_bridge_px = int(np.count_nonzero((agg_mask == 255) & bridge_only_mask))
        core_area_px_count = int(np.count_nonzero((agg_mask == 255) & (seed_mask == 255) &
                                                    (~bridge_only_mask)))
        hysteresis_area_px_count = int(np.count_nonzero((agg_mask == 255) & (seed_mask == 0)))
        core_area_mm2 = core_area_px_count * mm2_per_px2
        hysteresis_area_mm2 = hysteresis_area_px_count * mm2_per_px2
        if core_bridge_px > 0:
            print(f"  NOTE: aggregate #{i + 1} core_area_mm2 corrected for fragment-cluster "
                  f"bridge pixels - {core_bridge_px}px ({core_bridge_px * mm2_per_px2:.4f}mm2) "
                  f"of hull/bridge area excluded from the core pixel count.")

        is_overgrown = hysteresis_area_mm2 > OVERGROWTH_RATIO * max(core_area_mm2, 1e-9)
        overgrowth_ratio = hysteresis_area_mm2 / max(core_area_mm2, 1e-9)
        if is_overgrown:
            print(f"  WARNING: aggregate #{i + 1} hysteresis-recovered area "
                  f"({hysteresis_area_mm2:.4f}mm2) exceeds {OVERGROWTH_RATIO * 100:.0f}% "
                  f"of its core area ({core_area_mm2:.4f}mm2) - flagged as suspicious growth "
                  f"(recovered/core = {overgrowth_ratio:.2f}x)")

        display_core_mm2 = min(core_area_mm2, true_area_mm2)
        display_recovered_mm2 = true_area_mm2 - display_core_mm2

        is_small_verdict = contour_idx in small_indices
        if cluster_bridge_px > 0:
            corrected_is_small = true_area_mm2 < CONFIDENT_AREA_MM2
            if corrected_is_small != is_small_verdict:
                print(f"  NOTE: aggregate #{i + 1} is_small corrected for fragment-cluster "
                      f"hull inflation - was {is_small_verdict}, now {corrected_is_small} "
                      f"(based on corrected true_area_mm2={true_area_mm2:.4f}mm2 vs "
                      f"{CONFIDENT_AREA_MM2}mm2 confident floor).")
            is_small_verdict = corrected_is_small
        if core_area_mm2 > true_area_mm2 + 1e-6:
            print(f"  NOTE: aggregate #{i + 1} rasterized core pixel-count area "
                  f"({core_area_mm2:.4f}mm2) exceeds its contour-based total area "
                  f"({true_area_mm2:.4f}mm2) by {core_area_mm2 - true_area_mm2:.4f}mm2 - "
                  f"a boundary-rasterization discrepancy, most visible on tiny aggregates. "
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
            'relative_transmittance': relative_transmittance,
            'optical_density_index': optical_density_index,
            'combined_optical_density': combined_optical_density,
            'volume_mm3': volume_mm3,
        })

    result_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    final_mask = np.zeros(gray.shape, dtype=np.uint8)
    for contour, _ in outer_contours:
        cv2.drawContours(final_mask, [contour], -1, 255, -1)
    for hc, _ in hole_contours:
        cv2.drawContours(final_mask, [hc], -1, 0, -1)
    hysteresis_mask = np.where((final_mask == 255) & (seed_mask == 0), 255, 0).astype(np.uint8)

    if np.any(hysteresis_mask):
        result_image[hysteresis_mask == 255] = HYSTERESIS_COLOR_BGR

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
        'success': True, 'mode': mode,
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
        'hsv_diagnostics': hsv_diagnostics,
        'detection_diagnostics': detection_diagnostics,
        'local_contrast_diagnostics': local_contrast_diagnostics,
        'fragment_cluster_diagnostics': fragment_cluster_diagnostics,
    }



def compute_photo_summary(r):
    """[shared] avg_headline/total_combined hold whichever physics metric
    matches r['mode'] (relative_intensity_index/combined_index for
    reflected, optical_density_index/combined_optical_density for
    backlight) - see V5.0 changelog."""
    all_measurements = r['measurements']
    if not all_measurements:
        return None
    measurements = [m for m in all_measurements if not m['is_cluster_origin']]
    unconfirmed = [m for m in all_measurements if m['is_cluster_origin']]
    total_volume = sum(m['volume_mm3'] for m in measurements) if r['thickness_mm'] is not None else None
    overgrown = [m for m in measurements if m['is_overgrown']]
    if r['mode'] == 'BACKLIGHT':
        headline_key, combined_key = 'optical_density_index', 'combined_optical_density'
    else:
        headline_key, combined_key = 'relative_intensity_index', 'combined_index'
    return {
        'count': len(measurements),
        'total_area': sum(m['true_area_mm2'] for m in measurements),
        'total_core': sum(m['display_core_mm2'] for m in measurements),
        'total_recovered': sum(m['display_recovered_mm2'] for m in measurements),
        'total_holes': sum(m['num_holes'] for m in measurements),
        'avg_circularity': (sum(m['circularity'] for m in measurements) / len(measurements)
                             if measurements else 0.0),
        'avg_headline': (sum(m[headline_key] for m in measurements) / len(measurements)
                          if measurements else 0.0),
        'total_combined': sum(m[combined_key] for m in measurements),
        'total_volume': total_volume,
        'has_small': any(m['is_small'] for m in measurements),
        'has_overgrown': len(overgrown) > 0,
        'max_overgrowth_ratio': max((m['overgrowth_ratio'] for m in overgrown), default=None),
        'unconfirmed_count': len(unconfirmed),
        'unconfirmed_total_area': sum(m['true_area_mm2'] for m in unconfirmed),
        'has_unconfirmed_cluster': len(unconfirmed) > 0,
    }


def _headline_col_labels(mode):
    if mode == 'BACKLIGHT':
        return 'Optical density\nindex (unitless)', 'Combined optical\ndensity (mm2)'
    return 'Relative intensity\nindex (unitless)', 'Combined index\n(mm2)'


def _headline_values(m, mode):
    if mode == 'BACKLIGHT':
        return m['optical_density_index'], m['combined_optical_density']
    return m['relative_intensity_index'], m['combined_index']


def display_results(r, filename=None):
    measurements = r['measurements']
    mode = r['mode']
    mode_label = "Backlight" if mode == 'BACKLIGHT' else "Reflected Light"
    name_part = f"{filename}  |  " if filename else ""

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(
        f"Plasma Aggregation Analysis ({mode_label}) v{VERSION} [{BUILD_TAG}]  |  {name_part}"
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

        headline_label, combined_label = _headline_col_labels(mode)
        col_labels = ['#', 'Scale\n(mm/px)', 'Area\n(mm2)', 'Core area\n(mm2)',
                      'Recovered area\n(mm2)', 'Holes\n(count)',
                      'Circularity\n(unitless)', headline_label,
                      combined_label, 'Volume\n(mm3)',
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
            headline_val, combined_val = _headline_values(m, mode)
            cell_text.append([
                id_str, scale_str, f"{m['true_area_mm2']:.3f}",
                f"{m['display_core_mm2']:.3f}", f"{m['display_recovered_mm2']:.3f}",
                f"{m['num_holes']}",
                f"{m['circularity']:.3f}", f"{headline_val:.3f}",
                f"{combined_val:.3f}", vol_str,
                frag_str, fill_str,
            ])
        total_vol_str = f"{summary['total_volume']:.3f}" if summary['total_volume'] is not None else "TBD"
        total_label = 'TOTAL' if not any_cluster else 'TOTAL (confirmed only)'
        cell_text.append([
            total_label, scale_str, f"{summary['total_area']:.3f}",
            f"{summary['total_core']:.3f}", f"{summary['total_recovered']:.3f}",
            f"{summary['total_holes']}",
            f"{summary['avg_circularity']:.3f}", f"{summary['avg_headline']:.3f}",
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
    batch_mode = next((r['mode'] for r in results if r.get('success')), 'REFLECTED')
    headline_label, combined_label = _headline_col_labels(batch_mode)
    col_labels = ['Photo', 'Scale\n(mm/px)', 'Area\n(mm2)', 'Core area\n(mm2)',
                  'Recovered area\n(mm2)', 'Holes\n(count)',
                  'Circularity\n(unitless)', headline_label,
                  combined_label, 'Volume\n(mm3)']
    RECOVERED_COL = 4
    cell_text = []
    numeric_rows = []
    scales = []
    overgrown_row_indices = []

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
            overgrown_row_indices.append(len(cell_text) + 1)
        elif summary['has_small']:
            name_str += "*"
        if summary['has_unconfirmed_cluster']:
            name_str += f" [{summary['unconfirmed_count']} CLUSTER?]"
        cell_text.append([
            name_str, scale_str, f"{summary['total_area']:.3f}",
            f"{summary['total_core']:.3f}", f"{summary['total_recovered']:.3f}",
            f"{summary['total_holes']}",
            f"{summary['avg_circularity']:.3f}", f"{summary['avg_headline']:.3f}",
            f"{summary['total_combined']:.3f}", vol_str,
        ])
        numeric_rows.append(summary)

    if numeric_rows or scales:
        areas = [s['total_area'] for s in numeric_rows]
        cores = [s['total_core'] for s in numeric_rows]
        recovered = [s['total_recovered'] for s in numeric_rows]
        holes = [s['total_holes'] for s in numeric_rows]
        circs = [s['avg_circularity'] for s in numeric_rows]
        headlines = [s['avg_headline'] for s in numeric_rows]
        combined = [s['total_combined'] for s in numeric_rows]
        vols = [s['total_volume'] for s in numeric_rows if s['total_volume'] is not None]

        def stat_or_dash(fn, vals, nd=3):
            return f"{fn(vals):.{nd}f}" if vals else '-'

        cell_text.append([
            'AVG', stat_or_dash(np.mean, scales, 5), stat_or_dash(np.mean, areas),
            stat_or_dash(np.mean, cores), stat_or_dash(np.mean, recovered),
            stat_or_dash(np.mean, holes, 1), stat_or_dash(np.mean, circs),
            stat_or_dash(np.mean, headlines), stat_or_dash(np.mean, combined),
            stat_or_dash(np.mean, vols) if vols else "TBD",
        ])
        cell_text.append([
            'STD', stat_or_dash(np.std, scales, 5), stat_or_dash(np.std, areas),
            stat_or_dash(np.std, cores), stat_or_dash(np.std, recovered),
            stat_or_dash(np.std, holes, 1), stat_or_dash(np.std, circs),
            stat_or_dash(np.std, headlines), stat_or_dash(np.std, combined),
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



if __name__ == "__main__":
    print("=" * 60)
    print(f"PLASMA AGGREGATION ANALYSIS  v{VERSION}  [build: {BUILD_TAG}]")
    print(f"Master CSV log: {MASTER_CSV_PATH}")
    print("=" * 60)

    params = get_parameters()
    image_paths = params['image_paths']
    mode = params['mode']

    print(f"\n{len(image_paths)} image(s) selected")
    print(f"Lighting setup: {'Backlight' if mode == 'BACKLIGHT' else 'Reflected Light'}")
    print(f"Counting floor: {MINIMUM_AREA_MM2[mode]}mm2  |  Confident floor (no '*' mark): {CONFIDENT_AREA_MM2}mm2")
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
            result = {'success': False, 'mode': mode, 'reason': f"Could not read image file:\n{path}"}
            all_results.append(result)
            append_photo_to_master_csv(name, params, result)
            display_failure(result['reason'], name)
            continue

        result = analyze_image(image, params['reference_mm'], params['color_mode'], params['thickness_mm'], mode)
        all_results.append(result)

        if not result['success']:
            print(f"CALIBRATION FAILED: {result['reason']}")
            append_photo_to_master_csv(name, params, result)
            display_failure(result['reason'], name)
            continue

        print(f"Calibration: {result['line_px']:.0f}px = {params['reference_mm']}mm "
              f"(1px = {result['mm_per_px']:.5f}mm)")
        print(f"Otsu threshold (core, flattened): {result['otsu_threshold']:.0f}")
        hd = result['hsv_diagnostics']
        print(f"Adaptive HSV floors used: saturation>={hd['saturation_min']:.0f} "
              f"[99th pct S={hd['s_ref']:.0f}], value>={hd['value_min']:.0f} "
              f"[99th pct V={hd['v_ref']:.0f}]")
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
                      "not a solid blob.")
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
