# ARCHIVED: superseded by ../src/pipeline.py. Kept for iteration history; may reference paths relative to the old repo layout.
import cv2
import numpy as np
import matplotlib.pyplot as plt

# =============================================
# PLASMA PROTEIN AGGREGATION ANALYSIS TOOL
# Version 3.0 — Auto Calibration + Analysis
# =============================================
# WORKFLOW:
#   1. Paint glass plate edge with RED, GREEN, or BLUE marker
#   2. Set REFERENCE_MM to real length of that edge
#   3. Set COLOR_MODE to match your marker color
#   4. Run — two windows appear:
#      Window 1: Calibration result
#      Window 2: Analysis pipeline + final result
# =============================================

# =====================
# SETTINGS — only change these
# =====================

# --- Image ---
IMAGE_PATH        = 'Plasma/Used_Plasma_Pic/Senpai1.JPG'

# --- Calibration ---
REFERENCE_MM      = 18     # real length of painted edge (mm)
                              # change to 80.0, 100.0, etc. if different
COLOR_MODE        = 'RED'    # 'RED', 'GREEN', or 'BLUE'

# Color sensitivity — adjust if detection fails
RED_MIN_R         = 60
RED_MAX_G         = 90
RED_MAX_B         = 130

GREEN_MIN_G       = 90
GREEN_MAX_R       = 90
GREEN_MAX_B       = 95

BLUE_MIN_B        = 130
BLUE_MAX_R        = 100
BLUE_MAX_G        = 100

MIN_LINE_LENGTH_PX = 1000

# --- Analysis ---
BRIGHT_PERCENTILE = 98
BLUR_SIZE         = 15       # must be odd
MORPH_SIZE        = 3
MINIMUM_AREA_MM2  = 3

# =============================================
# DO NOT CHANGE BELOW THIS LINE
# =============================================

# =====================
# STEP 1: LOAD IMAGE
# =====================
print("=" * 60)
print("PLASMA AGGREGATION ANALYSIS  v3.0")
print("=" * 60)

image = cv2.imread(IMAGE_PATH)
if image is None:
    print(f"ERROR: Image not found at '{IMAGE_PATH}'")
    exit()
print(f"Image loaded: {image.shape[1]}w x {image.shape[0]}h px")

# =====================
# STEP 2: AUTO CALIBRATION
# =====================
print("\n--- Calibration ---")

b_ch = image[:, :, 0].astype(int)
g_ch = image[:, :, 1].astype(int)
r_ch = image[:, :, 2].astype(int)

if COLOR_MODE == 'RED':
    color_mask  = (r_ch > RED_MIN_R) & (g_ch < RED_MAX_G) & (b_ch < RED_MAX_B)
    color_label = 'Red'
    draw_color  = (0, 0, 255)
elif COLOR_MODE == 'GREEN':
    color_mask  = (g_ch > GREEN_MIN_G) & (r_ch < GREEN_MAX_R) & (b_ch < GREEN_MAX_B)
    color_label = 'Green'
    draw_color  = (0, 255, 0)
elif COLOR_MODE == 'BLUE':
    color_mask  = (b_ch > BLUE_MIN_B) & (r_ch < BLUE_MAX_R) & (g_ch < BLUE_MAX_G)
    color_label = 'Blue'
    draw_color  = (255, 0, 0)
else:
    print(f"ERROR: Unknown COLOR_MODE '{COLOR_MODE}' — use 'RED', 'GREEN', or 'BLUE'")
    exit()

mask_image           = color_mask.astype(np.uint8) * 255
total_colored_pixels = np.sum(color_mask)
print(f"  {color_label} pixels detected: {total_colored_pixels}")

cal_contours, _ = cv2.findContours(
    mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

line_candidates = []
for contour in cal_contours:
    if cv2.contourArea(contour) < 5:
        continue
    rect                = cv2.minAreaRect(contour)
    center, dims, angle = rect
    length_px           = max(dims[0], dims[1])
    thickness_px        = min(dims[0], dims[1])
    if length_px > MIN_LINE_LENGTH_PX:
        line_candidates.append({
            'rect':        rect,
            'length_px':   length_px,
            'thickness_px': thickness_px,
            'center':      center
        })

if len(line_candidates) == 0:
    print(f"  WARNING: No {color_label} line found — using fallback 601px")
    line_px     = 601
    cal_success = False
else:
    best        = max(line_candidates, key=lambda c: c['length_px'])
    line_px     = best['length_px']
    cal_success = True
    print(f"  Line detected: {line_px:.0f}px = {REFERENCE_MM}mm")
    print(f"  Thickness (ignored): {best['thickness_px']:.0f}px")

mm_per_px   = REFERENCE_MM / line_px
mm2_per_px2 = mm_per_px ** 2
px_per_mm   = line_px / REFERENCE_MM
print(f"  Scale: 1px = {mm_per_px:.5f}mm")

# Build calibration overlay image
cal_result = image.copy()
overlay    = cal_result.copy()
overlay[color_mask] = draw_color
cv2.addWeighted(overlay, 0.4, cal_result, 0.6, 0, cal_result)

if cal_success:
    box = np.int32(cv2.boxPoints(best['rect']))
    cv2.drawContours(cal_result, [box], 0, (0, 255, 255), 3)
    cx, cy = int(best['center'][0]), int(best['center'][1])
    cv2.putText(cal_result, f"{line_px:.0f}px = {REFERENCE_MM}mm",
        (cx - 150, cy - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
    cv2.putText(cal_result, f"1px = {mm_per_px:.4f}mm",
        (cx - 150, cy + 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

# =====================
# STEP 3: GRAYSCALE
# =====================
print("\n--- Analysis ---")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(f"  Brightness range: {gray.min()} - {gray.max()}")

# =====================
# STEP 4: BLUR
# =====================
blurred = cv2.GaussianBlur(gray, (BLUR_SIZE, BLUR_SIZE), 0)

# =====================
# STEP 5: THRESHOLD
# =====================
auto_threshold = int(np.percentile(gray.flatten(), BRIGHT_PERCENTILE))
print(f"  Auto threshold ({BRIGHT_PERCENTILE}th percentile): {auto_threshold}")
_, binary = cv2.threshold(blurred, auto_threshold, 255, cv2.THRESH_BINARY)

# =====================
# STEP 6: NOISE REMOVAL
# =====================
kernel  = np.ones((MORPH_SIZE, MORPH_SIZE), np.uint8)
opened  = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel)
cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

# =====================
# STEP 7: CONTOURS + HOLES
# =====================
contours, hierarchy = cv2.findContours(
    cleaned, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
)

MINIMUM_AREA_PX = MINIMUM_AREA_MM2 / mm2_per_px2
outer_contours  = []
hole_contours   = []

if hierarchy is not None:
    for i, contour in enumerate(contours):
        parent_idx = hierarchy[0][i][3]
        area_px    = cv2.contourArea(contour)
        if parent_idx == -1:
            if area_px > MINIMUM_AREA_PX:
                outer_contours.append((contour, i))
        else:
            hole_contours.append((contour, parent_idx))

print(f"  Outer aggregates: {len(outer_contours)}")
print(f"  Holes inside:     {len(hole_contours)}")

# =====================
# STEP 8: MEASURE
# =====================
print("\n" + "=" * 60)
print("MEASUREMENTS")
print("=" * 60)

all_measurements = []

for i, (contour, contour_idx) in enumerate(outer_contours):

    outer_area_px = cv2.contourArea(contour)
    my_holes      = [h for h, pidx in hole_contours if pidx == contour_idx]
    hole_area_px  = sum(cv2.contourArea(h) for h in my_holes)
    true_area_px  = outer_area_px - hole_area_px
    true_area_mm2 = true_area_px * mm2_per_px2
    perimeter_px  = cv2.arcLength(contour, True)
    perimeter_mm  = perimeter_px * mm_per_px
    circularity   = (4 * np.pi * true_area_px / (perimeter_px ** 2)
                     if perimeter_px > 0 else 0)
    x, y, w, h    = cv2.boundingRect(contour)
    M             = cv2.moments(contour)
    center_x      = int(M['m10'] / M['m00']) if M['m00'] > 0 else x
    center_y      = int(M['m01'] / M['m00']) if M['m00'] > 0 else y
    area_fraction = true_area_px / (image.shape[0] * image.shape[1])

    m = {
        'id': i+1, 'outer_area_px': outer_area_px,
        'hole_area_px': hole_area_px, 'true_area_px': true_area_px,
        'true_area_mm2': true_area_mm2, 'num_holes': len(my_holes),
        'perimeter_mm': perimeter_mm, 'circularity': circularity,
        'area_fraction': area_fraction, 'center_x': center_x,
        'center_y': center_y, 'bbox_x': x, 'bbox_y': y,
        'bbox_w': w, 'bbox_h': h
    }
    all_measurements.append(m)

    print(f"\nAggregate {i+1}:")
    print(f"  Outer area:    {outer_area_px:.0f} px²")
    print(f"  Holes:         {len(my_holes)} = {hole_area_px:.0f} px²")
    print(f"  TRUE area:     {true_area_px:.0f} px²  =  {true_area_mm2:.4f} mm²")
    print(f"  Perimeter:     {perimeter_mm:.2f} mm")
    print(f"  Circularity:   {circularity:.3f}")
    print(f"  Area fraction: {area_fraction*100:.4f}% of image")

if len(all_measurements) > 0:
    total_mm2 = sum(m['true_area_mm2'] for m in all_measurements)
    avg_circ  = sum(m['circularity']   for m in all_measurements) / len(all_measurements)
    largest   = max(all_measurements,  key=lambda m: m['true_area_mm2'])
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Aggregates found:    {len(all_measurements)}")
    print(f"  Total TRUE area:     {total_mm2:.4f} mm²")
    print(f"  Average circularity: {avg_circ:.3f}")
    print(f"  Largest aggregate:   {largest['true_area_mm2']:.4f} mm²")
    print(f"  Scale: {line_px:.0f}px = {REFERENCE_MM}mm")
else:
    print("\nNo aggregates detected")
    print("  Try: lower BRIGHT_PERCENTILE or lower MINIMUM_AREA_MM2")

# =====================
# STEP 9: DRAW RESULTS
# =====================
result_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
outer_only   = [c for c, _ in outer_contours]
hole_only    = [h for h, _ in hole_contours]
cv2.drawContours(result_image, outer_only, -1, (0, 255, 0), 2)
cv2.drawContours(result_image, hole_only,  -1, (0, 0, 255), 2)

for m in all_measurements:
    cv2.circle(result_image, (m['center_x'], m['center_y']), 6, (0, 255, 255), -1)
    cv2.putText(result_image, f"#{m['id']} {m['true_area_mm2']:.3f}mm2",
        (m['bbox_x'], m['bbox_y'] - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

# =====================
# WINDOW 1: CALIBRATION
# =====================
fig1, cal_axes = plt.subplots(1, 3, figsize=(20, 8))
fig1.suptitle(
    f'Window 1 — Auto Calibration ({color_label} Line)  |  '
    f'Reference = {REFERENCE_MM}mm  |  '
    f'1px = {mm_per_px:.4f}mm  |  '
    f'Status: {"Auto ✓" if cal_success else "Fallback !"}',
    fontsize=12, fontweight='bold')

cal_axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
cal_axes[0].set_title('1. Original Image')
cal_axes[0].axis('off')

cal_axes[1].imshow(mask_image, cmap='gray')
cal_axes[1].set_title(f'2. {color_label} Pixel Mask\n({total_colored_pixels} pixels detected)')
cal_axes[1].axis('off')

cal_axes[2].imshow(cv2.cvtColor(cal_result, cv2.COLOR_BGR2RGB))
cal_axes[2].set_title(f'3. Detected Line\n{line_px:.0f}px = {REFERENCE_MM}mm')
cal_axes[2].axis('off')

cal_axes[2].text(0.02, 0.02,
    f"CALIBRATION RESULT\n"
    f"{'─'*28}\n"
    f"Line:    {line_px:.0f} px\n"
    f"Real:    {REFERENCE_MM} mm\n"
    f"1 px  =  {mm_per_px:.5f} mm\n"
    f"1 mm  =  {px_per_mm:.2f} px\n"
    f"1 px² =  {mm2_per_px2:.7f} mm²\n"
    f"Status: {'Auto ✓' if cal_success else 'Fallback !'}",
    transform=cal_axes[2].transAxes, fontsize=9,
    verticalalignment='bottom',
    bbox=dict(boxstyle='round', facecolor='black', alpha=0.8),
    color='white', fontfamily='monospace')

plt.tight_layout()

# =====================
# WINDOW 2: ANALYSIS
# =====================
fig2, axes = plt.subplots(2, 3, figsize=(20, 14))
fig2.suptitle(
    f'Window 2 — Analysis  |  '
    f'Threshold={auto_threshold} (percentile={BRIGHT_PERCENTILE})  |  '
    f'{len(outer_contours)} aggregate(s) found',
    fontsize=12, fontweight='bold')

axes[0,0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0,0].set_title('1. Original')
axes[0,0].axis('off')

axes[0,1].imshow(gray, cmap='gray')
axes[0,1].set_title('2. Grayscale')
axes[0,1].axis('off')

axes[0,2].imshow(blurred, cmap='gray')
axes[0,2].set_title(f'3. Blurred (kernel={BLUR_SIZE})')
axes[0,2].axis('off')

axes[1,0].imshow(binary, cmap='gray')
axes[1,0].set_title(f'4. Binary (threshold={auto_threshold})')
axes[1,0].axis('off')

axes[1,1].imshow(cleaned, cmap='gray')
axes[1,1].set_title(f'5. Noise Removal (morph={MORPH_SIZE})')
axes[1,1].axis('off')

axes[1,2].imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
axes[1,2].set_title(f'6. Final Result ({len(outer_contours)} aggregates)')
axes[1,2].axis('off')

if len(all_measurements) > 0:
    axes[1,2].text(0.02, 0.02,
        f"SUMMARY\n"
        f"{'─'*30}\n"
        f"Aggregates:   {len(all_measurements)}\n"
        f"Total area:   {total_mm2:.4f} mm²\n"
        f"Largest:      {largest['true_area_mm2']:.4f} mm²\n"
        f"Avg circ:     {avg_circ:.3f}\n"
        f"{'─'*30}\n"
        f"Scale: {line_px:.0f}px={REFERENCE_MM}mm\n"
        f"1px = {mm_per_px:.5f}mm\n"
        f"{'─'*30}\n"
        f"Green = aggregate\n"
        f"Red   = holes",
        transform=axes[1,2].transAxes, fontsize=9,
        verticalalignment='bottom',
        bbox=dict(boxstyle='round', facecolor='black', alpha=0.75),
        color='white', fontfamily='monospace')

plt.tight_layout()
plt.show()

print("\nAnalysis complete!")