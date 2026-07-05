# ARCHIVED: superseded by ../src/pipeline.py. Kept for iteration history; may reference paths relative to the old repo layout.
import cv2
import numpy as np
import matplotlib.pyplot as plt

# =============================================
# PLASMA PROTEIN AGGREGATION ANALYSIS TOOL
# Version 7.0 — No ROI, No Crop, Full Image
# =============================================
# SIMPLE AND CLEAN:
#   1. Load full image
#   2. Grayscale
#   3. Blur
#   4. Relative threshold (percentile-based, auto)
#   5. Noise removal
#   6. Find contours + holes
#   7. Measure (true area = outer minus holes)
#   8. Two windows: pipeline + final result
# =============================================

# =====================
# SETTINGS — only change these
# =====================
IMAGE_PATH        = 'Plasma/Test3.JPG'
SCALE_BAR_PX      = 800      # measure red scale bar in pixels
SCALE_BAR_MM      = 50    # real length — never changes
BRIGHT_PERCENTILE = 90      # lower = catches more aggregate (try 35-50)
BLUR_SIZE         = 15       # must be odd number
MORPH_SIZE        = 3        # noise removal — keep small to not erase tiny aggregates
MINIMUM_AREA_MM2  = 60     # smallest blob to count (mm²)

# =============================================
# DO NOT CHANGE BELOW THIS LINE
# =============================================

# =====================
# STEP 1: LOAD
# =====================
print("=" * 60)
print("PLASMA AGGREGATION ANALYSIS  v7.0")
print("=" * 60)

image = cv2.imread(IMAGE_PATH)
if image is None:
    print(f"ERROR: Image not found at '{IMAGE_PATH}'")
    exit()

print(f"Image loaded: {image.shape[1]}w x {image.shape[0]}h px")

mm_per_px   = SCALE_BAR_MM / SCALE_BAR_PX
mm2_per_px2 = mm_per_px ** 2
print(f"Scale: {SCALE_BAR_PX}px = {SCALE_BAR_MM}mm → {mm_per_px:.4f} mm/px")

# =====================
# STEP 2: GRAYSCALE
# =====================
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(f"\nGrayscale done")
print(f"  Brightness range: {gray.min()} - {gray.max()}")

# =====================
# STEP 3: BLUR
# =====================
blurred = cv2.GaussianBlur(gray, (BLUR_SIZE, BLUR_SIZE), 0)
print(f"\nBlur done (kernel={BLUR_SIZE})")

# =====================
# STEP 4: RELATIVE THRESHOLD
# =====================
# Percentile of ALL pixels in the image
# Lower percentile = lower threshold = catches more aggregate
all_pixels     = gray.flatten()
auto_threshold = int(np.percentile(all_pixels, BRIGHT_PERCENTILE))

print(f"\nThreshold done")
print(f"  Auto threshold ({BRIGHT_PERCENTILE}th percentile): {auto_threshold}")

_, binary = cv2.threshold(blurred, auto_threshold, 255, cv2.THRESH_BINARY)

# =====================
# STEP 5: NOISE REMOVAL
# =====================
kernel  = np.ones((MORPH_SIZE, MORPH_SIZE), np.uint8)
opened  = cv2.morphologyEx(binary,  cv2.MORPH_OPEN,  kernel)
cleaned = cv2.morphologyEx(opened,  cv2.MORPH_CLOSE, kernel)
print(f"\nNoise removal done (morph={MORPH_SIZE})")

# =====================
# STEP 6: FIND CONTOURS + HOLES
# =====================
contours, hierarchy = cv2.findContours(
    cleaned, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
)
print(f"\nContours found: {len(contours)} total")

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
# STEP 7: MEASURE
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

    if perimeter_px > 0:
        circularity = 4 * np.pi * true_area_px / (perimeter_px ** 2)
    else:
        circularity = 0

    x, y, w, h = cv2.boundingRect(contour)
    M           = cv2.moments(contour)
    if M['m00'] > 0:
        center_x = int(M['m10'] / M['m00'])
        center_y = int(M['m01'] / M['m00'])
    else:
        center_x, center_y = x, y

    total_image_px = image.shape[0] * image.shape[1]
    area_fraction  = true_area_px / total_image_px

    m = {
        'id':            i + 1,
        'outer_area_px': outer_area_px,
        'hole_area_px':  hole_area_px,
        'true_area_px':  true_area_px,
        'true_area_mm2': true_area_mm2,
        'num_holes':     len(my_holes),
        'perimeter_mm':  perimeter_mm,
        'circularity':   circularity,
        'area_fraction': area_fraction,
        'center_x':      center_x,
        'center_y':      center_y,
        'bbox_x': x, 'bbox_y': y, 'bbox_w': w, 'bbox_h': h
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
    print(f"  Scale: {SCALE_BAR_PX}px = {SCALE_BAR_MM}mm")
else:
    print("\nNo aggregates detected")
    print("  Try: lower BRIGHT_PERCENTILE or lower MINIMUM_AREA_MM2")

# =====================
# STEP 8: DRAW RESULTS
# =====================
result_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

# Green = aggregate outer boundary
outer_only = [c for c, _ in outer_contours]
cv2.drawContours(result_image, outer_only, -1, (0, 255, 0), 2)

# Red = holes inside aggregate
hole_only = [h for h, _ in hole_contours]
cv2.drawContours(result_image, hole_only, -1, (0, 0, 255), 2)

# Yellow dot + label for each aggregate
for m in all_measurements:
    cv2.circle(result_image,
               (m['center_x'], m['center_y']), 6, (0, 255, 255), -1)
    label = f"#{m['id']} {m['true_area_mm2']:.3f}mm2"
    cv2.putText(result_image, label,
        (m['bbox_x'], m['bbox_y'] - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

# =====================
# WINDOW 1: PIPELINE
# =====================
fig1, axes = plt.subplots(2, 3, figsize=(18, 12))
fig1.suptitle('Plasma Protein Aggregation Analysis v7.0 — Pipeline',
              fontsize=13, fontweight='bold')

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
axes[1,0].set_title(f'4. Binary (threshold={auto_threshold}, percentile={BRIGHT_PERCENTILE})')
axes[1,0].axis('off')

axes[1,1].imshow(cleaned, cmap='gray')
axes[1,1].set_title(f'5. After Noise Removal (morph={MORPH_SIZE})')
axes[1,1].axis('off')

axes[1,2].imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
axes[1,2].set_title(f'6. Preview ({len(outer_contours)} aggregates found)')
axes[1,2].axis('off')

plt.tight_layout()

# =====================
# WINDOW 2: FINAL RESULT
# =====================
fig2, ax = plt.subplots(1, 1, figsize=(10, 14))
fig2.suptitle('Plasma Protein Aggregation Analysis v7.0 — Final Result',
              fontsize=13, fontweight='bold')

ax.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
ax.axis('off')

if len(all_measurements) > 0:
    summary_text = (
        f"Aggregates found:    {len(all_measurements)}\n"
        f"Total TRUE area:     {total_mm2:.4f} mm²\n"
        f"Largest aggregate:   {largest['true_area_mm2']:.4f} mm²\n"
        f"Avg circularity:     {avg_circ:.3f}\n"
        f"Scale: {SCALE_BAR_PX}px = {SCALE_BAR_MM}mm\n"
        f"Threshold: {auto_threshold} (percentile={BRIGHT_PERCENTILE})\n\n"
        f"Green = aggregate boundary\n"
        f"Red   = holes (excluded from area)"
    )
    ax.text(
        0.02, 0.02, summary_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='bottom',
        bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
        color='white',
        fontfamily='monospace'
    )

plt.tight_layout()
plt.show()

print("\nAnalysis complete!")