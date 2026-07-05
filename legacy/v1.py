# ARCHIVED: superseded by ../src/pipeline.py. Kept for iteration history; may reference paths relative to the old repo layout.
import cv2
import numpy as np
import matplotlib.pyplot as plt

# =============================================
# PLASMA PROTEIN AGGREGATION ANALYSIS TOOL
# Version 2.0 — Single Image, No Background Subtraction
# =============================================
# WHAT CHANGED FROM V1:
# - Removed before/after images — only ONE image needed now
# - Removed background subtraction (absdiff) — not applicable
# - Added Gaussian blur before thresholding — reduces halo/noise
# - Added morphological opening — removes small speckles
# - Added morphological closing — fills holes inside aggregate
# - Changed to THRESH_BINARY (not INV) — aggregate is WHITE on BLACK
# - Display updated from 6 panels to 5 panels (no diff image needed)
# =============================================

# =====================
# SETTINGS — change these to match your image
# =====================
IMAGE_PATH   = 'Plasma/Test3.JPG'   # your single image file
MINIMUM_AREA = 10000                  # pixels — increase to ignore small noise
BLUR_SIZE    = 15                     # must be odd number (5, 7, 9...) — higher = more blur
MORPH_SIZE   = 10                   # size of noise removal kernel — higher = more aggressive
MANUAL_THRESHOLD = 125

# =====================
# STEP 1: LOAD IMAGE
# =====================
print("=" * 50)
print("PLASMA AGGREGATION ANALYSIS  v2.0")
print("=" * 50)

image = cv2.imread(IMAGE_PATH)

if image is None:
    print("ERROR: Image not found!")
    print(f"  Looking for: {IMAGE_PATH}")
else:
    print(f"Image loaded successfully")
    print(f"  Size: {image.shape}")

    # =====================
    # STEP 2: GRAYSCALE
    # =====================
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print("\nConverted to grayscale")

    # =====================
    # STEP 3: GAUSSIAN BLUR  [NEW in V2]
    # =====================
    # Smooths out the translucent halo and small speckles
    # before thresholding so they don't get detected as aggregate
    blurred = cv2.GaussianBlur(gray, (BLUR_SIZE, BLUR_SIZE), 0)
    print(f"\nGaussian blur applied (kernel size: {BLUR_SIZE}x{BLUR_SIZE})")

    # =====================
    # STEP 4: THRESHOLDING (OTSU)  [CHANGED in V2]
    # =====================
    # THRESH_BINARY (not INV) because aggregate = WHITE on BLACK background

    #threshold_value, binary = cv2.threshold(
    #    blurred,
    #    0,
    #    255,
    #    cv2.THRESH_BINARY + cv2.THRESH_OTSU
    #)

    threshold_value = MANUAL_THRESHOLD
    _ , binary = cv2.threshold(
        blurred,
        MANUAL_THRESHOLD,
        255,
        cv2.THRESH_BINARY
    )
    
    print(f"\nOtsu thresholding done")
    print(f"  Threshold value chosen: {threshold_value}")
    print(f"  (pixels brighter than {threshold_value} = white = aggregate)")

    # =====================
    # STEP 5: MORPHOLOGICAL NOISE REDUCTION  [NEW in V2]
    # =====================
    kernel = np.ones((MORPH_SIZE, MORPH_SIZE), np.uint8)

    # Opening = erode then dilate
    # removes tiny bright speckles (salt noise, halo edges)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Closing = dilate then erode
    # fills small dark holes inside the aggregate clump
    cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    print(f"\nMorphological cleaning done (kernel: {MORPH_SIZE}x{MORPH_SIZE})")
    print(f"  Opening: removed small bright speckles")
    print(f"  Closing: filled holes inside aggregate")

    # =====================
    # STEP 6: FIND CONTOURS
    # =====================
    contours, hierarchy = cv2.findContours(
        cleaned,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    print(f"\nContours found: {len(contours)} total")

    # =====================
    # STEP 7: FILTER NOISE BY SIZE
    # =====================
    real_aggregates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > MINIMUM_AREA:
            real_aggregates.append(contour)

    print(f"Real aggregates (area > {MINIMUM_AREA}px): {len(real_aggregates)}")

    # =====================
    # STEP 8: MEASURE EACH AGGREGATE
    # =====================
    print("\n" + "=" * 50)
    print("MEASUREMENTS")
    print("=" * 50)

    all_measurements = []

    for i, contour in enumerate(real_aggregates):

        # Area
        area = cv2.contourArea(contour)

        # Perimeter
        perimeter = cv2.arcLength(contour, True)

        # Circularity
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
        else:
            circularity = 0

        # Bounding box
        x, y, w, h = cv2.boundingRect(contour)

        # Center point
        M = cv2.moments(contour)
        if M['m00'] > 0:
            center_x = int(M['m10'] / M['m00'])
            center_y = int(M['m01'] / M['m00'])
        else:
            center_x, center_y = x, y

        # Area fraction = aggregate area / total image area
        total_pixels = image.shape[0] * image.shape[1]
        area_fraction = area / total_pixels

        measurements = {
            'id':           i + 1,
            'area':         area,
            'perimeter':    perimeter,
            'circularity':  circularity,
            'area_fraction': area_fraction,
            'center_x':     center_x,
            'center_y':     center_y,
            'bbox_x':       x,
            'bbox_y':       y,
            'bbox_w':       w,
            'bbox_h':       h
        }
        all_measurements.append(measurements)

        print(f"\nAggregate {i+1}:")
        print(f"  Area:         {area:.1f} px²")
        print(f"  Perimeter:    {perimeter:.1f} px")
        print(f"  Circularity:  {circularity:.3f}  (1.0 = perfect circle)")
        print(f"  Area fraction:{area_fraction:.4f}  ({area_fraction*100:.2f}% of image)")
        print(f"  Center:       ({center_x}, {center_y})")
        print(f"  Bounding box: x={x}, y={y}, w={w}, h={h}")

    # Summary
    if len(all_measurements) > 0:
        total_area     = sum(m['area'] for m in all_measurements)
        avg_circ       = sum(m['circularity'] for m in all_measurements) / len(all_measurements)
        largest        = max(all_measurements, key=lambda m: m['area'])
        total_fraction = sum(m['area_fraction'] for m in all_measurements)

        print("\n" + "=" * 50)
        print("SUMMARY")
        print("=" * 50)
        print(f"  Total aggregates found:  {len(all_measurements)}")
        print(f"  Total aggregate area:    {total_area:.1f} px²")
        print(f"  Total area fraction:     {total_fraction*100:.2f}% of image")
        print(f"  Average circularity:     {avg_circ:.3f}")
        print(f"  Largest aggregate:       {largest['area']:.1f} px²  (aggregate {largest['id']})")
    else:
        print("\nNo aggregates detected")
        print("  Try: lower MINIMUM_AREA, lower BLUR_SIZE, or lower MORPH_SIZE")

    # =====================
    # STEP 9: DRAW RESULTS
    # =====================
    result_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(result_image, real_aggregates, -1, (0, 255, 0), 2)

    for m in all_measurements:
        cv2.circle(result_image, (m['center_x'], m['center_y']), 5, (0, 0, 255), -1)
        label = f"#{m['id']} | {m['area']:.0f}px"
        cv2.putText(
            result_image, label,
            (m['bbox_x'], m['bbox_y'] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1
        )

    # =====================
    # STEP 10: DISPLAY  [CHANGED in V2 — 5 panels, no diff image]
    # =====================
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Row 1
    axes[0,0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0,0].set_title('1. Original Image')
    axes[0,0].axis('off')

    axes[0,1].imshow(gray, cmap='gray')
    axes[0,1].set_title('2. Grayscale')
    axes[0,1].axis('off')

    axes[0,2].imshow(blurred, cmap='gray')
    axes[0,2].set_title(f'3. Blurred (kernel={BLUR_SIZE})')
    axes[0,2].axis('off')

    # Row 2
    axes[1,0].imshow(binary, cmap='gray')
    axes[1,0].set_title(f'4. Binary Otsu (threshold={threshold_value:.0f})')
    axes[1,0].axis('off')

    axes[1,1].imshow(cleaned, cmap='gray')
    axes[1,1].set_title(f'5. After Noise Removal (morph={MORPH_SIZE})')
    axes[1,1].axis('off')

    axes[1,2].imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    axes[1,2].set_title(f'6. Detected Aggregates ({len(real_aggregates)} found)')
    axes[1,2].axis('off')

    plt.suptitle('Plasma Protein Aggregation Analysis  v2.0', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    print("\nAnalysis complete!")