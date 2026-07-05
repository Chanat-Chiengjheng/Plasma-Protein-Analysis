# ARCHIVED: kept for history. References image files (Test_Before.png / Test_After.png) that no longer exist anywhere in this repo, so this script is not currently runnable as-is.
import cv2
import numpy as np
import matplotlib.pyplot as plt

# =============================================
# PLASMA PROTEIN AGGREGATION ANALYSIS TOOL
# Version 1.0 — Single File
# =============================================

# =====================
# SETTINGS — change these to match your images
# =====================
BEFORE_IMAGE = 'Plasma/Test_Before.png'
AFTER_IMAGE  = 'Plasma/Test_After.png'
MINIMUM_AREA = 100   # pixels — increase if too much noise detected

# =====================
# STEP 1: LOAD IMAGES
# =====================
print("=" * 50)
print("PLASMA AGGREGATION ANALYSIS")
print("=" * 50)

before = cv2.imread(BEFORE_IMAGE)
after  = cv2.imread(AFTER_IMAGE)

if before is None or after is None:
    print("ERROR: One or both images not found!")
    print(f"  Looking for: {BEFORE_IMAGE}")
    print(f"  Looking for: {AFTER_IMAGE}")
else:
    print(f"Images loaded successfully")
    print(f"  Before: {before.shape}")
    print(f"  After:  {after.shape}")

    # =====================
    # STEP 2: GRAYSCALE
    # =====================
    before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    after_gray  = cv2.cvtColor(after,  cv2.COLOR_BGR2GRAY)
    print("\nConverted to grayscale")

    # =====================
    # STEP 3: RESIZE IF NEEDED
    # =====================
    if before_gray.shape != after_gray.shape:
        print(f"\nSize mismatch detected — resizing after image")
        print(f"  Before: {before_gray.shape}")
        print(f"  After:  {after_gray.shape}")
        after_gray = cv2.resize(
            after_gray,
            (before_gray.shape[1], before_gray.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )
        print(f"  After resized: {after_gray.shape}")
    else:
        print("\nImage sizes match — no resize needed")

    # =====================
    # STEP 4: BACKGROUND SUBTRACTION
    # =====================
    diff = cv2.absdiff(after_gray, before_gray)
    print(f"\nBackground subtraction done")
    print(f"  Max difference value: {diff.max()}")
    print(f"  Min difference value: {diff.min()}")

    # =====================
    # STEP 5: THRESHOLDING (OTSU)
    # =====================
    threshold_value, binary = cv2.threshold(
        diff,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    print(f"\nOtsu thresholding done")
    print(f"  Threshold value chosen: {threshold_value}")

    # =====================
    # STEP 6: FIND CONTOURS
    # =====================
    contours, hierarchy = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    print(f"\nContours found: {len(contours)} total")

    # =====================
    # STEP 7: FILTER NOISE
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

    all_measurements = []  # store all results here

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

        # Store measurements
        measurements = {
            'id':          i + 1,
            'area':        area,
            'perimeter':   perimeter,
            'circularity': circularity,
            'center_x':    center_x,
            'center_y':    center_y,
            'bbox_x':      x,
            'bbox_y':      y,
            'bbox_w':      w,
            'bbox_h':      h
        }
        all_measurements.append(measurements)

        # Print
        print(f"\nAggregate {i+1}:")
        print(f"  Area:        {area:.1f} px²")
        print(f"  Perimeter:   {perimeter:.1f} px")
        print(f"  Circularity: {circularity:.3f}  (1.0 = perfect circle)")
        print(f"  Center:      ({center_x}, {center_y})")
        print(f"  Bounding box: x={x}, y={y}, w={w}, h={h}")

    # Summary statistics
    if len(all_measurements) > 0:
        total_area = sum(m['area'] for m in all_measurements)
        avg_circularity = sum(m['circularity'] for m in all_measurements) / len(all_measurements)
        largest = max(all_measurements, key=lambda m: m['area'])

        print("\n" + "=" * 50)
        print("SUMMARY")
        print("=" * 50)
        print(f"  Total aggregates found:  {len(all_measurements)}")
        print(f"  Total aggregate area:    {total_area:.1f} px²")
        print(f"  Average circularity:     {avg_circularity:.3f}")
        print(f"  Largest aggregate:       {largest['area']:.1f} px² (aggregate {largest['id']})")
    else:
        print("\nNo aggregates detected — try lowering MINIMUM_AREA")

    # =====================
    # STEP 9: DRAW RESULTS ON IMAGE
    # =====================
    result_image = cv2.cvtColor(after_gray, cv2.COLOR_GRAY2BGR)

    # Draw contours in green
    cv2.drawContours(result_image, real_aggregates, -1, (0, 255, 0), 2)

    # Draw center and label for each aggregate
    for m in all_measurements:
        # Red dot at center
        cv2.circle(result_image, (m['center_x'], m['center_y']), 5, (0, 0, 255), -1)

        # Label with aggregate number and area
        label = f"#{m['id']} | {m['area']:.0f}px"
        cv2.putText(
            result_image,
            label,
            (m['bbox_x'], m['bbox_y'] - 10),  # position: just above bounding box
            cv2.FONT_HERSHEY_SIMPLEX,          # font style
            0.5,                               # font size
            (0, 255, 255),                     # color: yellow
            1                                  # thickness
        )

    # =====================
    # STEP 10: DISPLAY EVERYTHING
    # =====================
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    # 2 rows, 3 columns

    # Row 1
    axes[0,0].imshow(cv2.cvtColor(before, cv2.COLOR_BGR2RGB))
    axes[0,0].set_title('1. Before (Original)')
    axes[0,0].axis('off')

    axes[0,1].imshow(cv2.cvtColor(after, cv2.COLOR_BGR2RGB))
    axes[0,1].set_title('2. After (Original)')
    axes[0,1].axis('off')

    axes[0,2].imshow(diff, cmap='gray')
    axes[0,2].set_title('3. Difference Image')
    axes[0,2].axis('off')

    # Row 2
    axes[1,0].imshow(binary, cmap='gray')
    axes[1,0].set_title(f'4. Binary (Otsu={threshold_value:.0f})')
    axes[1,0].axis('off')

    axes[1,1].imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    axes[1,1].set_title(f'5. Detected Aggregates ({len(real_aggregates)} found)')
    axes[1,1].axis('off')

    # Histogram in last panel
    axes[1,2].hist(diff.ravel(), bins=256, range=(0,256), color='blue', alpha=0.7)
    axes[1,2].axvline(x=threshold_value, color='red', label=f'Otsu={threshold_value:.0f}')
    axes[1,2].set_title('6. Pixel Intensity Histogram')
    axes[1,2].set_xlabel('Brightness value')
    axes[1,2].set_ylabel('Pixel count')
    axes[1,2].legend()

    plt.suptitle('Plasma Protein Aggregation Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    print("\nAnalysis complete!")