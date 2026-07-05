import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "used_plasma_pic")

# =====================
# LOAD AND PREPARE
# (everything from before)
# =====================
before = cv2.imread(os.path.join(DATA_DIR, "redM_blueO.JPG"))
after  = cv2.imread(os.path.join(DATA_DIR, "redM_blueO_Painted.png"))

if before is None or after is None:
    print("ERROR: Image not found!")
else:
    # Grayscale
    before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    after_gray  = cv2.cvtColor(after,  cv2.COLOR_BGR2GRAY)

    # Resize after to match before
    after_gray_resized = cv2.resize(
        after_gray,
        (before_gray.shape[1], before_gray.shape[0]),
        interpolation=cv2.INTER_LINEAR
    )

    # Background subtraction
    diff = cv2.absdiff(after_gray_resized, before_gray)

    # Otsu threshold
    threshold_value, binary = cv2.threshold(
        diff,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    print(f"Otsu threshold value: {threshold_value}")

    # =====================
    # STEP 1: FIND CONTOURS
    # =====================
    contours, hierarchy = cv2.findContours(
        binary,                 # input: binary image (0s and 255s)
        cv2.RETR_EXTERNAL,      # which contours to find
        cv2.CHAIN_APPROX_SIMPLE # how to store the points
    )

    print(f"Total contours found: {len(contours)}")

    # =====================
    # STEP 2: FILTER SMALL NOISE
    # =====================
    # Small contours are just noise (dust, lighting artifacts)
    # We only want real aggregates above a minimum size
    minimum_area = 100  # pixels — adjust based on your images

    real_aggregates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > minimum_area:
            real_aggregates.append(contour)

    print(f"Real aggregates (area > {minimum_area}): {len(real_aggregates)}")

    # =====================
    # STEP 3: MEASURE EACH AGGREGATE
    # =====================
    print("\n--- Measurements for each aggregate ---")

    for i, contour in enumerate(real_aggregates):

        # AREA
        area = cv2.contourArea(contour)

        # PERIMETER
        perimeter = cv2.arcLength(contour, True)
        # True = contour is closed (it loops back to start)

        # CIRCULARITY
        # Perfect circle = 1.0
        # Irregular blob = less than 1.0
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
        else:
            circularity = 0

        # BOUNDING BOX
        # Smallest rectangle containing the contour
        x, y, w, h = cv2.boundingRect(contour)
        # x, y = top-left corner position
        # w, h = width and height of rectangle

        # CENTROID (center point of the aggregate)
        M = cv2.moments(contour)
        if M['m00'] > 0:
            center_x = int(M['m10'] / M['m00'])
            center_y = int(M['m01'] / M['m00'])
        else:
            center_x, center_y = x, y

        print(f"\nAggregate {i+1}:")
        print(f"  Area:        {area:.1f} pixels")
        print(f"  Perimeter:   {perimeter:.1f} pixels")
        print(f"  Circularity: {circularity:.3f}  (1.0 = perfect circle)")
        print(f"  Position:    ({center_x}, {center_y})")
        print(f"  Bounding box: x={x}, y={y}, w={w}, h={h}")

    # =====================
    # STEP 4: DRAW CONTOURS ON IMAGE
    # =====================
    # Make a color copy of after image to draw on
    # (so we can draw colored lines on it)
    after_color = cv2.cvtColor(after_gray_resized, cv2.COLOR_GRAY2BGR)
    # GRAY2BGR = convert grayscale back to color format
    # (so we can draw colored annotations on it)

    # Draw all contours in GREEN
    cv2.drawContours(
        after_color,        # image to draw on
        real_aggregates,    # list of contours
        -1,                 # -1 = draw ALL contours
        (0, 255, 0),        # color in BGR: (0,255,0) = GREEN
        2                   # line thickness in pixels
    )

    # Draw center point for each aggregate in RED
    for contour in real_aggregates:
        M = cv2.moments(contour)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.circle(after_color, (cx, cy), 5, (0, 0, 255), -1)
            # circle(image, center, radius, color, thickness)
            # thickness = -1 means FILLED circle

    # =====================
    # STEP 5: DISPLAY RESULTS
    # =====================
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(diff, cmap='gray')
    axes[0].set_title('Difference Image')
    axes[0].axis('off')

    axes[1].imshow(binary, cmap='gray')
    axes[1].set_title(f'Binary (Otsu={threshold_value:.0f})')
    axes[1].axis('off')

    # Convert BGR to RGB for display
    axes[2].imshow(cv2.cvtColor(after_color, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f'Detected Aggregates ({len(real_aggregates)} found)')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()