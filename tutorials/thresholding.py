import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "used_plasma_pic")

# =====================
# LOAD AND PREPARE
# (same as before)
# =====================
before = cv2.imread(os.path.join(DATA_DIR, "redM_blueO.JPG"))
after  = cv2.imread(os.path.join(DATA_DIR, "redM_blueO_Painted.png"))

if before is None or after is None:
    print("ERROR: Image not found!")
else:
    before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    after_gray  = cv2.cvtColor(after,  cv2.COLOR_BGR2GRAY)

    print("Before shape:", before_gray.shape)
    print("After shape: ", after_gray.shape)

    # Resize 'after' to match 'before' size
    # before_gray.shape[1] = width of before
    # before_gray.shape[0] = height of before
    after_gray_resized = cv2.resize(
        after_gray,                              # image to resize
        (before_gray.shape[1], before_gray.shape[0]),  # target (width, height)
        interpolation=cv2.INTER_LINEAR           # method of resizing
    )

    print("After resized shape:", after_gray_resized.shape)
    # Should now match before_gray shape

    # NOW do absdiff with resized image
    diff = cv2.absdiff(after_gray_resized, before_gray)

    # =====================
    # STEP 1: MANUAL THRESHOLD
    # =====================
    # You pick the value (try 30, 50, 80 and see difference)
    manual_threshold_value = 50

    # cv2.threshold returns TWO things:
    # ret     = the threshold value actually used
    # manual  = the resulting binary image
    ret, manual = cv2.threshold(
        diff,                    # input image
        manual_threshold_value,  # threshold value
        255,                     # value to set if ABOVE threshold
        cv2.THRESH_BINARY        # type: simple above/below
    )
    print(f"Manual threshold used: {ret}")

    # =====================
    # STEP 2: OTSU THRESHOLD
    # (automatic - finds best value itself)
    # =====================
    otsu_ret, otsu = cv2.threshold(
        diff,
        0,                                    # ignored when using Otsu
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU  # Otsu flag added
    )
    print(f"Otsu automatically chose threshold: {otsu_ret}")

    # =====================
    # STEP 3: PLOT HISTOGRAM
    # (so you can SEE why Otsu picked that value)
    # =====================
    plt.figure(figsize=(8, 4))
    plt.hist(diff.ravel(), bins=256, range=(0, 256), color='blue', alpha=0.7)
    # diff.ravel() = flatten 2D array into 1D list of all pixel values
    # bins=256     = one bar per brightness value
    plt.axvline(x=manual_threshold_value, color='red',   
                label=f'Manual ({manual_threshold_value})')
    plt.axvline(x=otsu_ret, color='green', 
                label=f'Otsu ({otsu_ret:.0f})')
    plt.xlabel('Pixel brightness value (0-255)')
    plt.ylabel('Number of pixels')
    plt.title('Histogram of Difference Image')
    plt.legend()
    plt.show()

    # =====================
    # STEP 4: DISPLAY ALL RESULTS
    # =====================
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(diff, cmap='gray')
    axes[0].set_title('Difference Image (grayscale)')
    axes[0].axis('off')

    axes[1].imshow(manual, cmap='gray')
    axes[1].set_title(f'Manual Threshold ({manual_threshold_value})')
    axes[1].axis('off')

    axes[2].imshow(otsu, cmap='gray')
    axes[2].set_title(f'Otsu Threshold (auto={otsu_ret:.0f})')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

    # =====================
    # STEP 5: COUNT WHITE PIXELS
    # (first real measurement!)
    # =====================
    manual_aggregate_pixels = np.sum(manual == 255)
    otsu_aggregate_pixels   = np.sum(otsu == 255)

    print(f"Manual threshold -> aggregate pixels: {manual_aggregate_pixels}")
    print(f"Otsu threshold   -> aggregate pixels: {otsu_aggregate_pixels}")

    total_pixels = diff.shape[0] * diff.shape[1]
    print(f"Total image pixels: {total_pixels}")
    print(f"Aggregate fraction (Otsu): "
          f"{otsu_aggregate_pixels/total_pixels*100:.2f}%")