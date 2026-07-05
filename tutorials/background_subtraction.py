import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "used_plasma_pic")

# =====================
# LOAD BOTH IMAGES
# =====================
before = cv2.imread(os.path.join(DATA_DIR, "redM_blueO.JPG"))
after  = cv2.imread(os.path.join(DATA_DIR, "redM_blueO_Painted.png"))

# Safety check for both
if before is None or after is None:
    print("ERROR: One or both images not found!")
else:
    print("Both images loaded successfully")
    print("Before shape:", before.shape)
    print("After shape: ", after.shape)

    # =====================
    # CONVERT BOTH TO GRAYSCALE
    # =====================
    before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    after_gray  = cv2.cvtColor(after,  cv2.COLOR_BGR2GRAY)

    # After loading both images and converting to grayscale
    # ADD THIS before absdiff:

    # Check sizes
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
    # BACKGROUND SUBTRACTION
    # =====================
    # absdiff = absolute difference
    # safe subtraction — no negative numbers

    print("Diff shape:", diff.shape)
    print("Diff max value:", diff.max())   # how bright is brightest change?
    print("Diff min value:", diff.min())   # should be 0 (no change areas)

    # =====================
    # DISPLAY ALL THREE
    # =====================
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    # 1 row, 3 columns of images side by side

    axes[0].imshow(before_gray, cmap='gray')
    axes[0].set_title('Before (Background)')
    axes[0].axis('off')

    axes[1].imshow(after_gray, cmap='gray')
    axes[1].set_title('After (Treatment)')
    axes[1].axis('off')

    axes[2].imshow(diff, cmap='gray')
    axes[2].set_title('Difference (Aggregate only)')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

    # =====================
    # LOOK AT THE NUMBERS
    # =====================
    print("\nBefore - center region (200:205, 300:305):")
    print(before_gray[200:205, 300:305])

    print("\nAfter - same region:")
    print(after_gray[200:205, 300:305])

    print("\nDifference - same region:")
    print(diff[200:205, 300:305])