# =====================
# IMPORTS
# =====================
import os

import cv2          # OpenCV — image processing
import numpy as np  # NumPy — math on arrays
import matplotlib.pyplot as plt  # for displaying images

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "used_plasma_pic")

# =====================
# STEP 1: LOAD IMAGE
# =====================
image = cv2.imread(os.path.join(DATA_DIR, "redM_blueO.JPG"))

# Safety check
if image is None:
    print("ERROR: Image not found! Check your path")
else:
    print("SUCCESS: Image loaded!")
    print("Shape:", image.shape)
    print("Data type:", image.dtype)
    print("Max value:", image.max())
    print("Min value:", image.min())

    # =====================
    # STEP 2: DISPLAY ORIGINAL
    # =====================
    # IMPORTANT: OpenCV loads BGR, matplotlib wants RGB
    # Must flip channels before displaying
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(8, 6))
    plt.imshow(image_rgb)
    plt.title('Original Image (Color)')
    plt.axis('off')
    plt.show()

    # =====================
    # STEP 3: CONVERT TO GRAYSCALE
    # =====================
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    print("Grayscale shape:", gray.shape)
    # Notice: no more 3 at the end — one number per pixel now

    # =====================
    # STEP 4: DISPLAY GRAYSCALE
    # =====================
    plt.figure(figsize=(8, 6))
    plt.imshow(gray, cmap='gray')
    plt.title('Grayscale Image')
    plt.colorbar()  # shows the 0-255 scale on the side
    plt.axis('off')
    plt.show()

    # =====================
    # STEP 5: LOOK AT ACTUAL NUMBERS
    # =====================
    # This is important — see the real pixel values
    print("Top-left 5x5 corner of image:")
    print(gray[0:5, 0:5])
    # This shows you the actual numbers Python sees
    # rows first, columns second → gray[row, column]