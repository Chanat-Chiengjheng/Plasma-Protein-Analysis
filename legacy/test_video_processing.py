# ARCHIVED: kept for history. This is a live face-detection + area-vs-time demo unrelated to plasma protein analysis, likely a personal learning exercise mixed into this folder rather than part of the pipeline.
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# =============================================
# VIDEO PROCESSING LESSON
# Version 2.0 — Face Detection + Time Graph
# =============================================
# WHAT'S NEW IN V2:
#   - Record area data every frame into a list
#   - Record timestamp every frame
#   - After Q pressed → plot area vs time graph
# =============================================

# =====================
# SETTINGS
# =====================
CAMERA_INDEX  = 0
MIN_FACE_SIZE = 50

# =====================
# STEP 1: LOAD FACE DETECTOR
# =====================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
print("=" * 50)
print("VIDEO PROCESSING LESSON  v2.0")
print("=" * 50)
print("Face detector loaded")
print("Press Q to quit — graph will appear after")
print("=" * 50)

# =====================
# STEP 2: CONNECT TO CAMERA
# =====================
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("ERROR: Cannot open camera!")
    print("  Try changing CAMERA_INDEX to 1 or 2")
else:
    print("Camera connected successfully")

    # =====================
    # STEP 3: DATA STORAGE — NEW IN V2
    # =====================
    # These lists grow every frame — like a spreadsheet column
    time_log = []    # stores timestamp of each frame (seconds)
    area_log = []    # stores total detected area of each frame

    # Record the start time so we get relative time (0s, 1s, 2s...)
    start_time = time.time()

    # =====================
    # STEP 4: MAIN LOOP
    # =====================
    frame_count = 0

    while True:

        # STEP 4a: GRAB FRAME
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Cannot read frame")
            break

        frame_count += 1

        # STEP 4b: TIMESTAMP — NEW IN V2
        # How many seconds since we started recording
        current_time = time.time() - start_time

        # STEP 4c: GRAYSCALE
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # STEP 4d: DETECT FACES
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE)
        )

        # STEP 4e: DRAW RESULTS
        result_frame = frame.copy()
        num_detected = len(faces)

        # Calculate total area this frame
        total_area_this_frame = 0

        for i, (x, y, w, h) in enumerate(faces):
            area         = w * h
            center_x     = x + w // 2
            center_y     = y + h // 2
            aspect_ratio = w / h

            # Add this face's area to the frame total
            total_area_this_frame += area

            # Draw
            cv2.rectangle(result_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.circle(result_frame, (center_x, center_y), 5, (0, 0, 255), -1)
            label = f"#{i+1} | {area}px²"
            cv2.putText(
                result_frame, label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1
            )

        # STEP 4f: SAVE DATA THIS FRAME — NEW IN V2
        # Append current time and area to our lists
        time_log.append(current_time)
        area_log.append(total_area_this_frame)

        # STEP 4g: OVERLAY
        overlay_lines = [
            f"Time:     {current_time:.1f}s",
            f"Frame:    {frame_count}",
            f"Detected: {num_detected}",
            f"Area:     {total_area_this_frame}px²",
            f"Press Q to quit + plot"
        ]
        for j, line in enumerate(overlay_lines):
            cv2.putText(
                result_frame, line,
                (10, 25 + j * 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2
            )

        # STEP 4h: SHOW WINDOWS
        cv2.imshow('Live Detection (press Q to quit)', result_frame)
        cv2.imshow('Grayscale View', gray)

        # STEP 4i: QUIT CHECK
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print(f"\nStopped after {frame_count} frames ({current_time:.1f} seconds)")
            break

    # =====================
    # STEP 5: CLEANUP
    # =====================
    cap.release()
    cv2.destroyAllWindows()
    print("Camera released")

    # =====================
    # STEP 6: PLOT GRAPH — NEW IN V2
    # =====================
    print("\nGenerating graph...")

    if len(time_log) > 0:

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # --- Graph 1: Area vs Time ---
        axes[0].plot(time_log, area_log, color='blue', linewidth=1.5, label='Total detected area')
        axes[0].set_title('Detected Area Over Time', fontsize=13, fontweight='bold')
        axes[0].set_xlabel('Time (seconds)')
        axes[0].set_ylabel('Area (px²)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Highlight max area point
        max_area  = max(area_log)
        max_time  = time_log[area_log.index(max_area)]
        axes[0].axhline(y=max_area, color='red', linestyle='--', alpha=0.5, label=f'Max={max_area}px²')
        axes[0].annotate(
            f'Max: {max_area}px²',
            xy=(max_time, max_area),
            xytext=(max_time + 0.5, max_area * 0.95),
            fontsize=9, color='red'
        )

        # --- Graph 2: Smoothed area (rolling average) ---
        # Smoothing removes frame-to-frame jitter
        # Shows the real trend more clearly
        window = 15  # average over 15 frames
        smoothed = []
        for k in range(len(area_log)):
            start = max(0, k - window)
            smoothed.append(sum(area_log[start:k+1]) / (k - start + 1))

        axes[1].plot(time_log, area_log,  color='lightblue', linewidth=1,   alpha=0.6, label='Raw data')
        axes[1].plot(time_log, smoothed,  color='blue',      linewidth=2,   label=f'Smoothed (window={window} frames)')
        axes[1].set_title('Smoothed Area Over Time', fontsize=13, fontweight='bold')
        axes[1].set_xlabel('Time (seconds)')
        axes[1].set_ylabel('Area (px²)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.suptitle('Face Detection — Area vs Time Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

        # Print summary
        print("\n" + "=" * 50)
        print("SUMMARY")
        print("=" * 50)
        print(f"  Total recording time: {time_log[-1]:.1f} seconds")
        print(f"  Total frames:         {frame_count}")
        print(f"  Max area detected:    {max_area} px²")
        print(f"  Average area:         {sum(area_log)/len(area_log):.0f} px²")

    else:
        print("No data recorded")

    print("\nDone!")