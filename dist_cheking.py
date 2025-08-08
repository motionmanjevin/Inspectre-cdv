import cv2
import time
import os

# Create output folder
output_folder = "movement_frames"
os.makedirs(output_folder, exist_ok=True)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Background subtractor for motion detection
fgbg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

last_capture_time = 0  # Track last capture time
capture_interval = 0.5   # seconds

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply background subtraction
    fgmask = fgbg.apply(frame)

    # Threshold the mask to remove shadows
    _, thresh = cv2.threshold(fgmask, 250, 255, cv2.THRESH_BINARY)

    # Count the number of white pixels (movement areas)
    movement_pixels = cv2.countNonZero(thresh)

    # If movement is detected (tweak threshold as needed)
    if movement_pixels > 5000:  # Adjust sensitivity here
        current_time = time.time()

        # Capture every 2 seconds while movement continues
        if current_time - last_capture_time >= capture_interval:
            frame_count += 1
            filename = os.path.join(output_folder, f"movement_{frame_count}.jpg")
            cv2.imwrite(filename, frame)
            print(f"[INFO] Saved: {filename}")
            last_capture_time = current_time

    # (Optional) Show detection
    cv2.imshow("Live Feed", frame)
    cv2.imshow("Motion Mask", thresh)

    # Quit on 'q'
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
