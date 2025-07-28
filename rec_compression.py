import cv2
import subprocess
import os

# === CONFIG ===
output_raw = "output_raw.avi"
output_compressed = "output_compressed.mp4"
frame_width = 640
frame_height = 480
fps = 20.0

# === RECORD FROM WEBCAM ===
cap = cv2.VideoCapture(0)

# Set resolution
cap.set(3, frame_width)
cap.set(4, frame_height)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_raw, fourcc, fps, (frame_width, frame_height))

print("Recording... Press 'q' to stop.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    out.write(frame)
    cv2.imshow("Recording", frame)

    # Stop recording when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("Recording stopped.")

# === COMPRESS USING FFMPEG ===
print("Compressing video...")
subprocess.run([
    "ffmpeg", "-i", output_raw,
    "-vcodec", "libx264", "-crf", "28",  # CRF 28 gives decent compression
    output_compressed
])

print(f"Compression done. Saved as {output_compressed}")

# === CLEAN UP RAW FILE ===
if os.path.exists(output_raw):
    os.remove(output_raw)
