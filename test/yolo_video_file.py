from ultralytics import YOLO
import cv2
import os
import random
import datetime

# Load the model
model = YOLO("E:\\dev\\pi_vision\\yolo_models\\yolo11n-benc_xview_25-06-30.pt")

# Open video file
input_path = "E:\\dev\\pi_vision\\test\\practice\\ai_boat_shot1.mp4"
cap = cv2.VideoCapture(input_path)

if not cap.isOpened():
    print("❌ Failed to open video")
    exit()

# Get video info
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration_sec = frame_count / fps
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Output setup
output_folder = "output"
input_filename = os.path.basename(input_path)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = input_filename.replace(".mp4", f"_detected_{timestamp}.mp4")
output_path = os.path.join(output_folder, output_filename)
os.makedirs(os.path.dirname(output_path), exist_ok=True)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

print("🎥 Saving annotated video to:", output_path)
print("🎮 Controls: [r] random time | [1] +1 min | [5] +5 min | [q] quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("✅ Finished processing video")
        break

    # Run detection
    results = model(frame, conf=0.5)
    annotated = results[0].plot()
    out.write(annotated)
    cv2.imshow("Video Detection", annotated)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("r"):
        # Jump to random time
        random_time = random.uniform(0, duration_sec)
        cap.set(cv2.CAP_PROP_POS_MSEC, random_time * 1000)
        print(f"🔀 Jumped to {random_time:.2f}s")
    elif key == ord("1"):
        # Skip 1 minute
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        new_time = min(current_time + 60, duration_sec)
        cap.set(cv2.CAP_PROP_POS_MSEC, new_time * 1000)
        print(f"⏩ Skipped to {new_time:.2f}s (+1 min)")
    elif key == ord("5"):
        # Skip 5 minutes
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        new_time = min(current_time + 300, duration_sec)
        cap.set(cv2.CAP_PROP_POS_MSEC, new_time * 1000)
        print(f"⏩⏩ Skipped to {new_time:.2f}s (+5 min)")

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
