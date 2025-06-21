from ultralytics import YOLO
import cv2
import os

# Load the model
model = YOLO("yolov8n.pt")

# Open video file
input_folder = "practice"
input_filename = "youtube_clip_1 - 1080p.mp4"
input_path = os.path.join(input_folder, input_filename)
cap = cv2.VideoCapture(input_path)

if not cap.isOpened():
    print("❌ Failed to open video")
    exit()

# Get video info
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Set up VideoWriter
# use input filename for output
output_folder = "output"
output_filename = input_filename.replace(".mp4", "_detected.mp4")
output_path = os.path.join(output_folder, output_filename)
os.makedirs(os.path.dirname(output_path), exist_ok=True)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # or use "XVID" for .avi
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

print("🎥 Saving annotated video to:", output_path)

while True:
    ret, frame = cap.read()
    if not ret:
        print("✅ Finished processing video")
        break

    # Detect objects
    results = model(frame, conf=0.5)

    # Annotate frame
    annotated = results[0].plot()

    # Show frame
    cv2.imshow("Video Detection", annotated)

    # Write annotated frame to output video
    out.write(annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Clean up
cap.release()
out.release()
cv2.destroyAllWindows()
