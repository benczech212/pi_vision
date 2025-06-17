from ultralytics import YOLO
import cv2
import os

# --- Box smoother using exponential moving average ---
class BoxSmoother:
    def __init__(self, alpha=0.4):
        self.prev_boxes = {}  # {id: (x1, y1, x2, y2)}
        self.alpha = alpha

    def smooth(self, box_id, new_box):
        if box_id not in self.prev_boxes:
            self.prev_boxes[box_id] = new_box
            return new_box

        prev = self.prev_boxes[box_id]
        smoothed = tuple([
            int(self.alpha * n + (1 - self.alpha) * p)
            for n, p in zip(new_box, prev)
        ])
        self.prev_boxes[box_id] = smoothed
        return smoothed

# --- Setup ---
input_path = "practice\\ai_boat_shot2.mp4"
output_path = "output\\ai_boat_smoothed.mp4"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    print("❌ Failed to open video")
    exit()

# Video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Output writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Load YOLO model
model = YOLO("yolov8n.pt")

# Initialize smoother
smoother = BoxSmoother(alpha=0.4)

print("🚀 Starting detection. Press 'q' to stop.")

frame_id = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("✅ Finished processing video")
        break

    # Run detection (filter to boats with: classes=[8])
    results = model(frame, conf=0.4)

    for i, box in enumerate(results[0].boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = model.names[cls_id]

        # Smooth box based on frame+index
        smooth_box = smoother.smooth(f"{frame_id}_{i}", (x1, y1, x2, y2))

        # Confidence-based color (green)
        color = (0, int(conf * 255), 0)
        thickness = max(1, int(conf * 6))

        # Draw box and label
        cv2.rectangle(frame, (smooth_box[0], smooth_box[1]), (smooth_box[2], smooth_box[3]), color, thickness)
        cv2.putText(frame, f"{label} {conf:.2f}", (smooth_box[0], smooth_box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Show and write frame
    cv2.imshow("Smoothed YOLOv8 Detection", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_id += 1

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
print("💾 Saved smoothed video to:", output_path)
