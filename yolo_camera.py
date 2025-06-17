from ultralytics import YOLO
import cv2

# Load YOLOv8 nano model (smallest and fastest)
model = YOLO("yolov8n.pt")  # This will auto-download if not present

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()

print("✅ Running YOLOv8 object detection. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to grab frame")
        break

    # Inference
    results = model(frame)

    # Draw boxes
    annotated = results[0].plot()

    # Show the result
    cv2.imshow("YOLOv8 Object Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
