from ultralytics import YOLO
import easyocr
import cv2
import os

# === Configuration ===
input_folder = "test\\practice"
input_filename = "E:\\dev\\pi_vision\\test\\practice\\youtube_clip_1 - 1080p.mp4"
output_folder = "output"
model_path = "E:\\dev\\pi_vision\\yolov8n.pt"
ocr_languages = ['en', 'ja']  # Customize as needed

# === Prepare paths ===
input_path = os.path.join(input_folder, input_filename)
output_filename = input_filename.replace(".mp4", "_detected.mp4")
output_path = os.path.join(output_folder, output_filename)
os.makedirs(output_folder, exist_ok=True)

# === Load Models ===
print("📦 Loading models...")
model = YOLO(model_path)
reader = easyocr.Reader(ocr_languages, gpu=True)  # Use gpu=False if no GPU

# === Open Video ===
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    print("❌ Failed to open video:", input_path)
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

print(f"🎥 Processing '{input_filename}'")
print(f"💾 Saving annotated output to: {output_path}")

# === Frame Loop ===
while True:
    ret, frame = cap.read()
    if not ret:
        print("✅ Finished processing video.")
        break

    # YOLO Detection
    results = model(frame, conf=0.5)
    annotated = results[0].plot()

    # OCR Detection
    ocr_results = reader.readtext(frame)
    for (bbox, text, conf) in ocr_results:
        if conf > 0.5:
            top_left = tuple(map(int, bbox[0]))
            bottom_right = tuple(map(int, bbox[2]))
            cv2.rectangle(annotated, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(annotated, text, (top_left[0], top_left[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("Object + OCR Detection", annotated)
    out.write(annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("🛑 Interrupted by user.")
        break

# === Clean Up ===
cap.release()
out.release()
cv2.destroyAllWindows()
