import cv2
from ultralytics import YOLO
import easyocr
from metrics import detected_objects, ocr_texts, frame_number

model = YOLO("yolov8n.pt")
reader = easyocr.Reader(['en', 'ja'], gpu=True)

def process_frame(frame, frame_id):
    frame_number.set(frame_id)

    results = model(frame, conf=0.5)
    boxes = results[0].boxes
    annotated = results[0].plot()

    # YOLO metrics
    for cls_id in boxes.cls.tolist():
        cls_name = model.names[int(cls_id)]
        detected_objects.labels(class_name=cls_name).inc()

    # OCR
    ocr_results = reader.readtext(frame)
    for _, text, _ in ocr_results:
        ocr_texts.labels(language="en").inc()  # Simplified for now

    return annotated, ocr_results
