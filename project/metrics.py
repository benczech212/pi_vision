from prometheus_client import Counter, Gauge, start_http_server

start_http_server(8000)

detected_objects = Counter("detected_objects_total", "Detected object count", ['class_name'])
ocr_texts = Counter("ocr_text_total", "Total OCR text elements", ['language'])
frame_number = Gauge("frame_number", "Current video frame")
