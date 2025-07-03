import cv2
import requests
from ultralytics import YOLO
import numpy as np

# Hardcoded URL
url = "http://127.0.0.1:5000/video_feed"
model = YOLO('E:\\dev\\pi_vision\\yolo_models\\best_todd_manatee_yolo11l.pt')  # Replace with path to your YOLOv11 weights

stream = requests.get(url, stream=True)
bytes_stream = b''

for chunk in stream.iter_content(chunk_size=1024):
    bytes_stream += chunk
    a = bytes_stream.find(b'\xff\xd8')  # JPEG start
    b = bytes_stream.find(b'\xff\xd9')  # JPEG end
    if a != -1 and b != -1:
        jpg = bytes_stream[a:b+2]
        bytes_stream = bytes_stream[b+2:]

        img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
        results = model(img, conf=0.5)
        annotated = results[0].plot()

        cv2.imshow("YOLOv11 Detection", annotated)
        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()
