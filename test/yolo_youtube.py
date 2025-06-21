from ultralytics import YOLO
import cv2
import os
import yt_dlp

def download_youtube_video(youtube_url, output_dir="downloads"):
    print("📥 Downloading video from YouTube...")
    os.makedirs(output_dir, exist_ok=True)
    ydl_opts = {
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
        'quiet': True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        filename = ydl.prepare_filename(info)
        if not filename.endswith('.mp4'):
            filename = filename.replace('.webm', '.mp4')  # fallback
        return filename

# --- SET YOUR YOUTUBE LINK HERE ---
youtube_url = "https://www.youtube.com/watch?v=SGKM8rfnC2Y"

# Step 1: Download the video
input_path = download_youtube_video(youtube_url)

# Step 2: Load YOLO and process the video
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(input_path)

if not cap.isOpened():
    print("❌ Failed to open video")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

output_path = "output/ai_youtube_detected.mp4"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

print("🎥 Saving annotated video to:", output_path)

while True:
    ret, frame = cap.read()
    if not ret:
        print("✅ Finished processing video")
        break

    results = model(frame, conf=0.5)
    annotated = results[0].plot()
    out.write(annotated)
    cv2.imshow("Video Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
