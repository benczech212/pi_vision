from flask import Flask, Response, request
import cv2
import threading
import random
import os

app = Flask(__name__)
PORT = 5000

video_path = "E:\\Dev\\tekmara\\training media\\manatee footage\\aboveWater-001.mp4"
cap = None
lock = threading.Lock()
seek_time_sec = 0
should_seek = False
restart_stream = False


def get_video_length():
    temp_cap = cv2.VideoCapture(video_path)
    duration = temp_cap.get(cv2.CAP_PROP_FRAME_COUNT) / temp_cap.get(cv2.CAP_PROP_FPS)
    temp_cap.release()
    return duration


def generate_frames():
    global cap, seek_time_sec, should_seek, restart_stream

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Could not open video")
        return

    while True:
        with lock:
            if restart_stream:
                cap.release()
                cap = cv2.VideoCapture(video_path)
                restart_stream = False
            if should_seek:
                cap.set(cv2.CAP_PROP_POS_MSEC, seek_time_sec * 1000)
                should_seek = False

        success, frame = cap.read()
        if not success:
            break

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/skip_60_seconds')
def skip_60_seconds():
    global seek_time_sec, should_seek
    with lock:
        if cap:
            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            seek_time_sec = current_time + 60
            should_seek = True
            return f"⏩ Skipping to {seek_time_sec:.2f} seconds", 200
    return "❌ Video not open", 500


@app.route('/random_time')
def random_time():
    global seek_time_sec, should_seek
    with lock:
        duration = get_video_length()
        seek_time_sec = random.uniform(0, duration)
        should_seek = True
        return f"🎲 Jumping to random time: {seek_time_sec:.2f} seconds", 200


@app.route('/seek')
def seek_to_time():
    global seek_time_sec, should_seek
    try:
        t = float(request.args.get("t", "0"))
    except ValueError:
        return "❌ Invalid time format. Use /seek?t=45.5", 400

    with lock:
        duration = get_video_length()
        if 0 <= t <= duration:
            seek_time_sec = t
            should_seek = True
            return f"⏩ Seeking to {t:.2f} seconds", 200
        else:
            return f"❌ Time out of range (0–{duration:.2f})", 400


@app.route('/set_video')
def set_video():
    global video_path, restart_stream
    new_path = request.args.get("path")

    if not new_path or not os.path.isfile(new_path):
        return "❌ Invalid file path", 400

    with lock:
        video_path = new_path
        restart_stream = True
        return f"📂 Video source changed to: {video_path}", 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, threaded=True)
