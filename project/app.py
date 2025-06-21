import streamlit as st
import cv2
import tempfile
from detector import process_frame

st.set_page_config(page_title="YOLO + OCR Monitoring", layout="wide")
st.title("📹 YOLOv8 + OCR Monitoring Dashboard")

# Initialize session state
if 'stop' not in st.session_state:
    st.session_state.stop = False

# File uploader
uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

# Only show the Stop button once a video is loaded
if uploaded_video and not st.session_state.stop:
    if st.button("🛑 Stop Processing", key="stop_button"):
        st.session_state.stop = True

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())

    cap = cv2.VideoCapture(tfile.name)
    frame_id = 0
    frame_window = st.image([])
    ocr_box = st.empty()

    st.info("Processing video. Click 'Stop Processing' to interrupt.")

    while cap.isOpened() and not st.session_state.stop:
        ret, frame = cap.read()
        if not ret:
            st.success("✅ Finished processing the video.")
            break

        frame_id += 1
        annotated_frame, ocr_results = process_frame(frame, frame_id)

        # Convert BGR to RGB for display
        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        frame_window.image(frame_rgb, caption=f"Frame {frame_id}", use_column_width=True)

        # Display OCR text
        ocr_texts = [text for _, text, _ in ocr_results]
        if ocr_texts:
            ocr_box.markdown("**📝 OCR Text Detected:**<br>" + "<br>".join(ocr_texts), unsafe_allow_html=True)
        else:
            ocr_box.empty()

    cap.release()

# Reset functionality
if st.session_state.stop:
    if st.button("🔄 Reset App"):
        st.session_state.stop = False
        st.experimental_rerun()
