import streamlit as st
import tempfile
import os
import pandas as pd
import subprocess
import cv2

st.set_page_config(page_title="Deteksi Bib", layout="wide")

if "processed" not in st.session_state:
    st.session_state.processed = False

if "csv_path" not in st.session_state:
    st.session_state.csv_path = None

st.title("🎯 Sistem Deteksi Bib")

# =========================
# MODE
# =========================
mode = st.radio("Mode", ["Manual", "Auto (Adaptive)"])

if mode == "Manual":
    confidence = st.slider("Confidence", 0.1, 1.0, 0.45, 0.05)
    fps = st.slider("Frame Skip", 1, 10, 1)
    auto_flag = "0"
else:
    confidence = 0.45
    fps = 1
    auto_flag = "1"

# =========================
# UPLOAD
# =========================
col1, col2 = st.columns(2)

with col1:
    video_file = st.file_uploader("Video", type=["mp4","avi","mkv"])
    db_file = st.file_uploader("Database", type=["xlsx"])

with col2:
    if video_file:
        st.video(video_file)

# =========================
# PROCESS
# =========================
if st.button("🚀 Proses"):

    temp_dir = tempfile.mkdtemp()

    video_path = os.path.join(temp_dir, video_file.name)
    with open(video_path, "wb") as f:
        f.write(video_file.read())

    db_path = os.path.join(temp_dir, db_file.name)
    with open(db_path, "wb") as f:
        f.write(db_file.read())

    frame_placeholder = st.empty()
    progress_bar = st.progress(0)

    process = subprocess.Popen(
        [
            "python",
            "scripts/main.py",
            video_path,
            db_path,
            str(confidence),
            str(fps),
            auto_flag
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="ignore"
    )

    while True:
        line = process.stdout.readline()

        if not line:
            break

        if "PROGRESS:" in line:
            try:
                p = int(line.split(":")[1])
                progress_bar.progress(p)
            except:
                pass

        frame_path = os.path.join("scripts","output","frame.jpg")

        if os.path.exists(frame_path):
            img = cv2.imread(frame_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(img)

    process.wait()

    st.session_state.processed = True
    st.session_state.csv_path = "scripts/output/hasil_bib.csv"

# =========================
# RESULT
# =========================
if st.session_state.processed:

    df = pd.read_csv(st.session_state.csv_path)

    st.dataframe(df)

    with open(st.session_state.csv_path, "rb") as f:
        st.download_button("Download CSV", f)