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
    video_file = st.file_uploader("Video", type=["mp4", "avi", "mkv"])
    db_file = st.file_uploader("Database", type=["xlsx"])

with col2:
    if video_file:
        st.video(video_file)

# =========================
# PROCESS
# =========================
if st.button("🚀 Proses"):

    if not video_file or not db_file:
        st.error("❌ Harap upload Video dan Database terlebih dahulu.")
        st.stop()

    temp_dir = tempfile.mkdtemp()

    video_path = os.path.join(temp_dir, video_file.name)
    with open(video_path, "wb") as f:
        f.write(video_file.read())

    db_path = os.path.join(temp_dir, db_file.name)
    with open(db_path, "wb") as f:
        f.write(db_file.read())

    st.info("⚙️ Proses deteksi berjalan...")

    col_frame, col_log = st.columns([2, 1])

    with col_frame:
        st.markdown("**📸 Live Preview**")
        frame_placeholder = st.empty()

    with col_log:
        st.markdown("**📋 Log**")
        log_placeholder = st.empty()

    progress_bar = st.progress(0, text="Memproses video...")

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

    log_lines = []

    while True:
        line = process.stdout.readline()

        if not line:
            break

        line = line.strip()

        if not line:
            continue

        # =========================
        # PROGRESS BAR
        # =========================
        if "PROGRESS:" in line:
            try:
                p = int(line.split(":")[1].strip())
                p = max(0, min(p, 100))
                progress_bar.progress(p, text=f"Memproses video... {p}%")
            except:
                pass

        # =========================
        # LOG (semua baris termasuk error)
        # =========================
        else:
            log_lines.append(line)
            # Tampilkan 40 baris terakhir supaya tidak terlalu panjang
            log_placeholder.code("\n".join(log_lines[-40:]), language="bash")

        # =========================
        # LIVE FRAME PREVIEW
        # =========================
        frame_path = os.path.join("scripts", "output", "frame.jpg")
        if os.path.exists(frame_path):
            img = cv2.imread(frame_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(img, use_container_width=True)

    process.wait()

    # =========================
    # CEK STATUS AKHIR
    # =========================
    if process.returncode == 0:
        progress_bar.progress(100, text="✅ Selesai!")
        st.success("✅ Deteksi selesai!")
        st.session_state.processed = True
        st.session_state.csv_path = "scripts/output/hasil_bib.csv"
    else:
        progress_bar.progress(100, text="❌ Terjadi error")
        st.error("❌ Proses gagal. Lihat log di atas untuk detail error.")

# =========================
# RESULT
# =========================
if st.session_state.processed:

    csv_path = st.session_state.csv_path

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)

        st.subheader("📊 Hasil Deteksi")
        st.dataframe(df, use_container_width=True)

        with open(csv_path, "rb") as f:
            st.download_button(
                label="⬇️ Download CSV",
                data=f,
                file_name="hasil_bib.csv",
                mime="text/csv"
            )
    else:
        st.warning("⚠️ File hasil tidak ditemukan.")
