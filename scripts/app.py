import streamlit as st
import tempfile
import os
import pandas as pd
import subprocess
import cv2
import requests
from datetime import datetime

st.set_page_config(page_title="Deteksi Bib", layout="wide")

# =========================
# DETEKSI ENVIRONMENT
# =========================
IS_SERVER = os.environ.get("IS_SERVER", "0") == "1"
FASTAPI_URL = "http://localhost:8000"

# =========================
# SESSION STATE
# =========================
if "logs" not in st.session_state:
    st.session_state.logs = []
if "detections" not in st.session_state:
    st.session_state.detections = []
if "processing" not in st.session_state:
    st.session_state.processing = False
if "done" not in st.session_state:
    st.session_state.done = False
if "csv_path" not in st.session_state:
    st.session_state.csv_path = None
if "video_server_path" not in st.session_state:
    st.session_state.video_server_path = None
if "video_uploaded" not in st.session_state:
    st.session_state.video_uploaded = False

def add_log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.append(f"[{ts}] {msg}")

def add_detection(bib: str, frame_num: str, total: str):
    ts = datetime.now().strftime("%H:%M:%S")
    no = len(st.session_state.detections) + 1
    st.session_state.detections.append(f"#{no:03d} [{ts}] Bib {bib} — frame {frame_num}/{total}")

# =========================
# KONTEN UTAMA
# =========================
st.title("🎯 Sistem Deteksi Bib")

# Badge environment
if IS_SERVER:
    st.info("☁️ Mode: **Server** — video akan diupload ke server", icon="☁️")
else:
    st.success("💻 Mode: **Lokal** — video diproses langsung tanpa upload", icon="💻")

mode = st.radio("Mode", ["Manual", "Auto (Adaptive)"])

if mode == "Manual":
    confidence = st.slider("Confidence", 0.1, 1.0, 0.45, 0.05)
    fps = st.slider("Frame Skip", 1, 10, 1)
    auto_flag = "0"
else:
    confidence = 0.45
    fps = 1
    auto_flag = "1"

col1, col2 = st.columns(2)

with col1:
    video_file = st.file_uploader("🎬 Video", type=["mp4", "avi", "mkv"])
    db_file = st.file_uploader("📂 Database", type=["xlsx"])

    if video_file:
        size_mb = round(video_file.size / 1024 / 1024, 2)
        st.info(f"📁 {video_file.name} ({size_mb} MB)")

        # Reset jika file baru
        if st.session_state.get("last_video_name") != video_file.name:
            st.session_state.video_uploaded = False
            st.session_state.video_server_path = None
            st.session_state["last_video_name"] = video_file.name

        # -------------------------
        # MODE LOKAL: simpan langsung
        # -------------------------
        if not IS_SERVER:
            if not st.session_state.video_uploaded:
                save_bar = st.progress(0, text="💾 Menyimpan video lokal...")
                try:
                    video_data = video_file.read()
                    total_size = len(video_data)
                    chunk_size = 1024 * 512  # 512KB
                    uploaded = 0

                    temp_dir = tempfile.mkdtemp()
                    local_path = os.path.join(temp_dir, video_file.name)

                    with open(local_path, "wb") as f:
                        for i in range(0, total_size, chunk_size):
                            chunk = video_data[i:i + chunk_size]
                            f.write(chunk)
                            uploaded += len(chunk)
                            pct = min(int(uploaded / total_size * 100), 100)
                            save_bar.progress(pct, text=f"💾 Menyimpan video... {pct}%")

                    st.session_state.video_server_path = local_path
                    st.session_state.video_uploaded = True
                    save_bar.progress(100, text="✅ Video siap diproses!")
                    add_log(f"Video disimpan lokal: {video_file.name} ({size_mb} MB)")
                except Exception as e:
                    save_bar.empty()
                    st.error(f"❌ Gagal menyimpan: {str(e)}")
            else:
                st.success("✅ Video siap diproses (lokal)")

        # -------------------------
        # MODE SERVER: upload ke FastAPI
        # -------------------------
        else:
            if not st.session_state.video_uploaded:
                if st.button("☁️ Upload Video ke Server"):
                    upload_bar = st.progress(0, text="⏫ Mengupload video...")
                    upload_status = st.empty()
                    try:
                        video_data = video_file.read()
                        total_size = len(video_data)
                        chunk_size = 1024 * 512
                        uploaded = 0
                        chunks = []

                        for i in range(0, total_size, chunk_size):
                            chunk = video_data[i:i + chunk_size]
                            chunks.append(chunk)
                            uploaded += len(chunk)
                            pct = min(int(uploaded / total_size * 90), 90)
                            upload_bar.progress(pct, text=f"⏫ Membaca video... {pct}%")

                        upload_bar.progress(92, text="⏫ Mengirim ke server...")

                        response = requests.post(
                            f"{FASTAPI_URL}/upload/video",
                            files={"file": (video_file.name, video_data, "video/mp4")},
                            timeout=600
                        )
                        result = response.json()

                        if result["status"] == "ok":
                            st.session_state.video_server_path = result["path"]
                            st.session_state.video_uploaded = True
                            upload_bar.progress(100, text="✅ Upload selesai!")
                            upload_status.success(f"✅ {result['filename']} ({size_mb} MB) berhasil diupload!")
                            add_log(f"Video diupload ke server: {result['filename']} ({size_mb} MB)")
                        else:
                            upload_bar.empty()
                            upload_status.error(f"❌ Gagal: {result.get('message', 'Unknown error')}")
                    except requests.exceptions.ConnectionError:
                        upload_bar.empty()
                        st.error("❌ Tidak bisa terhubung ke FastAPI server.")
                    except Exception as e:
                        upload_bar.empty()
                        st.error(f"❌ Error: {str(e)}")
            else:
                st.success("✅ Video sudah diupload ke server — siap diproses!")

    if db_file:
        add_log(f"Database dipilih: {db_file.name}")

with col2:
    if video_file:
        st.video(video_file)

# =========================
# TOMBOL PROSES
# =========================
if st.button("🚀 Proses", disabled=st.session_state.processing):

    if not video_file or not db_file:
        st.error("❌ Harap upload Video dan Database terlebih dahulu.")
        st.stop()

    if not st.session_state.video_uploaded or not st.session_state.video_server_path:
        if IS_SERVER:
            st.error("❌ Harap klik 'Upload Video ke Server' terlebih dahulu.")
        else:
            st.error("❌ Video belum siap, coba pilih ulang file video.")
        st.stop()

    st.session_state.done = False
    st.session_state.csv_path = None
    st.session_state.detections = []
    st.session_state.processing = True

    add_log("Menyimpan file sementara...")

    temp_dir = tempfile.mkdtemp()
    video_path = st.session_state.video_server_path

    db_path = os.path.join(temp_dir, db_file.name)
    with open(db_path, "wb") as f:
        f.write(db_file.read())
    add_log(f"Database disimpan: {db_file.name}")
    add_log(f"Mulai proses — CONF:{confidence} SKIP:{fps} AUTO:{auto_flag}")

    col_preview, col_det = st.columns([2, 1])
    with col_preview:
        st.markdown("**📸 Live Preview**")
        frame_placeholder = st.empty()
    with col_det:
        det_header = st.empty()
        det_box = st.empty()

    progress_bar = st.progress(0, text="Memproses video...")

    with st.expander("📋 Log", expanded=False):
        log_box = st.empty()

    def render_panels():
        total_det = len(st.session_state.detections)
        det_header.markdown(f"**🎯 Bib Terdeteksi ({total_det})**")
        det_box.code(
            "\n".join(st.session_state.detections[-40:]) if st.session_state.detections else "— belum ada deteksi —",
            language="bash"
        )
        log_box.code(
            "\n".join(st.session_state.logs[-30:]) if st.session_state.logs else "— belum ada aktivitas —",
            language="bash"
        )

    render_panels()

    try:
        process = subprocess.Popen(
            ["python", "scripts/main.py", video_path, db_path, str(confidence), str(fps), auto_flag],
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
            line = line.strip()
            if not line:
                continue

            if line.startswith("PROGRESS:"):
                try:
                    p = max(0, min(int(line.split(":")[1].strip()), 100))
                    progress_bar.progress(p, text=f"Memproses video... {p}%")
                except:
                    pass
            elif line.startswith("DETECTED:"):
                try:
                    parts = line.split(":")
                    bib = parts[1].strip()
                    frame_num = parts[3].strip() if len(parts) >= 4 else "?"
                    total = parts[5].strip() if len(parts) >= 6 else "?"
                    add_detection(bib, frame_num, total)
                except:
                    add_log(f"DETECTED (parse error): {line}")
                render_panels()
            else:
                add_log(line)
                render_panels()

            frame_path = os.path.join("scripts", "output", "frame.jpg")
            if os.path.exists(frame_path):
                img = cv2.imread(frame_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(img, use_container_width=True)

        process.wait()

        if process.returncode == 0:
            progress_bar.progress(100, text="✅ Selesai!")
            add_log("✅ Proses selesai!")
            st.session_state.done = True
            st.session_state.csv_path = os.path.join("scripts", "output", "hasil_bib.csv")

            # Hapus file server hanya jika mode server
            if IS_SERVER:
                try:
                    requests.delete(f"{FASTAPI_URL}/upload/video", json={"path": video_path}, timeout=10)
                    add_log("File video temp dihapus dari server.")
                except:
                    pass
        else:
            progress_bar.progress(100, text="❌ Terjadi error")
            add_log(f"❌ Proses gagal. Return code: {process.returncode}")

    except Exception as e:
        add_log(f"❌ Exception: {str(e)}")
    finally:
        st.session_state.processing = False
        render_panels()

    st.rerun()

# =========================
# RESULT
# =========================
if st.session_state.done:
    csv_path = st.session_state.csv_path or os.path.join("scripts", "output", "hasil_bib.csv")
    col_result, col_det = st.columns([2, 1])

    with col_result:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            st.subheader(f"📊 Hasil Deteksi ({len(df)} runner)")
            st.dataframe(df, use_container_width=True)
            with open(csv_path, "rb") as f:
                st.download_button("⬇️ Download CSV", data=f, file_name="hasil_bib.csv", mime="text/csv")
        else:
            st.warning("⚠️ File hasil tidak ditemukan.")

    with col_det:
        total_det = len(st.session_state.detections)
        st.markdown(f"**🎯 Bib Terdeteksi ({total_det})**")
        st.code(
            "\n".join(st.session_state.detections) if st.session_state.detections else "— tidak ada deteksi —",
            language="bash"
        )

# =========================
# LOG
# =========================
if not st.session_state.processing:
    with st.expander("📋 Log", expanded=False):
        st.code(
            "\n".join(st.session_state.logs[-30:]) if st.session_state.logs else "— belum ada aktivitas —",
            language="bash"
        )