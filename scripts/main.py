import sys
import os

# 🔥 FIX: flush otomatis setiap baris tanpa perlu flush=True satu-satu
sys.stdout.reconfigure(line_buffering=True)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import pandas as pd

from modules.detector import Detector
from modules.preprocess import preprocess_image
from modules.ocr import read_text
from modules.utils import add_padding
from modules.database import RunnerDatabase
from modules.time_ocr import TimeOCR
from scripts.finish_module import FinishTimeSystem

# =========================
# DEFAULT
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "best_final.pt")

CONFIDENCE = 0.45
FRAME_SKIP = 1
AUTO_MODE = False

RESIZE_WIDTH = 960

# =========================
# ARGUMENT
# =========================
if len(sys.argv) >= 4:
    try:
        CONFIDENCE = float(sys.argv[3])
    except:
        pass

if len(sys.argv) >= 5:
    try:
        FRAME_SKIP = int(sys.argv[4])
    except:
        pass

if len(sys.argv) >= 6:
    AUTO_MODE = sys.argv[5] == "1"

# =========================
# INPUT
# =========================
if len(sys.argv) >= 3:
    video_path = sys.argv[1]
    DB_PATH = sys.argv[2]
else:
    print("ERROR: Jalankan melalui Streamlit app.")
    exit()

print("VIDEO:", video_path)

# =========================
# OUTPUT
# =========================
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FRAME_PATH = os.path.join(OUTPUT_DIR, "frame.jpg")

# =========================
# DATABASE
# =========================
print("Loading database...")
database = RunnerDatabase(DB_PATH)
print("Database loaded.")

# =========================
# INIT VIDEO
# =========================
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ERROR VIDEO: Tidak bisa membuka file video.")
    exit(1)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
if total_frames == 0:
    total_frames = 1

print("CONF:", CONFIDENCE)
print("FRAME_SKIP:", FRAME_SKIP)
print("AUTO_MODE:", AUTO_MODE)
print("TOTAL_FRAMES:", total_frames)

# =========================
# INIT MODEL
# =========================
print("Loading model...")
detector = Detector(MODEL_PATH, CONFIDENCE)
print("Model loaded.")

time_ocr = TimeOCR()
finish_system = FinishTimeSystem()

detected_numbers = set()
frame_count = 0

# =========================
# LOOP
# =========================
print("Mulai deteksi...")

while True:
    ret, frame = cap.read()

    if not ret or frame is None:
        break

    frame_count += 1

    # RESIZE FRAME
    h, w = frame.shape[:2]
    scale = RESIZE_WIDTH / w
    frame = cv2.resize(frame, (RESIZE_WIDTH, int(h * scale)))

    # FPS CONTROL (MANUAL)
    if not AUTO_MODE:
        if frame_count % FRAME_SKIP != 0:
            continue

    # DETECTION
    boxes = detector.detect(frame)

    # ADAPTIVE
    if AUTO_MODE:
        num_boxes = len(boxes)
        if num_boxes > 5:
            dynamic_skip = 1
        elif num_boxes > 2:
            dynamic_skip = 2
        else:
            dynamic_skip = 1

        if frame_count % dynamic_skip != 0:
            continue

    # PROGRESS
    progress = int((frame_count / total_frames) * 100)
    print(f"PROGRESS:{progress}")

    current_time = time_ocr.read_time(frame)

    for (x1, y1, x2, y2) in boxes:

        crop = frame[y1:y2, x1:x2]

        if crop is None or crop.size == 0:
            continue

        crop = cv2.resize(crop, None, fx=2, fy=2)
        processed = preprocess_image(crop)

        if frame_count % 7 == 0:
            text = read_text(processed)
        else:
            text = ""

        if text:
            matched_bib, runner = database.get_runner(text)

            if runner:
                detected_numbers.add(matched_bib)
                # 🔥 kirim bib + nomor frame asli
                print(f"DETECTED:{matched_bib}:FRAME:{frame_count}:TOTAL:{total_frames}")

                if current_time:
                    finish_system.process(
                        matched_bib,
                        text,
                        x1, y1, x2, y2,
                        1.0,
                        frame,
                        current_time
                    )

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imwrite(FRAME_PATH, frame)

cap.release()
print("Video selesai diproses.")

# =========================
# EXPORT
# =========================
print("Menyimpan hasil...")

results = finish_system.get_results()

data = []

for bib in detected_numbers:
    runner = database.db.get(bib, {})
    time_str = results[bib]["time"] if bib in results else "-"

    data.append({
        "Bib": bib,
        "Nama": runner.get("nama", ""),
        "Kategori": runner.get("kategori", ""),
        "Waktu Finish": time_str
    })

df = pd.DataFrame(data)
csv_out = os.path.join(OUTPUT_DIR, "hasil_bib.csv")
df.to_csv(csv_out, index=False)

print(f"Tersimpan: {len(data)} runner terdeteksi.")
print("DONE")