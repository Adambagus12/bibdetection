import sys
import os
import time

sys.stdout.reconfigure(line_buffering=True)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import pandas as pd

from modules.detector import Detector
from modules.preprocess import preprocess_image
from modules.ocr import read_text
from modules.ocr_tesseract import read_text_tesseract
from modules.database import RunnerDatabase
from modules.time_ocr import TimeOCR
from scripts.finish_module import FinishTimeSystem       # mode "zone"
from modules.grid_tracking import GridTrackingSystem      # mode "grid"
from modules.rotation_correction import correct_rotation
from modules.sharpen import sharpen_image

# =========================
# FUNGSI CER
# =========================
def hitung_cer(teks_asli, teks_ocr):
    if not teks_asli:
        return 0.0

    panjang_asli = len(teks_asli)
    panjang_ocr = len(teks_ocr)

    dp = [[0 for _ in range(panjang_ocr + 1)] for _ in range(panjang_asli + 1)]

    for i in range(panjang_asli + 1):
        dp[i][0] = i
    for j in range(panjang_ocr + 1):
        dp[0][j] = j

    for i in range(1, panjang_asli + 1):
        for j in range(1, panjang_ocr + 1):
            cost = 0 if teks_asli[i - 1] == teks_ocr[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )

    return dp[panjang_asli][panjang_ocr] / panjang_asli


# =========================
# DEFAULT
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "best_v8n.pt")
CONFIDENCE = 0.45
FRAME_SKIP = 1
AUTO_MODE = False
RESIZE_WIDTH = 960
MIN_FRAME_VALID = 2  # syarat multi-frame validation

# GANTI KE "tesseract" UNTUK MENGUJI TESSERACT, DEFAULT "easyocr"
OCR_ENGINE = "easyocr"
INTERVAL_OCR = 3

# =========================
# INPUT
# argv: [1]=video [2]=db [3]=confidence [4]=frame_skip
#       [5]=auto_mode [6]=tracking_mode ("zone" / "grid")
#       [7]=grid_size (opsional, hanya dipakai kalau mode grid)
#       [8]=ocr_engine ("easyocr" / "tesseract", opsional)
# =========================
if len(sys.argv) >= 3:
    video_path = sys.argv[1]
    DB_PATH = sys.argv[2]
else:
    print("ERROR: Jalankan melalui Streamlit app / argumen tidak lengkap.")
    exit()

print("VIDEO:", video_path)

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

TRACKING_MODE = "zone"
if len(sys.argv) >= 7:
    if sys.argv[6] in ("zone", "grid"):
        TRACKING_MODE = sys.argv[6]

GRID_SIZE = 120
if len(sys.argv) >= 8:
    try:
        GRID_SIZE = int(sys.argv[7])
    except:
        pass

if len(sys.argv) >= 9:
    if sys.argv[8] in ("easyocr", "tesseract"):
        OCR_ENGINE = sys.argv[8]

# argv[9] = path model, argv[10] = interval OCR, argv[11] = min frame valid
if len(sys.argv) >= 10 and sys.argv[9]:
    MODEL_PATH = sys.argv[9]

if len(sys.argv) >= 11:
    try:
        INTERVAL_OCR = int(sys.argv[10])
    except:
        pass

if len(sys.argv) >= 12:
    try:
        MIN_FRAME_VALID = int(sys.argv[11])
    except:
        pass

# argv[12] = koreksi rotasi (0/1), argv[13] = kekuatan sharpening
PAKAI_ROTASI = False
SHARPEN_AMOUNT = 0.0

if len(sys.argv) >= 13:
    PAKAI_ROTASI = sys.argv[12] == "1"

if len(sys.argv) >= 14:
    try:
        SHARPEN_AMOUNT = float(sys.argv[13])
    except:
        pass

print("ROTASI:", PAKAI_ROTASI)
print("SHARPEN:", SHARPEN_AMOUNT)

print("MODEL:", MODEL_PATH)
print("INTERVAL_OCR:", INTERVAL_OCR)
print("MIN_FRAME_VALID:", MIN_FRAME_VALID)

print("TRACKING_MODE:", TRACKING_MODE)
if TRACKING_MODE == "grid":
    print("GRID_SIZE:", GRID_SIZE)
print("OCR_ENGINE:", OCR_ENGINE)

# =========================
# OUTPUT
# =========================
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
FRAME_PATH = os.path.join(OUTPUT_DIR, "frame.jpg")

# --- BARU: folder khusus untuk gambar contoh Bab 4 (skripsi) ---
GAMBAR_DIR = os.path.join(OUTPUT_DIR, "gambar_skripsi")
os.makedirs(GAMBAR_DIR, exist_ok=True)

# Nama file video dipakai sebagai penanda (video1/video2) pada nama gambar
# akuisisi, supaya tidak tertimpa saat menjalankan video kedua.
video_tag = os.path.splitext(os.path.basename(video_path))[0]

# Flag/counter untuk memastikan tiap jenis contoh gambar cuma disimpan
# sekali (akuisisi, preprocessing) atau maksimal beberapa kali (deteksi),
# supaya tidak menumpuk ratusan file setiap kali sistem dijalankan.
akuisisi_tersimpan = False
preprocessing_tersimpan = False
deteksi_tersimpan = 0
MAKS_CONTOH_DETEKSI = 3

# =========================
# DATABASE
# =========================
print("Loading database...")
try:
    database = RunnerDatabase(DB_PATH)
except ValueError as e:
    print(f"ERROR DATABASE: {e}")
    exit(1)
except Exception as e:
    print(f"ERROR DATABASE: Gagal membaca berkas database ({e}). "
          f"Pastikan berkas tidak rusak dan berformat .csv atau .xlsx.")
    exit(1)
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
print("MODEL_PATH_ACTUAL:", os.path.abspath(MODEL_PATH))
detector = Detector(MODEL_PATH, CONFIDENCE)
print("Model loaded.")

time_ocr = TimeOCR()

# Inisialisasi sistem tracking sesuai mode yang dipilih
if TRACKING_MODE == "zone":
    finish_system = FinishTimeSystem(min_frame_valid=MIN_FRAME_VALID)
else:
    grid_tracker = GridTrackingSystem(grid_size=GRID_SIZE, min_frame_valid=MIN_FRAME_VALID)

# detected_numbers: dict berisi SEMUA bib yang PERNAH match minimal sekali
# (dipakai HANYA untuk info "Frame Pertama Match" pada hasil akhir).
detected_numbers = {}

frame_count = 0
processed_frame_count = 0
cer_records = {}

# List ini menampung SETIAP hasil bacaan OCR mentah (sebelum/terlepas dari
# hasil fuzzy matching), lengkap dengan frame, confidence, dan area box.
ocr_raw_log = []

# =========================
# LOOP - FASE 1 (per frame)
# =========================
print("Mulai deteksi...")
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        break

    frame_count += 1

    h, w = frame.shape[:2]
    scale = RESIZE_WIDTH / w
    frame = cv2.resize(frame, (RESIZE_WIDTH, int(h * scale)))

    if not AUTO_MODE:
        if frame_count % FRAME_SKIP != 0:
            continue

    processed_frame_count += 1

    # --- BARU: simpan 1 contoh gambar akuisisi data (frame mentah,
    # sebelum digambar bounding box apapun), untuk Subbab 4.1.1 ---
    if not akuisisi_tersimpan:
        akuisisi_path = os.path.join(GAMBAR_DIR, f"akuisisi_{video_tag}.jpg")
        cv2.imwrite(akuisisi_path, frame)
        akuisisi_tersimpan = True
        print(f"GAMBAR_TERSIMPAN: {akuisisi_path}")

    boxes = detector.detect(frame)  # sekarang (x1,y1,x2,y2,conf)

    if AUTO_MODE:
        num_boxes = len(boxes)
        dynamic_skip = 1 if (num_boxes > 5 or num_boxes <= 2) else 2
        if frame_count % dynamic_skip != 0:
            continue

    progress = int((frame_count / total_frames) * 100)
    print(f"PROGRESS:{progress}")

    current_time = time_ocr.read_time(frame)

    for (x1, y1, x2, y2, conf) in boxes:
        crop = frame[y1:y2, x1:x2]
        if crop is None or crop.size == 0:
            continue

                # Koreksi rotasi dan penajaman citra diuji sebagai konfigurasi
        # alternatif, tidak diaktifkan pada konfigurasi final (lihat Bab 3)
        if PAKAI_ROTASI:
            crop = correct_rotation(crop)

        if SHARPEN_AMOUNT > 0:
            crop = sharpen_image(crop, amount=SHARPEN_AMOUNT)

        # --- BARU: simpan 1 rangkaian contoh tahapan preprocessing
        # (7 gambar: asli s.d. hasil Otsu), untuk Subbab 4.1.4.
        # Direkonstruksi manual mengikuti 6 tahap yang didokumentasikan
        # pada Bab 2/3, terlepas dari implementasi internal
        # preprocess_image(), supaya progresnya bisa difoto tiap tahap.
        if not preprocessing_tersimpan and crop.shape[0] > 5 and crop.shape[1] > 5:
            cv2.imwrite(os.path.join(GAMBAR_DIR, "preprocessing_1_asli.jpg"), crop)

            step_resize1 = cv2.resize(crop, None, fx=2, fy=2)
            cv2.imwrite(os.path.join(GAMBAR_DIR, "preprocessing_2_resize2x.jpg"), step_resize1)

            step_gray = cv2.cvtColor(step_resize1, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(os.path.join(GAMBAR_DIR, "preprocessing_3_grayscale.jpg"), step_gray)

            step_resize2 = cv2.resize(step_gray, None, fx=3, fy=3)
            cv2.imwrite(os.path.join(GAMBAR_DIR, "preprocessing_4_resize3x.jpg"), step_resize2)

            step_blur = cv2.GaussianBlur(step_resize2, (3, 3), 0)
            cv2.imwrite(os.path.join(GAMBAR_DIR, "preprocessing_5_gaussianblur.jpg"), step_blur)

            _, step_otsu = cv2.threshold(step_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            cv2.imwrite(os.path.join(GAMBAR_DIR, "preprocessing_6_otsu.jpg"), step_otsu)

            preprocessing_tersimpan = True
            print(f"GAMBAR_TERSIMPAN: tahapan preprocessing -> {GAMBAR_DIR}")

        crop_resized = cv2.resize(crop, None, fx=2, fy=2)
        processed = preprocess_image(crop_resized)

        if frame_count % INTERVAL_OCR == 0:
            if OCR_ENGINE == "tesseract":
                text = read_text_tesseract(processed)
            else:
                text = read_text(processed)
        else:
            text = ""

        box_label = "BIB"
        box_color = (0, 255, 255)  # kuning = default/belum ada kepastian match

        if text:
            if TRACKING_MODE == "zone":
                matched_bib, runner, match_score, second_score = database.get_runner(text)

                ocr_raw_log.append({
                    "frame": frame_count,
                    "ocr_text_raw": text,
                    "matched_bib": matched_bib if runner else None,
                    "match_score": match_score,
                    "second_score": second_score,
                    "confidence": round(float(conf), 4),
                    "area": (x2 - x1) * (y2 - y1),
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2
                })

                if runner:
                    if matched_bib not in cer_records:
                        cer_records[matched_bib] = hitung_cer(str(matched_bib), str(text))

                    if matched_bib not in detected_numbers:
                        detected_numbers[matched_bib] = frame_count

                    print(f"DETECTED:{matched_bib}:FRAME:{frame_count}:TOTAL:{total_frames}")
                    box_label = f"BIB: {matched_bib}"
                    # --- PERBAIKAN WARNA: hijau HANYA jika berhasil match
                    # ke database, sesuai dokumentasi Bab 2 (sebelumnya
                    # selalu hijau terlepas status match) ---
                    box_color = (0, 255, 0)  # hijau = ditemukan di database

                                        # Syarat timestamp dihapus: deteksi tetap diproses meskipun
                    # overlay jam tidak berhasil dibaca, karena tidak semua video
                    # menyediakan overlay tersebut.
                    finish_system.process(
                        matched_bib, text, x1, y1, x2, y2,
                        conf, frame, current_time if current_time else ""
                    )
                else:
                    # OCR berhasil baca teks, tapi TIDAK ketemu di database
                    box_label = f"OCR: {text} (?)"
                    box_color = (0, 255, 255)  # kuning = tidak ditemukan di database
            else:
                grid_tracker.add_detection(text, x1, y1, x2, y2, conf, frame, current_time)
                box_label = f"OCR: {text}"
                box_color = (0, 255, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        y_text = max(20, y1 - 10)
        cv2.putText(frame, box_label, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

        # --- BARU: simpan hingga 3 contoh frame hasil deteksi (lengkap
        # dengan bounding box & label) untuk Subbab 4.1.3 / 4.1.9 ---
        if deteksi_tersimpan < MAKS_CONTOH_DETEKSI and text:
            deteksi_tersimpan += 1
            deteksi_path = os.path.join(
                GAMBAR_DIR, f"deteksi_contoh_{deteksi_tersimpan}.jpg"
            )
            cv2.imwrite(deteksi_path, frame)
            print(f"GAMBAR_TERSIMPAN: {deteksi_path}")

    cv2.imwrite(FRAME_PATH, frame)

cap.release()

# =========================
# FASE 2 (hanya untuk mode grid) - agregasi & matching sekali
# =========================
final_results = {}

if TRACKING_MODE == "grid":
    final_results = grid_tracker.finalize(database, cer_func=hitung_cer)
    for bib, info in final_results.items():
        if bib not in detected_numbers:
            detected_numbers[bib] = None
        cer_records[bib] = info.get("cer", 0.0)
        print(f"DETECTED:{bib}")
else:
    final_results = finish_system.get_results()

# =========================
# HITUNG DAN CETAK FPS & CER
# =========================
end_time = time.time()
total_time = end_time - start_time
fps_result = processed_frame_count / total_time if total_time > 0 else 0

if cer_records:
    rata_rata_cer = (sum(cer_records.values()) / len(cer_records)) * 100
else:
    rata_rata_cer = 0.0

print("Video selesai diproses.")
print(f"FPS_RESULT:{fps_result:.2f}")
print(f"TOTAL_TIME:{total_time:.2f}")
print(f"PROCESSED_FRAMES:{processed_frame_count}")
print(f"CER_RESULT (vs matched_bib, BUKAN ground truth independen):{rata_rata_cer:.2f}%")

# =========================
# EXPORT
# =========================
print("Menyimpan hasil...")

data = []
for bib in final_results:
    runner = database.db.get(bib, {})
    time_str = final_results[bib].get("time") or "-"
    frame_pertama = detected_numbers.get(bib, "-")

    data.append({
        "Bib": bib,
        "Nama": runner.get("nama", ""),
        "Kategori": runner.get("kategori", ""),
        "Waktu Finish": time_str,
        "Frame Pertama Match": frame_pertama
    })

df = pd.DataFrame(data)
csv_out = os.path.join(OUTPUT_DIR, "hasil_bib.csv")
df.to_csv(csv_out, index=False)

# Simpan salinan arsip dengan nama sesuai video, agar tidak tertimpa
TAG = os.environ.get("EKSPERIMEN_TAG", video_tag)
csv_arsip = os.path.join(OUTPUT_DIR, f"hasil_bib_{TAG}.csv")
df.to_csv(csv_arsip, index=False)
print(f"Arsip tersimpan: {csv_arsip}")

print(f"Tersimpan: {len(data)} runner terdeteksi -> {csv_out}")

df_ocr_log = pd.DataFrame(ocr_raw_log)
ocr_log_path = os.path.join(OUTPUT_DIR, "ocr_raw_log.csv")
df_ocr_log.to_csv(ocr_log_path, index=False)

# Simpan salinan arsip dengan nama sesuai video
ocr_arsip = os.path.join(OUTPUT_DIR, f"ocr_raw_log_{TAG}.csv")
df_ocr_log.to_csv(ocr_arsip, index=False)
print(f"Arsip log tersimpan: {ocr_arsip}")
print(f"Log OCR mentah tersimpan: {len(ocr_raw_log)} baris -> {ocr_log_path}")

print(f"Seluruh gambar contoh untuk Bab 4 tersimpan di: {GAMBAR_DIR}")
print("DONE")