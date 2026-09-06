"""
Skrip 3: Ilustrasi Best Detection Scoring (untuk Tahap 8 kerangka pemikiran)
==============================================================================
Tujuan: Mengambil crop bib 6406 dari 3 frame spesifik (957, 1008, 1020)
yang datanya sudah ada di Tabel bab4_scoring_contoh, supaya pembaca bisa
lihat langsung PERBANDINGAN VISUAL kenapa frame 1020 dipilih sebagai
"terbaik" meskipun confidence-nya bukan yang tertinggi.

Skrip ini TIDAK menjalankan deteksi ulang -- ia memakai koordinat yang
sudah kamu catat di ocr_raw_log.csv, supaya hasilnya 100% konsisten
dengan angka yang sudah kamu tulis di Tabel bab4_scoring_contoh.

Cara pakai:
  1. Sesuaikan VIDEO_PATH.
  2. Isi FRAMES_DATA di bawah dengan koordinat asli dari ocr_raw_log.csv
     untuk ketiga frame (957, 1008, 1020) pada bib 6406. Buka CSV kamu,
     cari baris dengan frame_id tersebut dan matched_bib=6406, lalu
     salin nilai x1,y1,x2,y2 dari kolom yang sesuai.
  3. Jalankan: python 3_best_detection_scoring.py
  4. Hasil: ./output/scoring_frame_957.jpg, scoring_frame_1008.jpg,
     scoring_frame_1020.jpg, dan gabungan scoring_comparison.jpg
"""

import cv2
import os

# ==== SESUAIKAN INI ====
VIDEO_PATH = r"D:\bib - detection\vidio\uji1.mp4"
RESIZE_WIDTH = 960

# Koordinat asli dari ocr_raw_log.csv untuk bib 6406 (Video 1)
FRAMES_DATA = [
    {"frame": 957,  "conf": 0.6885, "area": 925,  "x1": 474, "y1": 207, "x2": 511, "y2": 232},
    {"frame": 1008, "conf": 0.8216, "area": 2150, "x1": 757, "y1": 245, "x2": 807, "y2": 288},
    {"frame": 1020, "conf": 0.7675, "area": 2499, "x1": 842, "y1": 264, "x2": 891, "y2": 315},
]
# ========================

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)
frame_count = 0
target_frames = {d["frame"]: d for d in FRAMES_DATA}
crops = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    if frame_count in target_frames:
        data = target_frames[frame_count]

        h, w = frame.shape[:2]
        scale = RESIZE_WIDTH / w
        frame_resized = cv2.resize(frame, (RESIZE_WIDTH, int(h * scale)))

        x1, y1, x2, y2 = data["x1"], data["y1"], data["x2"], data["y2"]
        if x1 == x2 or y1 == y2:
            print(f"[PERINGATAN] Koordinat untuk frame {frame_count} belum diisi "
                  f"(masih placeholder 0,0,0,0). Isi dulu dari ocr_raw_log.csv kamu.")
            continue

        crop = frame_resized[y1:y2, x1:x2]
        # Perbesar crop supaya kelihatan jelas saat dibandingkan (semua 5x)
        crop_besar = cv2.resize(crop, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)

        skor = data["area"] * data["conf"]
        label = f"F{frame_count} | conf={data['conf']:.4f} | area={data['area']} | skor={skor:.1f}"
        cv2.putText(crop_besar, label, (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        path_out = os.path.join(OUTPUT_DIR, f"scoring_frame_{frame_count}.jpg")
        cv2.imwrite(path_out, crop_besar)
        crops[frame_count] = crop_besar
        print(f"Disimpan: {path_out}  (skor={skor:.1f})")

        if len(crops) == len(FRAMES_DATA):
            break

cap.release()

# Gabungkan ketiga crop jadi satu gambar perbandingan side-by-side
if len(crops) == len(FRAMES_DATA):
    ordered = [crops[d["frame"]] for d in FRAMES_DATA]
    max_h = max(c.shape[0] for c in ordered)
    resized_ordered = []
    for c in ordered:
        h, w = c.shape[:2]
        new_w = int(w * (max_h / h))
        resized_ordered.append(cv2.resize(c, (new_w, max_h)))
    combined = cv2.hconcat(resized_ordered)
    path_combined = os.path.join(OUTPUT_DIR, "scoring_comparison.jpg")
    cv2.imwrite(path_combined, combined)
    print(f"\nGambar perbandingan gabungan disimpan: {path_combined}")
else:
    print("\nBeberapa frame gagal diproses -- cek koordinat FRAMES_DATA dan "
          "pastikan video punya cukup banyak frame.")