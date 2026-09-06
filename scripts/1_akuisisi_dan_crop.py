"""
Skrip 1: Akuisisi Frame + Crop Region Bib (untuk Tahap 2 & 4 kerangka pemikiran)
=================================================================================
Tujuan: Menghasilkan 3 gambar untuk Bab 4:
  1. Frame asli (sebelum resize)               -> gambar/akuisisi_frame_asli.jpg
  2. Frame setelah normalisasi resize ke 960px  -> gambar/akuisisi_frame_resize.jpg
  3. Crop region bib mentah (sebelum preprocessing) -> gambar/crop_bib_mentah.jpg

Cara pakai:
  1. Sesuaikan VIDEO_PATH ke path video 1 kamu.
  2. Sesuaikan FRAME_TARGET ke nomor frame yang ingin dipakai sebagai contoh
     (disarankan pakai frame yang sama dengan contoh existing kamu, mis. frame 12
     untuk bib 10062, supaya konsisten dengan napak tilas Bab 4).
  3. Jalankan: python 1_akuisisi_dan_crop.py
  4. Hasil akan tersimpan di folder ./output/
"""

import cv2
import os

# ==== SESUAIKAN INI ====
VIDEO_PATH = r"D:\bib - detection\vidio\uji1.mp4"
FRAME_TARGET = 12              # frame yang mau dijadikan contoh (sesuai napak tilas bib 10062)
RESIZE_WIDTH = 960             # sama dengan RESIZE_WIDTH di main.py kamu
MODEL_PATH = r"D:\bib - detection\models\best_v8n.pt"
CONF_THRESHOLD = 0.45
IOU_THRESHOLD = 0.45
# ========================

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print(f"Video habis sebelum mencapai frame {FRAME_TARGET}")
        break

    frame_count += 1
    if frame_count != FRAME_TARGET:
        continue

    # ---- 1. Simpan frame ASLI (sebelum resize) ----
    path_asli = os.path.join(OUTPUT_DIR, "akuisisi_frame_asli.jpg")
    cv2.imwrite(path_asli, frame)
    h_asli, w_asli = frame.shape[:2]
    print(f"[1] Frame asli disimpan: {path_asli}  ({w_asli}x{h_asli})")

    # ---- 2. Resize sesuai normalisasi sistem (RESIZE_WIDTH=960) ----
    h, w = frame.shape[:2]
    scale = RESIZE_WIDTH / w
    frame_resized = cv2.resize(frame, (RESIZE_WIDTH, int(h * scale)))
    path_resize = os.path.join(OUTPUT_DIR, "akuisisi_frame_resize.jpg")
    cv2.imwrite(path_resize, frame_resized)
    print(f"[2] Frame resize disimpan: {path_resize}  ({RESIZE_WIDTH}x{int(h*scale)})")

    # ---- 3. Deteksi bib pada frame yang sudah di-resize, lalu crop mentahnya ----
    try:
        from ultralytics import YOLO
        model = YOLO(MODEL_PATH)
        results = model(frame_resized, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)

        boxes = results[0].boxes
        if len(boxes) == 0:
            print("Tidak ada bib terdeteksi pada frame ini. Coba ganti FRAME_TARGET.")
        else:
            # Ambil deteksi pertama sebagai contoh
            box = boxes[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])

            crop = frame_resized[y1:y2, x1:x2]
            path_crop = os.path.join(OUTPUT_DIR, "crop_bib_mentah.jpg")
            cv2.imwrite(path_crop, crop)
            print(f"[3] Crop bib mentah disimpan: {path_crop}")
            print(f"    Koordinat: ({x1},{y1},{x2},{y2})  Confidence: {conf:.4f}")
            print(f"    Ukuran crop: {x2-x1}x{y2-y1} piksel "
                  f"(luas {(x2-x1)*(y2-y1)} piksel^2)")

            # Bonus: frame dengan bounding box digambar, untuk ilustrasi visual di laporan
            frame_annotated = frame_resized.copy()
            cv2.rectangle(frame_annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_annotated, f"conf={conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            path_annotated = os.path.join(OUTPUT_DIR, "akuisisi_frame_annotated.jpg")
            cv2.imwrite(path_annotated, frame_annotated)
            print(f"[bonus] Frame dengan bounding box: {path_annotated}")

    except ImportError:
        print("Ultralytics tidak terinstall di environment ini. "
              "Jalankan 'pip install ultralytics' atau jalankan skrip ini "
              "di venv proyekmu (d:\\bib-detection) yang sudah ada ultralytics-nya.")

    break

cap.release()
print("\nSelesai. Cek folder ./output/ untuk hasil gambar.")