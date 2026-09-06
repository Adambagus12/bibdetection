"""
Skrip 2: Visualisasi Zona Finish (untuk Tahap 7 kerangka pemikiran)
=====================================================================
Tujuan: Menghasilkan gambar frame video dengan garis batas zona finish
(20%-80% lebar frame) digambar di atasnya, plus bounding box bib yang
terdeteksi, supaya pembaca laporan bisa lihat langsung area mana yang
dianggap "zona finish" oleh sistem.

Catatan: Meskipun filter zona finish akhirnya DIHAPUS pada konfigurasi
final (lihat Bab 4, Subbab bug_multiframe), gambar ini tetap relevan untuk
menjelaskan tahap 7 pada kerangka pemikiran Bab 3 SEBAGAI BAGIAN DARI
eksperimen yang diuji (Tabel bab4_tracking_compare), bukan sebagai
konfigurasi final. Sebutkan ini di caption gambar Bab 4, misalnya:
"Ilustrasi Zona Finish (20%-80% Lebar Frame) pada Tahap Pengujian
Perbandingan Metode Tracking".

Cara pakai:
  1. Sesuaikan VIDEO_PATH dan FRAME_TARGET (pilih frame yang ramai pelari
     supaya ilustrasi lebih jelas, tidak harus sama dengan skrip 1).
  2. Jalankan: python 2_zona_finish_overlay.py
  3. Hasil: ./output/zona_finish_overlay.jpg
"""

import cv2
import os

# ==== SESUAIKAN INI ====
VIDEO_PATH = r"D:\bib - detection\vidio\uji1.mp4"
FRAME_TARGET = 300          # pilih frame dengan beberapa pelari terlihat
RESIZE_WIDTH = 960
MODEL_PATH = r"D:\bib - detection\models\best_v8n.pt"
CONF_THRESHOLD = 0.45
IOU_THRESHOLD = 0.45
ZONA_MIN = 0.20               # 20% lebar frame (sesuai Bab 3)
ZONA_MAX = 0.80               # 80% lebar frame
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

    h, w = frame.shape[:2]
    scale = RESIZE_WIDTH / w
    frame_resized = cv2.resize(frame, (RESIZE_WIDTH, int(h * scale)))
    h2, w2 = frame_resized.shape[:2]

    overlay = frame_resized.copy()

    # Gambar garis batas zona finish (vertikal, di 20% dan 80% lebar)
    x_min = int(w2 * ZONA_MIN)
    x_max = int(w2 * ZONA_MAX)
    cv2.line(overlay, (x_min, 0), (x_min, h2), (0, 0, 255), 2)
    cv2.line(overlay, (x_max, 0), (x_max, h2), (0, 0, 255), 2)

    # Beri area zona finish warna semi-transparan agar jelas
    zona_layer = overlay.copy()
    cv2.rectangle(zona_layer, (x_min, 0), (x_max, h2), (0, 255, 255), -1)
    overlay = cv2.addWeighted(zona_layer, 0.15, overlay, 0.85, 0)

    cv2.putText(overlay, "Zona Finish (20%-80%)", (x_min + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Deteksi bib dan gambar bounding box, tandai apakah di dalam/luar zona
    try:
        from ultralytics import YOLO
        model = YOLO(MODEL_PATH)
        results = model(frame_resized, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
        boxes = results[0].boxes

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cx = (x1 + x2) / 2
            di_dalam_zona = x_min <= cx <= x_max
            warna = (0, 255, 0) if di_dalam_zona else (128, 128, 128)
            label = "Dalam zona" if di_dalam_zona else "Luar zona"
            cv2.rectangle(overlay, (x1, y1), (x2, y2), warna, 2)
            cv2.putText(overlay, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, warna, 1)

        print(f"Ditemukan {len(boxes)} deteksi pada frame {FRAME_TARGET}")
    except ImportError:
        print("Ultralytics tidak terinstall di environment ini. "
              "Jalankan skrip ini di venv proyekmu (d:\\bib-detection).")

    path_out = os.path.join(OUTPUT_DIR, "zona_finish_overlay.jpg")
    cv2.imwrite(path_out, overlay)
    print(f"Hasil disimpan: {path_out}")
    break

cap.release()