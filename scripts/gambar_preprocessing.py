"""
gambar_preprocessing.py
=======================
Menghasilkan gambar tahapan preprocessing untuk dokumentasi laporan,
pada dua kondisi visual: baik dan menantang.
"""

import cv2
import os
import pandas as pd

BASE = r"D:\bib - detection"
VIDEO = os.path.join(BASE, "vidio", "uji1.mp4")
LOG = os.path.join(BASE, "scripts", "output", "ocr_raw_log_final_v1.csv")
OUT = os.path.join(BASE, "scripts", "output", "gambar_skripsi")

RESIZE_WIDTH = 960

# Frame terpilih: (nomor frame, prefiks nama berkas)
PILIHAN = [
    (33, "preprocessing"),          # kondisi baik  - bib 10062, area 5.568 px
    (201, "preprocessing_sulit"),   # kondisi sulit - bib 10420, area 620 px
]

os.makedirs(OUT, exist_ok=True)
df = pd.read_csv(LOG)


def simpan_tahapan(crop, prefiks):
    """Simpan citra tiap tahap preprocessing."""
    cv2.imwrite(os.path.join(OUT, f"{prefiks}_1_asli.jpg"), crop)

    t2 = cv2.resize(crop, None, fx=2, fy=2)
    cv2.imwrite(os.path.join(OUT, f"{prefiks}_2_resize2x.jpg"), t2)

    t3 = cv2.cvtColor(t2, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(OUT, f"{prefiks}_3_grayscale.jpg"), t3)

    t4 = cv2.resize(t3, None, fx=3, fy=3)
    cv2.imwrite(os.path.join(OUT, f"{prefiks}_4_resize3x.jpg"), t4)

    t5 = cv2.GaussianBlur(t4, (3, 3), 0)
    cv2.imwrite(os.path.join(OUT, f"{prefiks}_5_gaussianblur.jpg"), t5)

    _, t6 = cv2.threshold(t5, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite(os.path.join(OUT, f"{prefiks}_6_otsu.jpg"), t6)


print("=" * 68)
print("PEMBUATAN GAMBAR TAHAPAN PREPROCESSING")
print("=" * 68)

target = {f: p for f, p in PILIHAN}
cap = cv2.VideoCapture(VIDEO)
count = 0

while target:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1

    if count not in target:
        continue

    prefiks = target.pop(count)
    h, w = frame.shape[:2]
    frame = cv2.resize(frame, (RESIZE_WIDTH, int(h * RESIZE_WIDTH / w)))

    baris = df[df["frame"] == count]
    if len(baris) == 0:
        print(f"  frame {count}: tidak ada deteksi pada log")
        continue

    baris["area"] = (baris["x2"] - baris["x1"]) * (baris["y2"] - baris["y1"])
    r = baris.nlargest(1, "area").iloc[0]
    x1, y1, x2, y2 = int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"])
    crop = frame[max(0, y1):y2, max(0, x1):x2]

    if crop.size == 0:
        print(f"  frame {count}: crop kosong")
        continue

    simpan_tahapan(crop, prefiks)
    ocr = str(r["ocr_text_raw"]).replace(".0", "")
    area = int(r["area"])
    print(f"  frame {count:<5} -> {prefiks}_*.jpg   "
          f"(OCR '{ocr}', area {area} px)")

cap.release()

print()
print(f"Gambar tersimpan di: {OUT}")
print("Periksa berkas *_6_otsu.jpg untuk memastikan digit terbaca jelas.")