"""
uji_fokus_digit.py
==================
Menguji apakah pemfokusan area digit sebelum binarisasi dapat
memperbaiki pembacaan OCR pada video dengan desain bib kompleks.

Konfigurasi yang diuji:
  A. Sekarang       - preprocessing seperti di sistem
  B. Fokus kontur   - crop dipersempit ke area kontur besar dulu
  C. Crop tengah    - ambil 70% bagian tengah crop
  D. Fokus + tengah - kombinasi keduanya
"""

import cv2
import numpy as np
import os
import sys
import csv
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.detector import Detector
from modules.ocr import read_text

# ============================================================
VIDEO_PATH = r"D:\bib - detection\vidio\UJI 1.mp4"
MODEL_PATH = r"D:\bib - detection\models\best_v8n.pt"
OUT_CSV    = r"D:\bib - detection\output\uji_fokus_digit.csv"

RESIZE_WIDTH = 960
CONF = 0.45
IOU = 0.45
INTERVAL_OCR = 3


# ============================================================
# PEMFOKUSAN AREA DIGIT
# ============================================================

def fokus_kontur(crop):
    """Persempit crop ke area yang memuat kontur-kontur besar."""
    if crop is None or crop.size == 0:
        return crop
    try:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kontur, _ = cv2.findContours(th, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
        if not kontur:
            return crop

        h, w = crop.shape[:2]
        luas_min = (h * w) * 0.02          # kontur < 2% luas dianggap noise
        besar = [c for c in kontur if cv2.contourArea(c) > luas_min]
        if not besar:
            return crop

        x, y, bw, bh = cv2.boundingRect(np.vstack(besar))
        if bw < 10 or bh < 6:
            return crop
        pad = 3
        x1 = max(0, x - pad); y1 = max(0, y - pad)
        x2 = min(w, x + bw + pad); y2 = min(h, y + bh + pad)
        return crop[y1:y2, x1:x2]
    except Exception:
        return crop


def crop_tengah(crop, rasio=0.70):
    """Ambil bagian tengah crop (buang tepi atas-bawah-kiri-kanan)."""
    if crop is None or crop.size == 0:
        return crop
    h, w = crop.shape[:2]
    nh, nw = int(h * rasio), int(w * rasio)
    y = (h - nh) // 2
    x = (w - nw) // 2
    if nh < 6 or nw < 10:
        return crop
    return crop[y:y+nh, x:x+nw]


# ============================================================
# PREPROCESSING (sesuai sistem sekarang, tanpa histeq)
# ============================================================

def preprocess(crop):
    if crop is None or crop.size == 0:
        return None
    c = cv2.resize(crop, None, fx=2, fy=2)
    gray = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(blur, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


VARIAN = {
    "A_sekarang":    lambda c: preprocess(c),
    "B_fokus":       lambda c: preprocess(fokus_kontur(c)),
    "C_tengah":      lambda c: preprocess(crop_tengah(c)),
    "D_fokus_tengah": lambda c: preprocess(fokus_kontur(crop_tengah(c))),
}


# ============================================================
print("Memuat model...")
detector = Detector(MODEL_PATH, CONF, IOU)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"ERROR: tidak bisa membuka {VIDEO_PATH}")
    raise SystemExit

total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video : {os.path.basename(VIDEO_PATH)}")
print(f"Frame : {total}")
print()

hasil = []
stat = {k: {"terbaca": 0, "kosong": 0} for k in VARIAN}
mulai = time.time()
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    h, w = frame.shape[:2]
    frame = cv2.resize(frame, (RESIZE_WIDTH, int(h * RESIZE_WIDTH / w)))

    if frame_count % INTERVAL_OCR != 0:
        continue

    if frame_count % 15 == 0:
        pct = frame_count / total * 100
        lewat = time.time() - mulai
        sisa = lewat / frame_count * (total - frame_count)
        isi = int(30 * frame_count / total)
        bar = "#" * isi + "-" * (30 - isi)
        print(f"\r[{bar}] {pct:5.1f}%  deteksi: {len(hasil)}  "
              f"sisa ~{sisa/60:.1f} menit", end="", flush=True)

    for (x1, y1, x2, y2, conf) in detector.detect(frame):
        crop = frame[max(0, y1):y2, max(0, x1):x2]
        if crop is None or crop.size == 0:
            continue

        baris = {"frame": frame_count, "conf": round(float(conf), 4),
                 "area": (x2 - x1) * (y2 - y1)}
        for nama, fn in VARIAN.items():
            try:
                teks = read_text(fn(crop))
            except Exception:
                teks = ""
            baris[nama] = teks
            stat[nama]["terbaca" if teks else "kosong"] += 1
        hasil.append(baris)

cap.release()
print(f"\r[{'#'*30}] 100.0%  selesai dalam "
      f"{(time.time()-mulai)/60:.1f} menit" + " " * 20)
print()

os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    wr = csv.DictWriter(f, fieldnames=["frame", "conf", "area"] + list(VARIAN))
    wr.writeheader()
    wr.writerows(hasil)

print("=" * 74)
print("HASIL PERBANDINGAN")
print("=" * 74)
print(f"Total deteksi diproses: {len(hasil)}")
print()
print(f"{'Konfigurasi':<20}{'Terbaca':<12}{'Kosong':<12}{'Tingkat Baca'}")
print("-" * 74)
for n in VARIAN:
    t, k = stat[n]["terbaca"], stat[n]["kosong"]
    pct = t / (t + k) * 100 if (t + k) else 0
    print(f"{n:<20}{t:<12}{k:<12}{pct:>6.1f}%")
print()

beda = [r for r in hasil if len(set(r[n] for n in VARIAN)) > 1][:20]
if beda:
    print("-" * 74)
    print("CONTOH DETEKSI DENGAN HASIL BERBEDA")
    print("-" * 74)
    print(f"{'Frame':<8}" + "".join(f"{n[:13]:<15}" for n in VARIAN))
    for r in beda:
        print(f"{r['frame']:<8}" + "".join(f"{(r[n] or '-'):<15}" for n in VARIAN))
    print()

print(f"Rincian tersimpan di: {OUT_CSV}")