"""
uji_kanal_warna.py
==================
Menguji pemilihan kanal warna yang paling kontras sebelum binarisasi,
untuk kasus bib dengan latar kuning terang.

Konfigurasi:
  A. Grayscale     - konversi standar (seperti sistem sekarang)
  B. Kanal Blue    - kuning punya nilai biru rendah, digit hitam juga rendah
  C. HSV Value     - komponen kecerahan
  D. HSV Saturation- kuning jenuh tinggi, hitam jenuh rendah
  E. Blue + tengah - kanal biru dengan crop tengah 70%
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
OUT_CSV    = r"D:\bib - detection\output\uji_kanal_warna.csv"

RESIZE_WIDTH = 960
CONF = 0.45
IOU = 0.45
INTERVAL_OCR = 3


# ============================================================
# EKSTRAKSI KANAL
# ============================================================

def kanal_gray(crop):
    return cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)


def kanal_blue(crop):
    return crop[:, :, 0]        # BGR: indeks 0 = Blue


def kanal_value(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    return hsv[:, :, 2]


def kanal_saturation(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    return hsv[:, :, 1]


def crop_tengah(crop, rasio=0.70):
    if crop is None or crop.size == 0:
        return crop
    h, w = crop.shape[:2]
    nh, nw = int(h * rasio), int(w * rasio)
    if nh < 6 or nw < 10:
        return crop
    y, x = (h - nh) // 2, (w - nw) // 2
    return crop[y:y+nh, x:x+nw]


# ============================================================
def proses(crop, fn_kanal, pakai_tengah=False):
    """Preprocessing dengan kanal warna tertentu."""
    if crop is None or crop.size == 0:
        return None
    try:
        c = crop_tengah(crop) if pakai_tengah else crop
        c = cv2.resize(c, None, fx=2, fy=2)
        satu_kanal = fn_kanal(c)
        satu_kanal = cv2.resize(satu_kanal, None, fx=3, fy=3)
        blur = cv2.GaussianBlur(satu_kanal, (3, 3), 0)
        _, th = cv2.threshold(blur, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return th
    except Exception:
        return None


VARIAN = {
    "A_grayscale":   lambda c: proses(c, kanal_gray),
    "B_blue":        lambda c: proses(c, kanal_blue),
    "C_value":       lambda c: proses(c, kanal_value),
    "D_saturation":  lambda c: proses(c, kanal_saturation),
    "E_blue_tengah": lambda c: proses(c, kanal_blue, pakai_tengah=True),
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
        print(f"\r[{'#'*isi}{'-'*(30-isi)}] {pct:5.1f}%  "
              f"deteksi: {len(hasil)}  sisa ~{sisa/60:.1f} menit",
              end="", flush=True)

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

print("=" * 80)
print("HASIL PERBANDINGAN KANAL WARNA")
print("=" * 80)
print(f"Total deteksi diproses: {len(hasil)}")
print()
print(f"{'Konfigurasi':<20}{'Terbaca':<12}{'Kosong':<12}{'Tingkat Baca'}")
print("-" * 80)
for n in VARIAN:
    t, k = stat[n]["terbaca"], stat[n]["kosong"]
    pct = t / (t + k) * 100 if (t + k) else 0
    print(f"{n:<20}{t:<12}{k:<12}{pct:>6.1f}%")
print()

beda = [r for r in hasil if len(set(r[n] for n in VARIAN)) > 1][:20]
if beda:
    print("-" * 80)
    print("CONTOH DETEKSI DENGAN HASIL BERBEDA")
    print("-" * 80)
    print(f"{'Frame':<8}" + "".join(f"{n[:13]:<15}" for n in VARIAN))
    for r in beda:
        print(f"{r['frame']:<8}" + "".join(f"{(r[n] or '-'):<15}" for n in VARIAN))
    print()

print(f"Rincian tersimpan di: {OUT_CSV}")