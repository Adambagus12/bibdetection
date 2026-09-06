"""
uji_preprocessing.py
====================
Membandingkan 4 konfigurasi preprocessing pada video yang sama,
untuk melihat mana yang menghasilkan pembacaan OCR paling baik.

Konfigurasi yang diuji:
  A. Sekarang        - preprocessing 6 tahap seperti di sistem
  B. Tanpa HistEq    - histogram equalization dihilangkan
  C. Crop ketat      - crop dipersempit 15% tiap sisi
  D. Tanpa HistEq + Crop ketat
"""

import cv2
import os
import sys
import csv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.detector import Detector
from modules.ocr import read_text

# ============================================================
# KONFIGURASI
# ============================================================

VIDEO_PATH = r"D:\bib - detection\vidio\uji1.mp4"   # ganti ke video yang mau diuji
MODEL_PATH = r"D:\bib - detection\models\best_v8n.pt"
OUT_CSV    = r"D:\bib - detection\output\uji_preprocessing.csv"

RESIZE_WIDTH = 960
CONF = 0.45
IOU = 0.45
INTERVAL_OCR = 3
MARGIN_CROP = 0.15      # 15% dipotong tiap sisi untuk konfigurasi crop ketat


# ============================================================
# VARIAN PREPROCESSING
# ============================================================

def prep_A(crop):
    """Konfigurasi sekarang: resize2x, gray, resize3x, histeq, blur, otsu"""
    if crop is None or crop.size == 0:
        return None
    c = cv2.resize(crop, None, fx=2, fy=2)
    gray = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3)
    gray = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


def prep_B(crop):
    """Tanpa histogram equalization"""
    if crop is None or crop.size == 0:
        return None
    c = cv2.resize(crop, None, fx=2, fy=2)
    gray = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


def crop_ketat(crop):
    """Persempit crop 15% tiap sisi"""
    if crop is None or crop.size == 0:
        return crop
    h, w = crop.shape[:2]
    mh, mw = int(h * MARGIN_CROP), int(w * MARGIN_CROP)
    if h - 2 * mh < 5 or w - 2 * mw < 5:
        return crop
    return crop[mh:h - mh, mw:w - mw]


def prep_C(crop):
    """Crop ketat + preprocessing sekarang"""
    return prep_A(crop_ketat(crop))


def prep_D(crop):
    """Crop ketat + tanpa histeq"""
    return prep_B(crop_ketat(crop))


VARIAN = {
    "A_sekarang":      prep_A,
    "B_tanpa_histeq":  prep_B,
    "C_crop_ketat":    prep_C,
    "D_ketat_nohisteq": prep_D,
}


# ============================================================
# PROSES
# ============================================================

print("Memuat model...")
detector = Detector(MODEL_PATH, CONF, IOU)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"ERROR: tidak bisa membuka {VIDEO_PATH}")
    raise SystemExit

total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video  : {VIDEO_PATH}")
print(f"Frame  : {total}")
print(f"Varian : {', '.join(VARIAN.keys())}")
print()
print("Memproses (setiap titik = 30 frame)...")

hasil = []
statistik = {k: {"terbaca": 0, "kosong": 0} for k in VARIAN}

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

    if frame_count % 30 == 0:
        print(".", end="", flush=True)

    boxes = detector.detect(frame)
    for (x1, y1, x2, y2, conf) in boxes:
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
            if teks:
                statistik[nama]["terbaca"] += 1
            else:
                statistik[nama]["kosong"] += 1

        hasil.append(baris)

cap.release()
print("\n")

# ============================================================
# SIMPAN & RINGKASAN
# ============================================================

os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    kolom = ["frame", "conf", "area"] + list(VARIAN.keys())
    wr = csv.DictWriter(f, fieldnames=kolom)
    wr.writeheader()
    wr.writerows(hasil)

print("=" * 74)
print("RINGKASAN PERBANDINGAN PREPROCESSING")
print("=" * 74)
print(f"Total deteksi diproses: {len(hasil)}")
print()
print(f"{'Konfigurasi':<22}{'Terbaca':<12}{'Kosong':<12}{'Tingkat Baca':<14}")
print("-" * 74)
for nama in VARIAN:
    t = statistik[nama]["terbaca"]
    k = statistik[nama]["kosong"]
    pct = t / (t + k) * 100 if (t + k) else 0
    print(f"{nama:<22}{t:<12}{k:<12}{pct:>6.1f}%")
print()

# Contoh perbedaan hasil antar varian
beda = [r for r in hasil if len(set(r[n] for n in VARIAN)) > 1][:15]
if beda:
    print("-" * 74)
    print("CONTOH DETEKSI DENGAN HASIL BERBEDA ANTAR KONFIGURASI")
    print("-" * 74)
    hdr = f"{'Frame':<8}" + "".join(f"{n[:14]:<16}" for n in VARIAN)
    print(hdr)
    for r in beda:
        row = f"{r['frame']:<8}" + "".join(f"{(r[n] or '-'):<16}" for n in VARIAN)
        print(row)
    print()

print(f"Rincian lengkap tersimpan di: {OUT_CSV}")