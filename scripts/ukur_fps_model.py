"""
ukur_fps_model.py
=================
Mengukur kecepatan inferensi murni (tanpa OCR) untuk membandingkan
ketiga arsitektur model pada kondisi perangkat yang sama.
"""

import cv2
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.detector import Detector

BASE = r"D:\bib - detection"
VIDEO = os.path.join(BASE, "vidio", "uji1.mp4")

MODEL = {
    "YOLOv5s": os.path.join(BASE, "models", "best_v5s.pt"),
    "YOLOv8n": os.path.join(BASE, "models", "best_v8n.pt"),
    "YOLOv8s": os.path.join(BASE, "models", "best_v8s.pt"),
}

RESIZE_WIDTH = 960
CONF = 0.45
IOU = 0.45
JUMLAH_FRAME = 200      # frame yang diukur
PEMANASAN = 10          # frame awal diabaikan


def ukur(nama, path):
    if not os.path.isfile(path):
        print(f"  {nama:<10} berkas model tidak ditemukan")
        return None

    detector = Detector(path, CONF, IOU)
    cap = cv2.VideoCapture(VIDEO)

    waktu = []
    deteksi = 0
    n = 0

    while n < JUMLAH_FRAME + PEMANASAN:
        ret, frame = cap.read()
        if not ret:
            break
        n += 1

        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (RESIZE_WIDTH, int(h * RESIZE_WIDTH / w)))

        t0 = time.perf_counter()
        boxes = detector.detect(frame)
        t1 = time.perf_counter()

        if n > PEMANASAN:
            waktu.append(t1 - t0)
            deteksi += len(boxes)

    cap.release()

    if not waktu:
        return None

    rata = sum(waktu) / len(waktu)
    return {
        "nama": nama,
        "ms": rata * 1000,
        "fps": 1 / rata,
        "deteksi": deteksi,
        "ukuran": os.path.getsize(path) / (1024 * 1024),
    }


print("=" * 72)
print("PENGUKURAN KECEPATAN INFERENSI MODEL")
print("=" * 72)
print(f"Video   : {os.path.basename(VIDEO)}")
print(f"Frame   : {JUMLAH_FRAME} (setelah {PEMANASAN} frame pemanasan)")
print(f"Resolusi: {RESIZE_WIDTH} px")
print()

hasil = []
for nama, path in MODEL.items():
    print(f"  Mengukur {nama}...", end="", flush=True)
    r = ukur(nama, path)
    if r:
        hasil.append(r)
        print(f" {r['fps']:.2f} FPS")
    else:
        print(" gagal")

print()
print("=" * 72)
print("HASIL")
print("=" * 72)
print(f"{'Model':<12}{'Ukuran':<12}{'Inferensi':<14}{'FPS':<10}{'Deteksi'}")
print("-" * 72)
for r in sorted(hasil, key=lambda x: -x["fps"]):
    print(f"{r['nama']:<12}{r['ukuran']:>6.1f} MB   "
          f"{r['ms']:>7.1f} ms   {r['fps']:>6.2f}    {r['deteksi']}")
print()

if len(hasil) >= 2:
    tercepat = max(hasil, key=lambda x: x["fps"])
    print(f"Tercepat: {tercepat['nama']} ({tercepat['fps']:.2f} FPS)")
    print()
    print("Relatif terhadap model tercepat:")
    for r in sorted(hasil, key=lambda x: -x["fps"]):
        selisih = (r["fps"] / tercepat["fps"] - 1) * 100
        print(f"  {r['nama']:<12}{selisih:+6.1f}%")