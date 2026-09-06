"""
ukur_fps_final.py
=================
Menjalankan ulang konfigurasi final pada Video 1 dan Video 2 sambil
menangkap nilai FPS dan CER yang dilaporkan sistem.
"""

import os
import subprocess
import sys
import time

BASE = r"D:\bib - detection"
PY = sys.executable
MAIN = os.path.join(BASE, "scripts", "main.py")

DB = os.path.join(BASE, "SHARE RESULT BANUA RUN 2025.xlsx")
MODEL = os.path.join(BASE, "models", "best_v8n.pt")

VIDEO = [
    ("Video 1", os.path.join(BASE, "vidio", "uji1.mp4"), "fps_final_v1"),
    ("Video 2", os.path.join(BASE, "vidio", "uji2.mp4"), "fps_final_v2"),
]

print("=" * 72)
print("PENGUKURAN FPS SISTEM END-TO-END (KONFIGURASI FINAL)")
print("=" * 72)
print()

hasil = []

for label, video, tag in VIDEO:
    print(f"{label} berjalan...")

    env = os.environ.copy()
    env["EKSPERIMEN_TAG"] = tag
    env["AMBIGUITY_MARGIN"] = "10"

    t0 = time.time()
    proses = subprocess.Popen(
        [PY, MAIN, video, DB, "0.45", "1", "0", "zone", "120",
         "easyocr", MODEL, "3", "2"],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1
    )

    data = {"label": label}
    for baris in proses.stdout:
        baris = baris.rstrip()

        if baris.startswith("PROGRESS:"):
            try:
                pct = int(baris.split(":")[1])
            except (ValueError, IndexError):
                continue
            isi = int(24 * pct / 100)
            print(f"\r  [{'#'*isi}{'-'*(24-isi)}] {pct:>3}%", end="", flush=True)

        elif baris.startswith("FPS_RESULT:"):
            data["fps"] = baris.split(":", 1)[1].strip()
        elif baris.startswith("TOTAL_TIME:"):
            data["waktu"] = baris.split(":", 1)[1].strip()
        elif baris.startswith("PROCESSED_FRAMES:"):
            data["frame"] = baris.split(":", 1)[1].strip()
        elif baris.startswith("CER_RESULT"):
            data["cer"] = baris.split(":", 1)[1].strip()
        elif "Tersimpan:" in baris:
            data["runner"] = baris.split(":", 1)[1].strip()

    proses.wait()
    durasi = (time.time() - t0) / 60
    print(f"\r  [{'#'*24}] 100%  selesai dalam {durasi:.1f} menit" + " " * 10)
    print()

    hasil.append(data)

print("=" * 72)
print("HASIL PENGUKURAN")
print("=" * 72)
print(f"{'Video':<12}{'FPS':<10}{'Waktu (s)':<14}{'Frame':<10}{'CER'}")
print("-" * 72)
for d in hasil:
    print(f"{d.get('label','-'):<12}{d.get('fps','-'):<10}"
          f"{d.get('waktu','-'):<14}{d.get('frame','-'):<10}{d.get('cer','-')}")
print()
for d in hasil:
    if "runner" in d:
        print(f"{d['label']}: {d['runner']}")