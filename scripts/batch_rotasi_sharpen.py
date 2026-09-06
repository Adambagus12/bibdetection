"""
batch_rotasi_sharpen.py
=======================
Menjalankan eksperimen koreksi rotasi otomatis dan penajaman citra
pada konfigurasi preprocessing lima tahap.
"""

import os
import subprocess
import sys
import time

BASE = r"D:\bib - detection"
PY = sys.executable
MAIN = os.path.join(BASE, "scripts", "main.py")
OUT = os.path.join(BASE, "scripts", "output")

JEDA_TIAP = 3
JEDA_MENIT = 15

V1 = os.path.join(BASE, "vidio", "uji1.mp4")
V2 = os.path.join(BASE, "vidio", "uji2.mp4")
DB = os.path.join(BASE, "SHARE RESULT BANUA RUN 2025.xlsx")
MODEL = os.path.join(BASE, "models", "best_v8n.pt")

# tag, video, rotasi, sharpen
EKSPERIMEN = [
    ("rotasi_v1",     V1, "1", "0"),
    ("rotasi_v2",     V2, "1", "0"),
    ("sharpen15_v1",  V1, "0", "1.5"),
    ("sharpen15_v2",  V2, "0", "1.5"),
    ("sharpen05_v1",  V1, "0", "0.5"),
    ("sharpen05_v2",  V2, "0", "0.5"),
]


def sudah_ada(tag):
    return os.path.isfile(os.path.join(OUT, f"hasil_bib_{tag}.csv"))


def hitung_mundur(menit, pesan):
    for sisa in range(menit * 60, 0, -1):
        m, d = divmod(sisa, 60)
        print(f"\r        {pesan} {m:02d}:{d:02d}  (Ctrl+C untuk berhenti)",
              end="", flush=True)
        time.sleep(1)
    print("\r        Melanjutkan..." + " " * 45)
    print()


total = len(EKSPERIMEN)
lewat = sum(1 for e in EKSPERIMEN if sudah_ada(e[0]))
kerja = total - lewat

print("=" * 74)
print("EKSPERIMEN KOREKSI ROTASI DAN PENAJAMAN CITRA")
print("=" * 74)
print(f"Total konfigurasi : {total}")
print(f"Sudah selesai     : {lewat}")
print(f"Akan dijalankan   : {kerja}")
print(f"Perkiraan waktu   : {kerja * 16 + (kerja // JEDA_TIAP) * JEDA_MENIT} menit")
print("=" * 74)
print()

if kerja == 0:
    print("Semua konfigurasi sudah selesai.")
    raise SystemExit

mulai = time.time()
selesai = 0

for i, (tag, video, rot, shp) in enumerate(EKSPERIMEN, 1):
    if sudah_ada(tag):
        print(f"[{i}/{total}] {tag:<16} SUDAH ADA, dilewati")
        continue

    selesai += 1
    print(f"[{i}/{total}] {tag:<16} berjalan...")

    env = os.environ.copy()
    env["EKSPERIMEN_TAG"] = tag
    env["AMBIGUITY_MARGIN"] = "10"

    t0 = time.time()
    proses = subprocess.Popen(
        [PY, MAIN, video, DB, "0.45", "1", "0", "zone", "120", "easyocr",
         MODEL, "3", "2", rot, shp],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1
    )

    galat = ""
    for baris in proses.stdout:
        baris = baris.rstrip()
        if baris.startswith("PROGRESS:"):
            try:
                pct = int(baris.split(":")[1])
            except (ValueError, IndexError):
                continue
            lewat_dtk = time.time() - t0
            sisa_dtk = (lewat_dtk / pct * (100 - pct)) if pct > 0 else 0
            isi = int(24 * pct / 100)
            print(f"\r        [{'#'*isi}{'-'*(24-isi)}] {pct:>3}%  "
                  f"sisa ~{sisa_dtk/60:.1f} menit", end="", flush=True)
        elif "Error" in baris or "Traceback" in baris:
            galat = baris

    proses.wait()
    durasi = (time.time() - t0) / 60

    if proses.returncode == 0:
        print(f"\r        [{'#'*24}] 100%  selesai dalam {durasi:.1f} menit"
              + " " * 15)
    else:
        print(f"\r        GAGAL setelah {durasi:.1f} menit" + " " * 25)
        if galat:
            print(f"        {galat[:100]}")

    sisa = kerja - selesai
    if sisa > 0:
        print(f"        sisa {sisa} konfigurasi")
        if selesai % JEDA_TIAP == 0:
            print()
            print(f"        Jeda pendinginan {JEDA_MENIT} menit...")
            hitung_mundur(JEDA_MENIT, "Melanjutkan dalam")

print()
print("=" * 74)
print(f"SELESAI dalam {(time.time() - mulai)/60:.1f} menit")
print("=" * 74)