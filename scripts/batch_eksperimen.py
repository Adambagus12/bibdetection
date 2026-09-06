"""
batch_eksperimen.py
===================
Menjalankan seluruh eksperimen Bab 4 secara berurutan dengan konfigurasi
preprocessing lima tahap.

Hasil tiap konfigurasi disimpan dengan nama berbeda sehingga tidak saling
menimpa. Skrip dapat dihentikan dan dilanjutkan kembali; konfigurasi yang
hasilnya sudah ada akan dilewati.
"""

import os
import subprocess
import sys
import time

BASE = r"D:\bib - detection"
PY = sys.executable
MAIN = os.path.join(BASE, "scripts", "main.py")
OUT = os.path.join(BASE, "scripts", "output")

# ---- Pengaturan jeda ----
JEDA_AWAL_MENIT = 0   # jeda sebelum konfigurasi pertama (0 = langsung jalan)
JEDA_TIAP = 3          # jeda pendinginan setiap N konfigurasi
JEDA_MENIT = 15        # lama jeda pendinginan (menit)

V1 = os.path.join(BASE, "vidio", "uji1.mp4")
V2 = os.path.join(BASE, "vidio", "uji2.mp4")
DB = os.path.join(BASE, "SHARE RESULT BANUA RUN 2025.xlsx")

M_V8N = os.path.join(BASE, "models", "best_v8n.pt")
M_V8S = os.path.join(BASE, "models", "best_v8s.pt")
M_V5S = os.path.join(BASE, "models", "best_v5s.pt")
M_BASE = os.path.join(BASE, "models", "best_roboflow_v8n.pt")

# tag, video, conf, tracking, grid, ocr, model, interval, minframe, ambig
EKSPERIMEN = [
    # --- 1. Perbandingan arsitektur model ---
    ("model_v5s_v1",  V1, "0.45", "zone", "120", "easyocr", M_V5S, "3", "2", "10"),
    ("model_v5s_v2",  V2, "0.45", "zone", "120", "easyocr", M_V5S, "3", "2", "10"),
    ("model_v8n_v1",  V1, "0.45", "zone", "120", "easyocr", M_V8N, "3", "2", "10"),
    ("model_v8n_v2",  V2, "0.45", "zone", "120", "easyocr", M_V8N, "3", "2", "10"),
    ("model_v8s_v1",  V1, "0.45", "zone", "120", "easyocr", M_V8S, "3", "2", "10"),
    ("model_v8s_v2",  V2, "0.45", "zone", "120", "easyocr", M_V8S, "3", "2", "10"),

    # --- 2. Perbandingan confidence threshold ---
    ("conf025_v1", V1, "0.25", "zone", "120", "easyocr", M_V8N, "3", "2", "10"),
    ("conf025_v2", V2, "0.25", "zone", "120", "easyocr", M_V8N, "3", "2", "10"),
    ("conf065_v1", V1, "0.65", "zone", "120", "easyocr", M_V8N, "3", "2", "10"),
    ("conf065_v2", V2, "0.65", "zone", "120", "easyocr", M_V8N, "3", "2", "10"),

    # --- 3. Perbandingan OCR engine ---
    ("tesseract_v1", V1, "0.45", "zone", "120", "tesseract", M_V8N, "3", "2", "10"),
    ("tesseract_v2", V2, "0.45", "zone", "120", "tesseract", M_V8N, "3", "2", "10"),

    # --- 4. Perbandingan metode tracking ---
    ("grid120_v1", V1, "0.45", "grid", "120", "easyocr", M_V8N, "3", "2", "10"),
    ("grid250_v1", V1, "0.45", "grid", "250", "easyocr", M_V8N, "3", "2", "10"),

    # --- 5. Deteksi ambiguitas dinonaktifkan ---
    ("noambig_v1", V1, "0.45", "zone", "120", "easyocr", M_V8N, "3", "2", "0"),
    ("noambig_v2", V2, "0.45", "zone", "120", "easyocr", M_V8N, "3", "2", "0"),

    # --- 6. Interval OCR alternatif ---
    ("interval2_v1", V1, "0.45", "zone", "120", "easyocr", M_V8N, "2", "2", "10"),
    ("interval7_v1", V1, "0.45", "zone", "120", "easyocr", M_V8N, "7", "2", "10"),

    # --- 7. Model base ---
    ("base_v1", V1, "0.45", "zone", "120", "easyocr", M_BASE, "3", "2", "10"),
    ("base_v2", V2, "0.45", "zone", "120", "easyocr", M_BASE, "3", "2", "10"),

    # --- 8. Konfigurasi final ---
    ("final_v1", V1, "0.45", "zone", "120", "easyocr", M_V8N, "3", "2", "10"),
    ("final_v2", V2, "0.45", "zone", "120", "easyocr", M_V8N, "3", "2", "10"),
]


def sudah_ada(tag):
    return os.path.isfile(os.path.join(OUT, f"hasil_bib_{tag}.csv"))


def hitung_mundur(menit, pesan):
    for sisa_detik in range(menit * 60, 0, -1):
        m, d = divmod(sisa_detik, 60)
        print(f"\r        {pesan} {m:02d}:{d:02d}  "
              f"(Ctrl+C untuk berhenti)", end="", flush=True)
        time.sleep(1)
    print("\r        Melanjutkan..." + " " * 45)
    print()


total = len(EKSPERIMEN)
lewat = sum(1 for e in EKSPERIMEN if sudah_ada(e[0]))
kerja = total - lewat

print("=" * 74)
print("BATCH EKSPERIMEN - PREPROCESSING LIMA TAHAP")
print("=" * 74)
print(f"Total konfigurasi : {total}")
print(f"Sudah selesai     : {lewat}")
print(f"Akan dijalankan   : {kerja}")
print(f"Perkiraan proses  : {kerja * 18} menit")
if kerja > 0:
    jeda_total = (kerja // JEDA_TIAP) * JEDA_MENIT + JEDA_AWAL_MENIT
    print(f"Perkiraan jeda    : {jeda_total} menit")
    print(f"Perkiraan total   : {kerja * 18 + jeda_total} menit "
          f"({(kerja * 18 + jeda_total) / 60:.1f} jam)")
print("=" * 74)
print()

if kerja == 0:
    print("Semua konfigurasi sudah selesai.")
    raise SystemExit

if JEDA_AWAL_MENIT > 0:
    print(f"Menunggu {JEDA_AWAL_MENIT} menit sebelum mulai...")
    hitung_mundur(JEDA_AWAL_MENIT, "Mulai dalam")

mulai_semua = time.time()
selesai = 0

for i, (tag, video, conf, track, grid, ocr, model, interval, minfr, ambig) in enumerate(EKSPERIMEN, 1):
    if sudah_ada(tag):
        print(f"[{i:>2}/{total}] {tag:<16} SUDAH ADA, dilewati")
        continue

    selesai += 1
    print(f"[{i:>2}/{total}] {tag:<16} berjalan...")

    env = os.environ.copy()
    env["EKSPERIMEN_TAG"] = tag
    env["AMBIGUITY_MARGIN"] = ambig

    t0 = time.time()
    proses = subprocess.Popen(
        [PY, MAIN, video, DB, conf, "1", "0", track, grid, ocr,
         model, interval, minfr],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1
    )

    galat_terakhir = ""
    for baris in proses.stdout:
        baris = baris.rstrip()
        if baris.startswith("PROGRESS:"):
            try:
                pct = int(baris.split(":")[1])
            except (ValueError, IndexError):
                continue
            lewat_detik = time.time() - t0
            sisa_detik = (lewat_detik / pct * (100 - pct)) if pct > 0 else 0
            isi = int(24 * pct / 100)
            print(f"\r        [{'#'*isi}{'-'*(24-isi)}] {pct:>3}%  "
                  f"sisa ~{sisa_detik/60:.1f} menit", end="", flush=True)
        elif "Error" in baris or "Traceback" in baris:
            galat_terakhir = baris

    proses.wait()
    durasi = (time.time() - t0) / 60

    if proses.returncode == 0:
        print(f"\r        [{'#'*24}] 100%  selesai dalam "
              f"{durasi:.1f} menit" + " " * 15)
    else:
        print(f"\r        GAGAL setelah {durasi:.1f} menit" + " " * 25)
        if galat_terakhir:
            print(f"        {galat_terakhir[:100]}")

    sisa = kerja - selesai
    if sisa > 0:
        print(f"        sisa {sisa} konfigurasi, perkiraan ~{sisa * durasi:.0f} menit")

        if selesai % JEDA_TIAP == 0:
            print()
            print(f"        Jeda pendinginan {JEDA_MENIT} menit...")
            hitung_mundur(JEDA_MENIT, "Melanjutkan dalam")

print()
print("=" * 74)
print(f"SELESAI dalam {(time.time() - mulai_semua)/60:.1f} menit")
print(f"Hasil tersimpan di: {OUT}")
print("=" * 74)