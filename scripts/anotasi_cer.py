"""
anotasi_cer.py
==============
Anotasi manual nomor bib SEBENARNYA dari region deteksi, untuk menghitung
CER yang dibandingkan terhadap ground truth (bukan terhadap hasil sistem).

CARA PAKAI:
  - Gambar crop bib muncul di jendela (diperbesar)
  - Ketik angka nomor bib yang Anda lihat
  - Enter  = simpan, lanjut gambar berikutnya
  - Backspace = hapus digit terakhir
  - s      = skip (tidak terbaca / bukan bib)
  - q      = keluar (progres tersimpan otomatis)

Progres disimpan tiap kali Enter, jadi aman kalau berhenti di tengah.
"""

import cv2
import os
import csv
import random
import pandas as pd

# ============================================================
# KONFIGURASI
# ============================================================

VIDEO_PATH = r"D:\bib - detection\vidio\uji1.mp4"
LOG_PATH   = r"D:\bib - detection\scripts\output\ocr_raw_log_uji1.csv"
OUT_CSV    = r"D:\bib - detection\output\anotasi_cer_video1.csv"

JUMLAH_SAMPEL = 100
SEED = 42          # supaya sampel acaknya bisa direplikasi
RESIZE_WIDTH = 960 # samakan dengan normalisasi di main.py
SKALA_TAMPIL = 6   # perbesaran crop di layar


# ============================================================
def muat_progres():
    """Baca hasil anotasi sebelumnya kalau ada."""
    if not os.path.isfile(OUT_CSV):
        return {}
    sudah = {}
    with open(OUT_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sudah[int(row["idx"])] = row
    return sudah


def simpan_progres(hasil):
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "idx", "frame", "ocr_text_raw", "matched_bib", "bib_asli", "status"
        ])
        w.writeheader()
        for k in sorted(hasil):
            w.writerow(hasil[k])


# ============================================================
print("Membaca log OCR...")
df = pd.read_csv(LOG_PATH)
df = df.reset_index(drop=True)

random.seed(SEED)
idx_terpilih = sorted(random.sample(range(len(df)), min(JUMLAH_SAMPEL, len(df))))
print(f"  {len(df)} baris log, diambil {len(idx_terpilih)} sampel acak\n")

hasil = muat_progres()
if hasil:
    print(f"Melanjutkan progres sebelumnya: {len(hasil)} sudah dianotasi\n")

sisa = [i for i in idx_terpilih if i not in hasil]
if not sisa:
    print("Semua sampel sudah dianotasi. Hapus file CSV kalau mau mulai ulang.")
    raise SystemExit

# Kumpulkan frame yang dibutuhkan
frame_dibutuhkan = {}
for i in sisa:
    f = int(df.loc[i, "frame"])
    frame_dibutuhkan.setdefault(f, []).append(i)

print(f"Mengambil {len(frame_dibutuhkan)} frame dari video...")
crops = {}
cap = cv2.VideoCapture(VIDEO_PATH)
count = 0
while frame_dibutuhkan:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count in frame_dibutuhkan:
        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (RESIZE_WIDTH, int(h * RESIZE_WIDTH / w)))
        for i in frame_dibutuhkan.pop(count):
            r = df.loc[i]
            x1, y1, x2, y2 = int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"])
            crop = frame[max(0, y1):y2, max(0, x1):x2]
            if crop.size > 0:
                crops[i] = crop
cap.release()
print(f"  {len(crops)} crop berhasil diambil\n")

print("=" * 70)
print("MULAI ANOTASI")
print("=" * 70)
print("  ketik angka  = nomor bib yang Anda lihat")
print("  Enter        = simpan & lanjut")
print("  Backspace    = hapus digit terakhir")
print("  s            = skip (tidak terbaca)")
print("  q            = keluar")
print("=" * 70)
print()

urutan = [i for i in sisa if i in crops]
posisi = 0

while posisi < len(urutan):
    i = urutan[posisi]
    r = df.loc[i]
    crop = crops[i]

    ketikan = ""
    while True:
        besar = cv2.resize(crop, None, fx=SKALA_TAMPIL, fy=SKALA_TAMPIL,
                           interpolation=cv2.INTER_CUBIC)
        h, w = besar.shape[:2]
        panel = cv2.copyMakeBorder(besar, 90, 70, 20, 20,
                                   cv2.BORDER_CONSTANT, value=(30, 30, 30))

        info = f"[{posisi+1}/{len(urutan)}]  frame {int(r['frame'])}"
        cv2.putText(panel, info, (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(panel, f"OCR baca: {r['ocr_text_raw']}", (20, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        cv2.putText(panel, f"Bib asli: {ketikan}_", (20, panel.shape[0] - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Anotasi CER", panel)
        key = cv2.waitKey(0) & 0xFF

        if key == ord('q'):
            simpan_progres(hasil)
            cv2.destroyAllWindows()
            print(f"\nKeluar. {len(hasil)} sampel tersimpan di {OUT_CSV}")
            raise SystemExit

        elif key == ord('s'):
            hasil[i] = {"idx": i, "frame": int(r["frame"]),
                        "ocr_text_raw": r["ocr_text_raw"],
                        "matched_bib": r["matched_bib"],
                        "bib_asli": "", "status": "skip"}
            simpan_progres(hasil)
            print(f"  [{posisi+1}/{len(urutan)}] frame {int(r['frame'])}: SKIP")
            posisi += 1
            break

        elif key in (8, 127):          # backspace
            ketikan = ketikan[:-1]

        elif key in (13, 10):          # enter
            if ketikan:
                hasil[i] = {"idx": i, "frame": int(r["frame"]),
                            "ocr_text_raw": r["ocr_text_raw"],
                            "matched_bib": r["matched_bib"],
                            "bib_asli": ketikan, "status": "ok"}
                simpan_progres(hasil)
                print(f"  [{posisi+1}/{len(urutan)}] frame {int(r['frame'])}: "
                      f"OCR='{r['ocr_text_raw']}' -> asli='{ketikan}'")
                posisi += 1
                break

        elif ord('0') <= key <= ord('9'):
            ketikan += chr(key)

cv2.destroyAllWindows()
simpan_progres(hasil)

print()
print("=" * 70)
print(f"SELESAI. {len(hasil)} sampel tersimpan di:")
print(f"  {OUT_CSV}")
print("=" * 70)