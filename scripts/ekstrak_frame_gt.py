"""
ekstrak_frame_gt.py
====================
Skrip untuk mengambil frame tertentu dari video sebagai bukti visual
ground truth pada Bab 4.

Ada 2 MODE:

MODE 1 - "browse"  : simpan banyak frame ke folder supaya bisa dipilih manual
MODE 2 - "ambil"   : ambil frame spesifik berdasarkan nomor frame yang dipilih

CARA PAKAI:
  1. Jalankan MODE "browse" dulu untuk melihat isi video per frame
  2. Buka folder hasilnya, cari frame yang bagus (bib terbaca jelas / buram / dst)
  3. Catat nomor frame-nya
  4. Ganti MODE ke "ambil", isi FRAME_DIPILIH, jalankan lagi
"""

import cv2
import os

# ============================================================
# KONFIGURASI - SESUAIKAN INI
# ============================================================

VIDEO_PATH = r"D:\bib - detection\vidio\uji2.mp4"   # ganti sesuai video
OUTPUT_DIR = r"D:\bib - detection\output\frame_gt2"  # folder hasil

MODE = "ambil"   # "browse" atau "ambil"

# --- Pengaturan MODE browse ---
INTERVAL_BROWSE = 15   # simpan tiap 15 frame (makin kecil = makin banyak)

# --- Pengaturan MODE ambil ---
# Isi setelah Anda tahu nomor frame yang diinginkan.
# Format: "nomor_frame": "nama_file_output"
FRAME_DIPILIH = {
    69:  "gt2_mudah_1.jpg",
    348: "gt2_mudah_2.jpg",
    78:  "gt2_mudah_3.jpg",
    252: "gt2_mudah_4.jpg",
    243: "gt2_sulit_1.jpg",
    198: "gt2_sulit_2.jpg",
    300: "gt2_sulit_3.jpg",
    366: "gt2_sulit_4.jpg",
}

RESIZE_WIDTH = 960   # samakan dengan normalisasi di main.py


# ============================================================
# FUNGSI
# ============================================================

def normalisasi(frame):
    """Resize frame ke lebar RESIZE_WIDTH, sama seperti di main.py."""
    h, w = frame.shape[:2]
    scale = RESIZE_WIDTH / w
    return cv2.resize(frame, (RESIZE_WIDTH, int(h * scale)))


def mode_browse():
    """Simpan frame tiap INTERVAL_BROWSE supaya bisa dipilih manual."""
    folder = os.path.join(OUTPUT_DIR, "browse")
    os.makedirs(folder, exist_ok=True)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"ERROR: tidak bisa membuka video di {VIDEO_PATH}")
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video    : {VIDEO_PATH}")
    print(f"Total    : {total} frame")
    print(f"Interval : setiap {INTERVAL_BROWSE} frame")
    print(f"Output   : {folder}")
    print()

    count = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1

        if count % INTERVAL_BROWSE == 0:
            frame = normalisasi(frame)
            nama = os.path.join(folder, f"frame_{count:05d}.jpg")
            cv2.imwrite(nama, frame)
            saved += 1

    cap.release()
    print(f"Selesai. {saved} frame tersimpan.")
    print()
    print("LANGKAH BERIKUTNYA:")
    print(f"  1. Buka folder: {folder}")
    print("  2. Pilih frame yang bagus, lihat angka pada nama filenya")
    print("     (contoh: frame_00957.jpg -> nomor frame = 957)")
    print("  3. Isi FRAME_DIPILIH di skrip ini, lalu ganti MODE = 'ambil'")


def mode_ambil():
    """Ambil frame spesifik sesuai FRAME_DIPILIH."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"ERROR: tidak bisa membuka video di {VIDEO_PATH}")
        return

    target = dict(FRAME_DIPILIH)
    print(f"Video  : {VIDEO_PATH}")
    print(f"Target : {len(target)} frame")
    print(f"Output : {OUTPUT_DIR}")
    print()

    count = 0
    while target:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1

        if count in target:
            nama_file = target.pop(count)
            frame = normalisasi(frame)
            path = os.path.join(OUTPUT_DIR, nama_file)
            cv2.imwrite(path, frame)
            print(f"  [OK] frame {count:>5} -> {nama_file}")

    cap.release()

    if target:
        print()
        print("PERINGATAN: frame berikut tidak ditemukan (melebihi durasi video):")
        for f, n in target.items():
            print(f"  frame {f} ({n})")

    print()
    print(f"Selesai. Salin isi folder {OUTPUT_DIR} ke folder 'gambar/' project LaTeX Anda.")


if __name__ == "__main__":
    if MODE == "browse":
        mode_browse()
    elif MODE == "ambil":
        mode_ambil()
    else:
        print("ERROR: MODE harus 'browse' atau 'ambil'")