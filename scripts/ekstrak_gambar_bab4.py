"""
ekstrak_gambar_bab4.py

Skrip bantu untuk mengambil dua jenis gambar tambahan yang dibutuhkan Bab 4:

1. GAMBAR BUKTI OCR
   Menampilkan citra crop bib hasil preprocessing berdampingan
   dengan teks hasil bacaan OCR untuk frame yang sudah didokumentasikan
   di Bab 4.

2. GAMBAR BUKTI FALSE NEGATIVE KATEGORI C
   Mengekstrak frame-frame spesifik yang disebut pada Tabel 4.x
   (Kategori C).

CARA PAKAI

1. Pastikan struktur proyek:

   D:\\bib - detection\\
   ├── modules\\
   ├── models\\
   ├── scripts\\
   ├── vidio\\
   └── SHARE RESULT BANUA RUN 2025.xlsx

2. Sesuaikan VIDEO_PATH, MODEL_PATH, dan target frame
   pada bagian KONFIGURASI.

3. Jalankan:

   python ekstrak_gambar_bab4.py

4. Gambar hasil akan tersimpan di:

   D:\\bib - detection\\output\\gambar_skripsi_bab4\\

Skrip ini TIDAK memproses seluruh video.
Skrip langsung melakukan seek ke frame yang diperlukan sehingga
lebih cepat dibandingkan menjalankan main.py secara penuh.
"""

import os
import sys
from pathlib import Path

import cv2


# ============================================================
# PATH SETUP
# ============================================================

# Lokasi file ekstrak_gambar_bab4.py
SCRIPT_DIR = Path(__file__).resolve().parent

# Root project:
# D:\bib - detection
PROJECT_ROOT = SCRIPT_DIR.parent

# Tambahkan root project ke Python path agar:
# from modules.detector import Detector
# dapat ditemukan.
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORT MODULE PROJECT
# ============================================================

from modules.detector import Detector
from modules.preprocess import preprocess_image
from modules.ocr import read_text
from modules.database import RunnerDatabase


# ============================================================
# KONFIGURASI
# ============================================================

# Video pengujian
VIDEO_PATH = PROJECT_ROOT / "vidio" / "uji1.mp4"

# Model YOLOv8
MODEL_PATH = PROJECT_ROOT / "models" / "best_v8n.pt"

# Database peserta
DB_PATH = PROJECT_ROOT / "SHARE RESULT BANUA RUN 2025.xlsx"

# Confidence YOLO
CONFIDENCE = 0.45

# Lebar frame yang digunakan.
# Disamakan dengan main.py.
RESIZE_WIDTH = 960

# Folder output
OUTPUT_DIR = PROJECT_ROOT / "output" / "gambar_bab4"

# Buat folder output jika belum ada
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# TARGET 1: BUKTI OCR
# ============================================================

TARGET_OCR = {
    "frame_number": 12,
    "bib_diharapkan": "10062",
}


# ============================================================
# TARGET 2: FALSE NEGATIVE KATEGORI C
# ============================================================

# Format:
# (
#     nomor_frame,
#     bib_target_ground_truth,
#     keterangan
# )

TARGET_FN_KATEGORI_C = [
    (
        57,
        "5193",
        'OCR sempat membaca 5196 (mirip, tetapi salah bib)'
    ),
    (
        60,
        "5187",
        'OCR sempat membaca 5198 (mirip, tetapi salah bib)'
    ),
    (
        132,
        "6429",
        'OCR sempat membaca 6149 (mirip, tetapi salah bib)'
    ),
]


# ============================================================
# FUNGSI AMBIL FRAME
# ============================================================

def ambil_frame(cap, frame_number):
    """
    Lompat langsung ke frame tertentu.

    Parameter:
        cap          : objek cv2.VideoCapture
        frame_number : nomor frame mulai dari 1

    Return:
        frame yang sudah di-resize,
        atau None jika frame gagal dibaca.
    """

    # OpenCV menggunakan indeks frame mulai dari 0.
    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        frame_number - 1
    )

    ret, frame = cap.read()

    if not ret or frame is None:
        return None

    # Ukuran asli frame
    h, w = frame.shape[:2]

    # Hitung skala berdasarkan lebar
    scale = RESIZE_WIDTH / w

    # Tinggi baru agar aspect ratio tetap
    new_height = int(h * scale)

    # Resize frame
    frame = cv2.resize(
        frame,
        (RESIZE_WIDTH, new_height)
    )

    return frame


# ============================================================
# FUNGSI PROSES SATU FRAME
# ============================================================

def proses_satu_frame(frame, detector, database):
    """
    Menjalankan:
        1. Deteksi YOLO
        2. Crop bounding box
        3. Resize crop
        4. Preprocessing
        5. OCR
        6. Pencocokan database

    Return:
        list hasil setiap bounding box.
    """

    # Jalankan YOLO
    boxes = detector.detect(frame)

    hasil = []

    # Proses setiap bounding box
    for (x1, y1, x2, y2, conf) in boxes:

        # Pastikan koordinat integer
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        # Crop bib
        crop = frame[y1:y2, x1:x2]

        # Validasi crop
        if crop is None or crop.size == 0:
            continue

        # Perbesar crop sebelum preprocessing
        crop_resized = cv2.resize(
            crop,
            None,
            fx=2,
            fy=2
        )

        # Preprocessing
        processed = preprocess_image(crop_resized)

        # OCR
        text = read_text(processed)

        # Nilai default pencocokan database
        matched_bib = None
        runner = None
        match_score = None
        second_score = None

        # Jika OCR menghasilkan teks,
        # lakukan pencocokan dengan database peserta.
        if text:
            (
                matched_bib,
                runner,
                match_score,
                second_score
            ) = database.get_runner(text)

        # Simpan hasil
        hasil.append({
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "conf": conf,

            "crop_asli": crop,
            "crop_processed": processed,

            "ocr_text": text,

            "matched_bib": matched_bib,
            "runner": runner,

            "match_score": match_score,
            "second_score": second_score,
        })

    return hasil


# ============================================================
# FUNGSI SIMPAN GAMBAR BUKTI OCR
# ============================================================

def simpan_gambar_ocr(cap, detector, database):
    """
    Mengambil gambar bukti OCR untuk satu frame contoh.
    """

    print()
    print("=" * 60)
    print("MENGAMBIL GAMBAR BUKTI OCR")
    print("=" * 60)

    frame_number = TARGET_OCR["frame_number"]
    bib_diharapkan = TARGET_OCR["bib_diharapkan"]

    # Ambil frame
    frame = ambil_frame(
        cap,
        frame_number
    )

    if frame is None:
        print(
            f"GAGAL: frame {frame_number} "
            f"tidak terbaca dari video."
        )
        return

    print(
        f"Frame target : {frame_number}"
    )

    print(
        f"Bib diharapkan : {bib_diharapkan}"
    )

    # Proses frame
    hasil = proses_satu_frame(
        frame,
        detector,
        database
    )

    if not hasil:
        print(
            f"PERINGATAN: tidak ada bib terdeteksi "
            f"pada frame {frame_number}."
        )
        return

    # Proses setiap bounding box
    for i, h in enumerate(hasil):

        # ----------------------------------------------------
        # SIMPAN CROP HASIL PREPROCESSING
        # ----------------------------------------------------

        nama_processed = (
            OUTPUT_DIR
            / f"ocr_bukti_frame{frame_number}"
              f"_bib{bib_diharapkan}"
              f"_box{i}_processed.jpg"
        )

        cv2.imwrite(
            str(nama_processed),
            h["crop_processed"]
        )

        # ----------------------------------------------------
        # BUAT GAMBAR BERLABEL OCR
        # ----------------------------------------------------

        processed = h["crop_processed"]

        # Pastikan gambar dapat dikonversi ke BGR.
        if len(processed.shape) == 2:
            processed_bgr = cv2.cvtColor(
                processed,
                cv2.COLOR_GRAY2BGR
            )
        else:
            processed_bgr = processed.copy()

        # Area tambahan untuk tulisan OCR
        pad = 50

        kanvas = cv2.copyMakeBorder(
            processed_bgr,
            0,
            pad,
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=(255, 255, 255)
        )

        # Teks OCR
        if h["ocr_text"]:
            label = f'OCR: "{h["ocr_text"]}"'
        else:
            label = "OCR: (kosong)"

        # Tulis label OCR
        cv2.putText(
            kanvas,
            label,
            (10, kanvas.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )

        # Nama file
        nama_berlabel = (
            OUTPUT_DIR
            / f"ocr_bukti_frame{frame_number}"
              f"_bib{bib_diharapkan}"
              f"_box{i}_berlabel.jpg"
        )

        # Simpan
        cv2.imwrite(
            str(nama_berlabel),
            kanvas
        )

        # ----------------------------------------------------
        # INFORMASI KE TERMINAL
        # ----------------------------------------------------

        print()
        print(f"Box              : {i}")
        print(f"Confidence YOLO  : {h['conf']:.4f}")
        print(f"OCR              : {h['ocr_text']}")
        print(f"Matched Bib      : {h['matched_bib']}")
        print(f"Match Score      : {h['match_score']}")
        print(f"Second Score     : {h['second_score']}")
        print(f"File             : {nama_berlabel}")

    print()
    print(
        f"Selesai mengambil bukti OCR."
    )

    print(
        f"Folder output: {OUTPUT_DIR}"
    )


# ============================================================
# FUNGSI SIMPAN FALSE NEGATIVE KATEGORI C
# ============================================================

def simpan_gambar_fn_kategori_c(
    cap,
    detector,
    database
):
    """
    Mengambil gambar bukti tiap kasus False Negative
    Kategori C.
    """

    print()
    print("=" * 60)
    print("MENGAMBIL GAMBAR BUKTI FALSE NEGATIVE KATEGORI C")
    print("=" * 60)

    # Proses seluruh target
    for (
        frame_number,
        bib_target,
        keterangan
    ) in TARGET_FN_KATEGORI_C:

        print()
        print("-" * 60)
        print(
            f"Frame {frame_number} | "
            f"Bib target {bib_target}"
        )
        print(
            f"Keterangan: {keterangan}"
        )

        # Ambil frame
        frame = ambil_frame(
            cap,
            frame_number
        )

        if frame is None:
            print(
                f"GAGAL: frame {frame_number} "
                f"(bib target {bib_target}) "
                f"tidak terbaca."
            )
            continue

        # Jalankan deteksi + OCR
        hasil = proses_satu_frame(
            frame,
            detector,
            database
        )

        if not hasil:
            print(
                f"PERINGATAN: tidak ada bib terdeteksi "
                f"pada frame {frame_number} "
                f"(bib target {bib_target})."
            )
            continue

        # Proses semua bounding box
        for i, h in enumerate(hasil):

            # ------------------------------------------------
            # FRAME PENUH DENGAN BOUNDING BOX
            # ------------------------------------------------

            frame_anotasi = frame.copy()

            # Bounding box
            cv2.rectangle(
                frame_anotasi,
                (h["x1"], h["y1"]),
                (h["x2"], h["y2"]),
                (0, 255, 255),
                2
            )

            # Label OCR
            if h["ocr_text"]:
                label = f'OCR: "{h["ocr_text"]}"'
            else:
                label = "OCR: (kosong)"

            # Tambahkan label di atas bounding box
            cv2.putText(
                frame_anotasi,
                label,
                (
                    h["x1"],
                    max(25, h["y1"] - 10)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

            # ------------------------------------------------
            # NAMA FILE FRAME
            # ------------------------------------------------

            nama_frame = (
                OUTPUT_DIR
                / f"fn_kategoriC"
                  f"_bib{bib_target}"
                  f"_frame{frame_number}"
                  f"_box{i}_frame.jpg"
            )

            # Simpan frame
            cv2.imwrite(
                str(nama_frame),
                frame_anotasi
            )

            # ------------------------------------------------
            # SIMPAN CROP PREPROCESSING
            # ------------------------------------------------

            nama_crop = (
                OUTPUT_DIR
                / f"fn_kategoriC"
                  f"_bib{bib_target}"
                  f"_frame{frame_number}"
                  f"_box{i}_crop.jpg"
            )

            cv2.imwrite(
                str(nama_crop),
                h["crop_processed"]
            )

            # ------------------------------------------------
            # INFORMASI KE TERMINAL
            # ------------------------------------------------

            print()
            print(
                f"Box              : {i}"
            )
            print(
                f"Confidence YOLO  : {h['conf']:.4f}"
            )
            print(
                f"OCR              : {h['ocr_text']}"
            )
            print(
                f"Matched Bib      : {h['matched_bib']}"
            )
            print(
                f"Match Score      : {h['match_score']}"
            )
            print(
                f"Second Score     : {h['second_score']}"
            )
            print(
                f"Ground Truth     : {bib_target}"
            )
            print(
                f"Keterangan       : {keterangan}"
            )
            print(
                f"Frame disimpan   : {nama_frame}"
            )
            print(
                f"Crop disimpan    : {nama_crop}"
            )

    print()
    print(
        "Selesai mengambil gambar "
        "False Negative Kategori C."
    )

    print(
        f"Folder output: {OUTPUT_DIR}"
    )


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print()
    print("=" * 60)
    print("EKSTRAK GAMBAR BAB 4")
    print("=" * 60)

    # --------------------------------------------------------
    # INFORMASI PATH
    # --------------------------------------------------------

    print()
    print("Project root:")
    print(PROJECT_ROOT)

    print()
    print("Video:")
    print(VIDEO_PATH)

    print()
    print("Model:")
    print(MODEL_PATH)

    print()
    print("Database:")
    print(DB_PATH)

    print()
    print("Output:")
    print(OUTPUT_DIR)

    # --------------------------------------------------------
    # CEK FILE
    # --------------------------------------------------------

    print()
    print("Memeriksa file...")

    if not VIDEO_PATH.exists():
        print()
        print("ERROR: file video tidak ditemukan!")
        print(VIDEO_PATH)
        return

    print(
        f"[OK] Video ditemukan: {VIDEO_PATH}"
    )

    if not MODEL_PATH.exists():
        print()
        print("ERROR: file model tidak ditemukan!")
        print(MODEL_PATH)
        return

    print(
        f"[OK] Model ditemukan: {MODEL_PATH}"
    )

    if not DB_PATH.exists():
        print()
        print("ERROR: file database tidak ditemukan!")
        print(DB_PATH)
        return

    print(
        f"[OK] Database ditemukan: {DB_PATH}"
    )

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("LOADING MODEL")
    print("=" * 60)

    try:
        detector = Detector(
            str(MODEL_PATH),
            CONFIDENCE
        )
    except Exception as e:
        print()
        print("ERROR saat loading model:")
        print(e)
        return

    print()
    print("Model berhasil dimuat.")

    # --------------------------------------------------------
    # LOAD DATABASE
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("LOADING DATABASE")
    print("=" * 60)

    try:
        database = RunnerDatabase(
            str(DB_PATH)
        )
    except Exception as e:
        print()
        print("ERROR saat loading database:")
        print(e)
        return

    print()
    print("Database berhasil dimuat.")

    # --------------------------------------------------------
    # BUKA VIDEO
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("MEMBUKA VIDEO")
    print("=" * 60)

    cap = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not cap.isOpened():
        print()
        print(
            f"ERROR: tidak bisa membuka video:"
        )
        print(VIDEO_PATH)
        return

    # Informasi video
    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    print()
    print(
        f"Resolusi video : {width} x {height}"
    )

    print(
        f"FPS video      : {fps:.2f}"
    )

    print(
        f"Total frame    : {total_frames}"
    )

    # --------------------------------------------------------
    # PROSES BUKTI OCR
    # --------------------------------------------------------

    simpan_gambar_ocr(
        cap,
        detector,
        database
    )

    # --------------------------------------------------------
    # PROSES FALSE NEGATIVE KATEGORI C
    # --------------------------------------------------------

    simpan_gambar_fn_kategori_c(
        cap,
        detector,
        database
    )

    # --------------------------------------------------------
    # TUTUP VIDEO
    # --------------------------------------------------------

    cap.release()

    print()
    print("=" * 60)
    print("SELESAI SEMUA")
    print("=" * 60)

    print()
    print(
        "Periksa hasil gambar di:"
    )

    print(
        OUTPUT_DIR
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()