"""""

Skrip menghasilkan gambar bukti "multi-frame validation" versi FINAL
(tanpa filter zona finish) untuk dilampirkan di Bab 4.

Cara pakai:
1. Sesuaikan CSV_PATH, VIDEO_PATH, dan konfigurasi jika diperlukan.
2. Jalankan:
       python generate_bukti_tracking.py
3. Output:
       output_bukti/
       ├── frame_a.jpg
       ├── frame_b.jpg
       └── bukti_multiframe.jpg

Struktur CSV yang digunakan:
    frame          -> nomor frame video
    ocr_text_raw   -> teks mentah hasil OCR
    matched_bib    -> hasil pencocokan database (boleh kosong)
    confidence     -> confidence deteksi YOLOv8n
    area           -> luas bounding box
    x1, y1, x2, y2 -> koordinat bounding box

Catatan:
Koordinat x1, y1, x2, y2 pada CSV diasumsikan sudah berasal dari
frame yang di-resize ke lebar RESIZE_WIDTH = 960 px,
sesuai pipeline utama.
"""

import cv2
import pandas as pd
import os


# ============================================================
# KONFIGURASI
# ============================================================

CSV_PATH = r"D:\bib - detection\scripts\output\ocr_raw_log.csv"

VIDEO_PATH = r"D:\bib - detection\vidio\uji1.mp4"

OUTPUT_DIR = r"D:\bib - detection\scripts\output_bukti"

# Harus sama dengan pipeline utama
RESIZE_WIDTH = 960

# Syarat minimum multi-frame validation
MIN_FRAME_VALID = 2

# None = pilih otomatis berdasarkan OCR yang muncul >= 2 kali
# Contoh manual:
# TARGET_BIB = "6406"
TARGET_BIB = None

# Offset nomor frame.
#
# 0  = nomor frame CSV sama dengan indeks frame OpenCV
# -1 = nomor frame CSV dimulai dari 1
#
# Berdasarkan pengujian video kamu:
# frame 957 dapat dibaca langsung oleh OpenCV,
# sehingga digunakan 0.
FRAME_INDEX_OFFSET = 0


# ============================================================
# VALIDASI FILE
# ============================================================

def validasi_konfigurasi():
    """Memastikan file input tersedia."""

    if not os.path.isfile(CSV_PATH):
        raise FileNotFoundError(
            f"CSV tidak ditemukan:\n{CSV_PATH}"
        )

    if not os.path.isfile(VIDEO_PATH):
        raise FileNotFoundError(
            f"Video tidak ditemukan:\n{VIDEO_PATH}"
        )


# ============================================================
# PILIH BIB OTOMATIS
# ============================================================

def pilih_bib_otomatis(df):
    """
    Mencari teks OCR yang muncul minimal MIN_FRAME_VALID kali.

    Contoh:
        6406 muncul 10 kali
        10062 muncul 4 kali
        5232 muncul 2 kali

    Maka bib 6406 dipilih karena memiliki kemunculan terbanyak.
    """

    # Hilangkan nilai kosong
    teks = df["ocr_text_raw"].dropna().astype(str).str.strip()

    # Hilangkan string kosong
    teks = teks[teks != ""]

    counter = teks.value_counts()

    kandidat = counter[
        counter >= MIN_FRAME_VALID
    ].index.tolist()

    if not kandidat:
        raise ValueError(
            f"Tidak ditemukan OCR yang muncul >= "
            f"{MIN_FRAME_VALID} kali.\n"
            f"Cek isi CSV atau turunkan MIN_FRAME_VALID."
        )

    # Kandidat pertama adalah yang paling sering muncul
    return kandidat[0]


# ============================================================
# BACA FRAME VIDEO
# ============================================================

def ambil_frame(video_path, frame_no, box, label, out_path):
    """
    Mengambil frame tertentu dari video, melakukan resize,
    menggambar bounding box, kemudian menyimpan hasilnya.

    Pembacaan frame dibuat robust:
    1. Coba langsung menggunakan CAP_PROP_POS_FRAMES.
    2. Jika gagal, ulangi pembacaan dari frame 0 secara berurutan.
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(
            f"Gagal membuka video:\n{video_path}"
        )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = cap.get(cv2.CAP_PROP_FPS)

    # Konversi nomor frame CSV ke indeks OpenCV
    target_frame = int(frame_no) + FRAME_INDEX_OFFSET

    print(
        f"\nMembaca frame CSV {frame_no} "
        f"(indeks video {target_frame})..."
    )

    # Pastikan frame berada dalam batas video
    if target_frame < 0 or target_frame >= total_frames:
        cap.release()

        raise RuntimeError(
            f"Frame {frame_no} berada di luar batas video.\n"
            f"Total frame: {total_frames}\n"
            f"FPS: {fps}"
        )

    # --------------------------------------------------------
    # CARA 1: langsung menuju frame yang diminta
    # --------------------------------------------------------

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        target_frame
    )

    ret, frame = cap.read()

    # --------------------------------------------------------
    # CARA 2: fallback jika seeking gagal
    # --------------------------------------------------------

    if not ret or frame is None:

        print(
            f"Seeking langsung ke frame {target_frame} gagal."
        )

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            0
        )

        ret = False
        frame = None

        for current_frame in range(
            target_frame + 1
        ):
            ret, frame = cap.read()

            if not ret:
                break

        if ret and frame is not None:
            print(
                f"Frame {target_frame} berhasil "
                f"dibaca menggunakan fallback."
            )

    cap.release()

    # --------------------------------------------------------
    # Jika tetap gagal
    # --------------------------------------------------------

    if not ret or frame is None:
        raise RuntimeError(
            f"Gagal membaca frame {frame_no} "
            f"dari video.\n"
            f"Indeks video yang diminta: {target_frame}\n"
            f"Total frame video: {total_frames}"
        )

    # --------------------------------------------------------
    # RESIZE
    # --------------------------------------------------------

    h, w = frame.shape[:2]

    scale = RESIZE_WIDTH / float(w)

    new_width = RESIZE_WIDTH
    new_height = int(h * scale)

    frame = cv2.resize(
        frame,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )

    # --------------------------------------------------------
    # BOUNDING BOX
    # --------------------------------------------------------

    x1, y1, x2, y2 = map(
        int,
        box
    )

    # Pastikan koordinat tidak keluar dari ukuran gambar
    x1 = max(0, min(x1, new_width - 1))
    x2 = max(0, min(x2, new_width - 1))

    y1 = max(0, min(y1, new_height - 1))
    y2 = max(0, min(y2, new_height - 1))

    # Pastikan urutan koordinat benar
    if x1 > x2:
        x1, x2 = x2, x1

    if y1 > y2:
        y1, y2 = y2, y1

    # Bounding box
    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    # --------------------------------------------------------
    # LABEL
    # --------------------------------------------------------

    # Posisi label
    label_x = x1
    label_y = max(
        y1 - 10,
        25
    )

    cv2.putText(
        frame,
        label,
        (label_x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # SIMPAN GAMBAR
    # --------------------------------------------------------

    success = cv2.imwrite(
        out_path,
        frame
    )

    if not success:
        raise RuntimeError(
            f"Gagal menyimpan gambar:\n{out_path}"
        )

    print(
        f"Berhasil menyimpan: {out_path}"
    )

    return out_path


# ============================================================
# GABUNGKAN DUA FRAME
# ============================================================

def gabungkan_berdampingan(
    path_a,
    path_b,
    out_path
):
    """
    Menggabungkan frame A dan frame B
    secara berdampingan horizontal.
    """

    img_a = cv2.imread(path_a)
    img_b = cv2.imread(path_b)

    if img_a is None:
        raise RuntimeError(
            f"Gagal membaca gambar:\n{path_a}"
        )

    if img_b is None:
        raise RuntimeError(
            f"Gagal membaca gambar:\n{path_b}"
        )

    # Samakan tinggi gambar
    h = min(
        img_a.shape[0],
        img_b.shape[0]
    )

    width_a = int(
        img_a.shape[1]
        * h
        / img_a.shape[0]
    )

    width_b = int(
        img_b.shape[1]
        * h
        / img_b.shape[0]
    )

    img_a = cv2.resize(
        img_a,
        (width_a, h),
        interpolation=cv2.INTER_AREA
    )

    img_b = cv2.resize(
        img_b,
        (width_b, h),
        interpolation=cv2.INTER_AREA
    )

    # Gabungkan horizontal
    gabung = cv2.hconcat(
        [img_a, img_b]
    )

    success = cv2.imwrite(
        out_path,
        gabung
    )

    if not success:
        raise RuntimeError(
            f"Gagal menyimpan gambar gabungan:\n"
            f"{out_path}"
        )

    return out_path


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("GENERATE BUKTI MULTI-FRAME VALIDATION")
    print("=" * 60)

    # --------------------------------------------------------
    # VALIDASI KONFIGURASI
    # --------------------------------------------------------

    validasi_konfigurasi()

    # Buat folder output
    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # BACA CSV
    # --------------------------------------------------------

    print(
        f"\nMembaca CSV:\n{CSV_PATH}"
    )

    df = pd.read_csv(
        CSV_PATH
    )

    print(
        f"Jumlah baris CSV: {len(df)}"
    )

    # --------------------------------------------------------
    # VALIDASI KOLOM
    # --------------------------------------------------------

    kolom_wajib = [
        "frame",
        "ocr_text_raw",
        "matched_bib",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
    ]

    kolom_hilang = [
        kolom
        for kolom in kolom_wajib
        if kolom not in df.columns
    ]

    if kolom_hilang:
        raise ValueError(
            "Kolom CSV berikut tidak ditemukan:\n"
            + "\n".join(
                f"- {kolom}"
                for kolom in kolom_hilang
            )
            + "\n\nKolom CSV yang tersedia:\n"
            + "\n".join(
                f"- {kolom}"
                for kolom in df.columns
            )
        )

    # --------------------------------------------------------
    # PILIH BIB
    # --------------------------------------------------------

    if TARGET_BIB is None:
        bib = pilih_bib_otomatis(df)
    else:
        bib = str(TARGET_BIB).strip()

    print(
        f"\nBib terpilih: {bib}"
    )

    # --------------------------------------------------------
    # FILTER DATA BIB
    # --------------------------------------------------------

    df_ocr = df.copy()

    df_ocr["ocr_text_raw"] = (
        df_ocr["ocr_text_raw"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    subset = df_ocr[
        df_ocr["ocr_text_raw"] == str(bib)
    ].copy()

    # Urutkan berdasarkan nomor frame
    subset = subset.sort_values(
        "frame"
    )

    # --------------------------------------------------------
    # VALIDASI JUMLAH KEMUNCULAN
    # --------------------------------------------------------

    if len(subset) < MIN_FRAME_VALID:
        raise ValueError(
            f"Bib '{bib}' hanya muncul "
            f"{len(subset)} kali.\n"
            f"Minimum yang diperlukan: "
            f"{MIN_FRAME_VALID}"
        )

    # --------------------------------------------------------
    # TAMPILKAN LOG
    # --------------------------------------------------------

    print("\nLog OCR untuk bib terpilih:")
    print("-" * 60)

    kolom_tampil = [
        "frame",
        "ocr_text_raw",
        "matched_bib",
        "confidence",
    ]

    print(
        subset[kolom_tampil]
        .to_string(index=False)
    )

    print("-" * 60)

    # --------------------------------------------------------
    # PILIH DUA FRAME PERTAMA
    # --------------------------------------------------------

    baris_a = subset.iloc[0]
    baris_b = subset.iloc[1]

    frame_a = int(
        baris_a["frame"]
    )

    frame_b = int(
        baris_b["frame"]
    )

    print(
        f"\nFrame bukti A : {frame_a}"
    )

    print(
        f"Frame bukti B : {frame_b}"
    )

    print(
        f"Jarak frame   : {frame_b - frame_a} frame"
    )

    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    conf_a = float(
        baris_a["confidence"]
    )

    conf_b = float(
        baris_b["confidence"]
    )

    # --------------------------------------------------------
    # LABEL FRAME A
    # --------------------------------------------------------

    label_a = (
        f"OCR: {bib} | "
        f"frame {frame_a} | "
        f"ke-1 | "
        f"conf {conf_a:.3f}"
    )

    # --------------------------------------------------------
    # LABEL FRAME B
    # --------------------------------------------------------

    label_b = (
        f"OCR: {bib} | "
        f"frame {frame_b} | "
        f"ke-2 | "
        f"VALID | "
        f"conf {conf_b:.3f}"
    )

    # --------------------------------------------------------
    # PATH OUTPUT
    # --------------------------------------------------------

    path_a = os.path.join(
        OUTPUT_DIR,
        "frame_a.jpg"
    )

    path_b = os.path.join(
        OUTPUT_DIR,
        "frame_b.jpg"
    )

    # --------------------------------------------------------
    # GENERATE FRAME A
    # --------------------------------------------------------

    path_a = ambil_frame(
        VIDEO_PATH,
        frame_a,
        (
            baris_a["x1"],
            baris_a["y1"],
            baris_a["x2"],
            baris_a["y2"],
        ),
        label_a,
        path_a,
    )

    # --------------------------------------------------------
    # GENERATE FRAME B
    # --------------------------------------------------------

    path_b = ambil_frame(
        VIDEO_PATH,
        frame_b,
        (
            baris_b["x1"],
            baris_b["y1"],
            baris_b["x2"],
            baris_b["y2"],
        ),
        label_b,
        path_b,
    )

    # --------------------------------------------------------
    # GABUNGKAN
    # --------------------------------------------------------

    gabung_path = os.path.join(
        OUTPUT_DIR,
        "bukti_multiframe.jpg"
    )

    gabungkan_berdampingan(
        path_a,
        path_b,
        gabung_path
    )

    # --------------------------------------------------------
    # HASIL AKHIR
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("SELESAI")
    print("=" * 60)

    print(
        f"\nBib tervalidasi : {bib}"
    )

    print(
        f"Frame A        : {frame_a}"
    )

    print(
        f"Confidence A   : {conf_a:.4f}"
    )

    print(
        f"Frame B        : {frame_b}"
    )

    print(
        f"Confidence B   : {conf_b:.4f}"
    )

    print(
        f"\nFrame A tersimpan:"
        f"\n{path_a}"
    )

    print(
        f"\nFrame B tersimpan:"
        f"\n{path_b}"
    )

    print(
        f"\nBukti gabungan tersimpan:"
        f"\n{gabung_path}"
    )

    print(
        "\nGambar bukti ini dapat digunakan "
        "di Bab 4 pada bagian Tracking / "
        "Multi-Frame Validation."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()