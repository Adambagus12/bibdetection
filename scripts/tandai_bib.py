import cv2
import os
import pandas as pd

# --- VIDEO 1 ---
LOG_V1 = r"D:\bib - detection\scripts\output\ocr_raw_log_uji1.csv"
DIR_V1 = r"D:\bib - detection\output\frame_gt"

# --- VIDEO 2 ---
LOG_V2 = r"D:\bib - detection\scripts\output\ocr_raw_log_uji2.csv"
DIR_V2 = r"D:\bib - detection\output\frame_gt2"

MAP_V1 = {
    "gt_mudah_1.jpg": 249,
    "gt_mudah_2.jpg": 33,
    "gt_mudah_3.jpg": 342,
    "gt_mudah_4.jpg": 336,
    "gt_sulit_1.jpg": 180,
    "gt_sulit_2.jpg": 384,
    "gt_sulit_3.jpg": 201,
    "gt_sulit_4.jpg": 297,
    "gt_beda_6049.jpg": 156,
    "gt_beda_6149.jpg": 132,
    "gt_beda_10373.jpg": 309,
}

MAP_V2 = {
    "gt2_mudah_1.jpg": 69,
    "gt2_mudah_2.jpg": 348,
    "gt2_mudah_3.jpg": 78,
    "gt2_mudah_4.jpg": 252,
    "gt2_sulit_1.jpg": 243,
    "gt2_sulit_2.jpg": 198,
    "gt2_sulit_3.jpg": 300,
    "gt2_sulit_4.jpg": 366,
}

WARNA_KOTAK = (0, 255, 255)
TEBAL_KOTAK = 3


def tandai(log_path, dir_gambar, mapping, label_video):
    if not os.path.isfile(log_path):
        print(f"[!] Log tidak ditemukan: {log_path}")
        print(f"    Lewati {label_video}.\n")
        return

    df = pd.read_csv(log_path)
    out_dir = os.path.join(dir_gambar, "bertanda")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print(f"MENANDAI GAMBAR - {label_video}")
    print("=" * 70)

    for nama_file, no_frame in mapping.items():
        path_gambar = os.path.join(dir_gambar, nama_file)
        if not os.path.isfile(path_gambar):
            print(f"  [SKIP] {nama_file} tidak ditemukan")
            continue

        img = cv2.imread(path_gambar)
        if img is None:
            print(f"  [SKIP] {nama_file} gagal dibaca")
            continue

        baris = df[df["frame"] == no_frame]
        if len(baris) == 0:
            print(f"  [SKIP] frame {no_frame} tidak ada di log")
            continue

        jumlah = 0
        for _, r in baris.iterrows():
            x1, y1, x2, y2 = int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"])
            cv2.rectangle(img, (x1, y1), (x2, y2), WARNA_KOTAK, TEBAL_KOTAK)

            teks_ocr = str(r["ocr_text_raw"]).replace(".0", "")
            matched = r["matched_bib"]
            if pd.notna(matched):
                label = f"Bib {str(matched).replace('.0','')}"
            else:
                label = f"OCR: {teks_ocr} (?)"

            y_teks = max(20, y1 - 8)
            cv2.putText(img, label, (x1, y_teks),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, WARNA_KOTAK, 2)
            jumlah += 1

        simpan = os.path.join(out_dir, nama_file)
        cv2.imwrite(simpan, img)
        print(f"  [OK] {nama_file:<22} frame {no_frame:<5} ({jumlah} kotak)")

    print()
    print(f"Hasil tersimpan di: {out_dir}\n")


if __name__ == "__main__":
    tandai(LOG_V1, DIR_V1, MAP_V1, "VIDEO 1")
    tandai(LOG_V2, DIR_V2, MAP_V2, "VIDEO 2")
    print("Selesai. Gunakan gambar dari folder 'bertanda' untuk laporan.")