"""
Skrip untuk mencari kandidat kasus bib yang TIDAK PERNAH terdeteksi
oleh YOLOv8n sama sekali (bukan sekadar gagal validasi/OCR).

Membandingkan:
1. Daftar ground truth manual (bib yang benar-benar melintas kamera)
2. Daftar SEMUA nomor yang pernah muncul di log OCR mentah (ocr_raw_log.csv)
   -- termasuk yang salah baca sekalipun

Bib yang ada di (1) tapi TIDAK ADA SAMA SEKALI di (2) -- meski dengan
toleransi kemiripan -- adalah kandidat kuat "gagal terdeteksi YOLO",
karena kalau YOLO berhasil mendeteksi box-nya, EasyOCR pasti akan mencoba
membaca sesuatu dari box itu (walau hasilnya salah), sehingga akan
tercatat di log mentah.

CARA PAKAI:
1. Sesuaikan GROUND_TRUTH_V1 dan GROUND_TRUTH_V2 dengan daftar bib manual kamu
2. Sesuaikan LOG_PATH ke lokasi ocr_raw_log.csv kamu
3. Jalankan: python cari_kasus_fn.py
"""

import csv
from rapidfuzz import fuzz

# ======== ISI DENGAN DAFTAR GROUND TRUTH MANUAL KAMU ========
# Daftar nomor bib yang kamu catat manual saat menonton video (44 untuk Video 1, 21 untuk Video 2)
GROUND_TRUTH_V1 = [
    "5377", "5231", "10062", "5198", "5059", "6377", "6070", "5934",
    "5480", "5473", "10426", "6319", "12044", "10021", "5244", "10610",
    "5271", "5187", "10573", "6085", "6086", "5260", "5257", "10271",
    "5990", "6156", "10237", "5843", "5845", "10386", "5569", "5567",
    "6462", "10499", "7141", "10483", "6406", "6429", "5708", "5535",
    "10023", "6155", "6154", "5193",
]

LOG_PATH_V1 = r"D:\bib - detection\scripts\output\ocr_raw_log.csv"
KOLOM_TEKS_OCR = "ocr_text_raw"
# ================================================================


def baca_semua_teks_ocr(log_path, kolom):
    """Baca semua teks yang pernah dihasilkan OCR sepanjang video (termasuk yang salah)."""
    semua_teks = set()
    with open(log_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            teks = row.get(kolom, "").strip()
            if teks:
                semua_teks.add(teks)
    return semua_teks


def cari_kandidat_gagal_deteksi(ground_truth, semua_teks_ocr, ambang_mirip=70):
    """
    Untuk tiap bib ground truth, cek apakah PERNAH muncul di log OCR
    (exact match ATAU mirip, untuk mengakomodasi salah baca 1-2 digit).
    Jika skor kemiripan tertinggi terhadap SEMUA teks OCR < ambang_mirip,
    berarti bib ini kemungkinan besar tidak pernah terdeteksi YOLO sama sekali.
    """
    kandidat_gagal = []
    for bib in ground_truth:
        skor_terbaik = 0
        teks_terdekat = None
        for teks_ocr in semua_teks_ocr:
            skor = fuzz.ratio(bib, teks_ocr)
            if skor > skor_terbaik:
                skor_terbaik = skor
                teks_terdekat = teks_ocr

        if skor_terbaik < ambang_mirip:
            kandidat_gagal.append({
                "bib": bib,
                "skor_terbaik": skor_terbaik,
                "teks_ocr_terdekat": teks_terdekat,
            })
    return kandidat_gagal


def main():
    semua_teks = baca_semua_teks_ocr(LOG_PATH_V1, KOLOM_TEKS_OCR)
    print(f"Total teks unik pernah muncul di log OCR: {len(semua_teks)}")
    print()

    kandidat = cari_kandidat_gagal_deteksi(GROUND_TRUTH_V1, semua_teks)

    print(f"Ditemukan {len(kandidat)} kandidat bib yang TIDAK PERNAH")
    print("terdeteksi YOLO (tidak ada bacaan OCR yang mirip sama sekali):")
    print()
    for k in kandidat:
        print(f"  Bib {k['bib']}: skor kemiripan tertinggi hanya {k['skor_terbaik']} "
              f"(teks OCR terdekat: '{k['teks_ocr_terdekat']}')")

    print()
    print("Langkah selanjutnya untuk tiap kandidat di atas:")
    print("1. Cari di video, kira-kira di detik/frame berapa pelari bib ini melintas")
    print("2. Screenshot frame itu, lihat kondisinya (kecil/blur/oklusi/sudut miring)")
    print("3. Itu jadi bukti visual untuk Bab 4")


if __name__ == "__main__":
    main()