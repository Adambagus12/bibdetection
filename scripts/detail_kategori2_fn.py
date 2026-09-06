"""
Skrip lanjutan: menampilkan detail teks OCR mentah untuk 10 bib yang
match ke database di level frame TAPI gagal jadi TP di output akhir.
Tujuannya: memastikan apakah kegagalan mereka disebabkan oleh teks OCR
yang TIDAK IDENTIK antar frame (gagal exact-match multi-frame validation)
meski secara fuzzy tetap cocok ke bib yang sama.
"""

import csv

LOG_PATH = r"D:\bib - detection\scripts\output\ocr_raw_log.csv"

# 10 bib yang match di log tapi tidak sampai TP (hasil analisis sebelumnya)
BIB_PERLU_DICEK = ["5198", "12044", "5244", "5271", "6085", "10271",
                   "5990", "10483", "5708", "6155"]


def main():
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for bib in BIB_PERLU_DICEK:
        baris_cocok = [r for r in rows if r.get("matched_bib", "").strip() == bib]
        teks_unik = set(r.get("ocr_text_raw", "").strip() for r in baris_cocok)

        print(f"--- Bib {bib} ({len(baris_cocok)} kali match) ---")
        for r in baris_cocok:
            print(f"    frame={r.get('frame')}, ocr_text_raw='{r.get('ocr_text_raw')}', "
                  f"match_score={r.get('match_score')}, confidence={r.get('confidence')}")
        if len(teks_unik) == 1:
            print(f"    => SEMUA teks identik ('{list(teks_unik)[0]}') tapi tetap gagal jadi TP. "
                  f"Kemungkinan kalah di best-detection-scoring atau tersaring sebab lain.")
        else:
            print(f"    => Teks BERVARIASI antar frame ({len(teks_unik)} versi berbeda): {teks_unik}. "
                  f"Kemungkinan gagal exact-match multi-frame validation.")
        print()


if __name__ == "__main__":
    main()