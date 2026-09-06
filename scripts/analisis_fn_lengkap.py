"""
Skrip analisis lengkap penyebab False Negative (FN) untuk Bab 4.
Membaca ocr_raw_log.csv dan mengklasifikasikan SETIAP bib ground truth
ke salah satu kategori berdasarkan jejaknya di log:

  KATEGORI 1 - TERKONFIRMASI DI OUTPUT AKHIR (TP)
      Bib ini berhasil match ke database (matched_bib) pada >=1 frame
      DAN kemungkinan besar lolos ke output akhir. (Cross-check manual
      dengan tabel hasil akhir tetap diperlukan untuk memastikan TP vs FN,
      skrip ini hanya menandai "match_bib ditemukan di log".)

  KATEGORI 2 - MATCH DI LOG TAPI TIDAK SAMPAI OUTPUT AKHIR (bukan gagal YOLO)
      Bib muncul sebagai matched_bib di >=1 baris log, artinya deteksi YOLO
      + OCR + database matching SEMPAT berhasil pada level per-frame, tapi
      gagal lolos multi-frame validation / best scoring / bug lain.

  KATEGORI 3 - OCR MEMBACA TEKS MIRIP TAPI TIDAK PERNAH MATCH KE DATABASE
      Bib TIDAK PERNAH muncul sebagai matched_bib, TAPI ada teks OCR mentah
      (ocr_text_raw) yang skor kemiripannya tinggi (>=85) terhadap bib ini.
      Artinya YOLO+OCR berhasil "melihat" sesuatu yang dekat, tapi gagal di
      tahap pencocokan database (skor <80 atau ditolak ambiguitas).

  KATEGORI 4 - TIDAK ADA JEJAK SAMA SEKALI (kandidat kuat gagal deteksi YOLO)
      Tidak ada satupun ocr_text_raw di seluruh log yang mirip (skor <70)
      dengan bib ini. Kandidat paling kuat untuk "YOLO tidak pernah
      mendeteksi kotaknya".

CARA PAKAI:
1. Pastikan GROUND_TRUTH_V1 (44 bib) dan LOG_PATH sudah benar
2. Jalankan: python analisis_fn_lengkap.py
3. Salin tabel hasil ke Bab 4, SILANGKAN dengan tabel rincian_v1 (hasil akhir)
   untuk memisahkan yang benar-benar FN vs yang sudah TP
"""

import csv
from rapidfuzz import fuzz

# ======== SESUAIKAN INI ========
LOG_PATH = r"D:\bib - detection\scripts\output\ocr_raw_log.csv"

GROUND_TRUTH_V1 = [
    "5377", "5231", "10062", "5198", "5059", "6377", "6070", "5934",
    "5480", "5473", "10426", "6319", "12044", "10021", "5244", "10610",
    "5271", "5187", "10573", "6085", "6086", "5260", "5257", "10271",
    "5990", "6156", "10237", "5843", "5845", "10386", "5569", "5567",
    "6462", "10499", "7141", "10483", "6406", "6429", "5708", "5535",
    "10023", "6155", "6154", "5193",
]
# =================================


def baca_log(log_path):
    """Baca seluruh baris log, kembalikan sebagai list of dict."""
    rows = []
    with open(log_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def klasifikasi_bib(bib, rows):
    """
    Klasifikasikan satu bib ke salah satu dari 4 kategori berdasarkan
    jejaknya di log mentah.
    """
    # Cek apakah bib ini pernah muncul sebagai matched_bib (match ke database)
    baris_matched = [r for r in rows if r.get("matched_bib", "").strip() == bib]

    if baris_matched:
        frames = [r.get("frame", "?") for r in baris_matched]
        confs = [r.get("confidence", "?") for r in baris_matched]
        return {
            "bib": bib,
            "kategori": 2,  # default: match di log; kategori 1 (TP) dicek manual di luar skrip
            "keterangan": f"Match ke database pada {len(baris_matched)} baris log "
                           f"(frame: {', '.join(frames[:5])}{'...' if len(frames) > 5 else ''})",
            "skor_terbaik": 100,
            "teks_terdekat": bib,
        }

    # Tidak pernah match ke database. Cek kemiripan dengan semua ocr_text_raw
    skor_terbaik = 0
    baris_terdekat = None
    for r in rows:
        teks = r.get("ocr_text_raw", "").strip()
        if not teks:
            continue
        skor = fuzz.ratio(bib, teks)
        if skor > skor_terbaik:
            skor_terbaik = skor
            baris_terdekat = r

    if skor_terbaik >= 85:
        kategori = 3
        keterangan = (
            f"Tidak pernah match database, tapi OCR sempat baca '{baris_terdekat.get('ocr_text_raw')}' "
            f"(skor kemiripan {skor_terbaik:.1f}) pada frame {baris_terdekat.get('frame')}, "
            f"match_score={baris_terdekat.get('match_score', '-')}, "
            f"second_score={baris_terdekat.get('second_score', '-')}, "
            f"confidence={baris_terdekat.get('confidence', '-')}"
        )
    elif skor_terbaik >= 70:
        kategori = 3
        keterangan = (
            f"Kemiripan sedang: OCR pernah baca '{baris_terdekat.get('ocr_text_raw')}' "
            f"(skor {skor_terbaik:.1f}) pada frame {baris_terdekat.get('frame')}, "
            f"confidence={baris_terdekat.get('confidence', '-')} -- perlu ditinjau manual, "
            f"kemungkinan bukan bib ini yang terbaca"
        )
    else:
        kategori = 4
        teks_info = f"'{baris_terdekat.get('ocr_text_raw')}'" if baris_terdekat else "(tidak ada)"
        keterangan = (
            f"TIDAK ADA jejak pembacaan yang mirip di seluruh log "
            f"(skor tertinggi hanya {skor_terbaik:.1f} terhadap {teks_info}) "
            f"-- kandidat kuat gagal deteksi YOLO"
        )

    return {
        "bib": bib,
        "kategori": kategori,
        "keterangan": keterangan,
        "skor_terbaik": skor_terbaik,
        "teks_terdekat": baris_terdekat.get("ocr_text_raw") if baris_terdekat else None,
    }


def main():
    rows = baca_log(LOG_PATH)
    print(f"Total baris di log: {len(rows)}")
    print(f"Total bib ground truth yang dicek: {len(GROUND_TRUTH_V1)}")
    print()

    hasil = [klasifikasi_bib(bib, rows) for bib in GROUND_TRUTH_V1]

    kat2 = [h for h in hasil if h["kategori"] == 2]
    kat3 = [h for h in hasil if h["kategori"] == 3]
    kat4 = [h for h in hasil if h["kategori"] == 4]

    print("=" * 70)
    print(f"KATEGORI 2 -- Match di log (perlu dicek manual: TP atau gagal validasi) [{len(kat2)} bib]")
    print("=" * 70)
    for h in kat2:
        print(f"  Bib {h['bib']}: {h['keterangan']}")

    print()
    print("=" * 70)
    print(f"KATEGORI 3 -- OCR pernah baca mirip, TAPI TIDAK PERNAH match database [{len(kat3)} bib]")
    print("=" * 70)
    for h in kat3:
        print(f"  Bib {h['bib']}: {h['keterangan']}")

    print()
    print("=" * 70)
    print(f"KATEGORI 4 -- TIDAK ADA JEJAK SAMA SEKALI (kandidat kuat gagal deteksi YOLO) [{len(kat4)} bib]")
    print("=" * 70)
    for h in kat4:
        print(f"  Bib {h['bib']}: {h['keterangan']}")

    print()
    print("=" * 70)
    print("LANGKAH SELANJUTNYA:")
    print("1. Bib di KATEGORI 2 -- silangkan dengan Tabel rincian_v1 (hasil akhir).")
    print("   Kalau ADA di tabel itu -> memang TP, abaikan dari analisis FN.")
    print("   Kalau TIDAK ADA -> FN yang gagal di tahap downstream (bukan salah YOLO).")
    print("2. Bib di KATEGORI 3 -- FN akibat gagal database matching (OCR + YOLO OK).")
    print("3. Bib di KATEGORI 4 -- kandidat kuat FN akibat gagal deteksi YOLO,")
    print("   cari framenya di video untuk bukti visual di Bab 4.")


if __name__ == "__main__":
    main()