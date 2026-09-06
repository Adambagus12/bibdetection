"""
uji_pencocokan_awalan.py
========================
Menguji strategi pencocokan berbasis awalan untuk menangani perbedaan
format nomor bib antara bib fisik (2062.1) dan basis data (206201).

Logika: jika teks OCR 5 digit tidak ditemukan secara persis, cari bib
6 digit yang diawali 4 digit pertama dan diakhiri digit terakhir yang
sama.

Skrip ini hanya menyimulasikan, tidak mengubah sistem.
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.database import RunnerDatabase

DB_PATH = r"D:\bib - detection\SHARE REKAP GREEN FORCE RUN 2026.xlsx"

# Teks OCR yang tercatat pada log video UJI 1
OCR_LOG = [
    "4974", "4444", "95444", "4456", "2021", "70611", "70621", "20621",
    "41822", "4622", "8622", "20621", "20622", "50622", "10622", "4400",
    "44002", "4092", "14705", "5669", "40122", "19122", "10182", "10182",
    "1016", "39106", "3916", "16039",
]

# Nomor bib sebenarnya di video (hasil pengamatan manual)
GROUND_TRUTH = [
    "10655", "2062.1", "2062.3", "10307", "14072", "10305", "10132",
    "10223", "10213", "10867", "10133", "10720", "23018", "2034.1", "2034.2",
]


def ke_format_database(bib_fisik):
    """2062.1 -> 206201 (titik diganti dua digit)"""
    if "." not in bib_fisik:
        return bib_fisik
    depan, belakang = bib_fisik.split(".", 1)
    return depan + belakang.zfill(2)


def cocok_awalan(teks, daftar_bib):
    """
    Cari bib 6 digit yang diawali 4 digit pertama teks dan diakhiri
    digit terakhir teks. Contoh: '20621' -> '206201'
    """
    if len(teks) != 5 or not teks.isdigit():
        return None
    awalan = teks[:4]
    akhiran = teks[4]
    kandidat = [b for b in daftar_bib
                if len(b) == 6 and b.startswith(awalan) and b.endswith(akhiran)]
    return kandidat[0] if len(kandidat) == 1 else None


db = RunnerDatabase(DB_PATH)
daftar = list(db.db.keys())

gt_db = {ke_format_database(b) for b in GROUND_TRUTH}

print("=" * 78)
print("UJI STRATEGI PENCOCOKAN BERBASIS AWALAN")
print("=" * 78)
print(f"Bib dalam basis data : {len(daftar)}")
print(f"Ground truth         : {len(GROUND_TRUTH)} bib")
print(f"  format database    : {', '.join(sorted(gt_db))}")
print()

hasil = []
for teks in OCR_LOG:
    exact = teks if teks in db.db else None
    awalan = cocok_awalan(teks, daftar) if not exact else None
    final = exact or awalan
    hasil.append({
        "ocr": teks,
        "exact": exact,
        "awalan": awalan,
        "final": final,
        "benar": final in gt_db if final else None,
    })

print("-" * 78)
print(f"{'OCR':<10}{'Exact':<12}{'Via awalan':<14}{'Hasil akhir':<14}{'Benar?'}")
print("-" * 78)
for h in hasil:
    if h["final"]:
        status = "YA" if h["benar"] else "tidak"
        print(f"{h['ocr']:<10}{h['exact'] or '-':<12}{h['awalan'] or '-':<14}"
              f"{h['final']:<14}{status}")
print()

ada_hasil = [h for h in hasil if h["final"]]
lewat_awalan = [h for h in hasil if h["awalan"]]
benar = [h for h in ada_hasil if h["benar"]]

print("=" * 78)
print("RINGKASAN")
print("=" * 78)
print(f"Total pembacaan OCR        : {len(OCR_LOG)}")
print(f"Berhasil dicocokkan        : {len(ada_hasil)}")
print(f"  - lewat exact match      : {len([h for h in ada_hasil if h['exact']])}")
print(f"  - lewat pencocokan awalan: {len(lewat_awalan)}")
print(f"Identitas benar            : {len(benar)}")
print()

if lewat_awalan:
    print("Yang tertangani strategi awalan:")
    for h in lewat_awalan:
        tanda = "BENAR" if h["benar"] else "salah"
        print(f"  OCR '{h['ocr']}' -> {h['awalan']}  ({tanda})")
    print()

bib_unik = {h["final"] for h in benar}
print(f"Bib unik teridentifikasi benar: {len(bib_unik)} dari {len(gt_db)}")
if bib_unik:
    print(f"  {', '.join(sorted(bib_unik))}")