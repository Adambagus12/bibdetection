"""
uji_ambang_kemunculan.py
========================
Menyimulasikan berbagai ambang minimal kemunculan pada multi-frame
validation, untuk melihat pengaruhnya terhadap precision dan recall.

Perhitungan dilakukan dari log OCR yang sudah ada, tanpa memproses
ulang video.
"""

import pandas as pd
from collections import Counter

# ============================================================
LOG_PATH = r"D:\bib - detection\scripts\output\ocr_raw_log_uji1.csv"

# Ground truth Video 1 - varian mayoritas (dicatat >= 2 dari 3 pengamat)
GROUND_TRUTH = [
    "5377","5231","10062","5198","5059","6377","6070","5934","5480","5473",
    "10426","6319","12044","10021","5244","10610","5271","5187","10573","6085",
    "5260","5257","10271","5990","6156","10237","5843","5845","10386","5569",
    "5567","6462","10499","7141","10483","6406","6429","5708","5535","10023",
    "6155","6154","5193","6049","6149","5280","10373",
]

AMBANG_UJI = [2, 3, 4, 5]


def bersih(x):
    s = str(x).strip()
    return "" if s in ("nan", "-", "") else s.replace(".0", "")


df = pd.read_csv(LOG_PATH)
gt = set(GROUND_TRUTH)

# Hanya baris yang berhasil dicocokkan ke database
cocok = df[df["matched_bib"].notna()].copy()
cocok["bib"] = cocok["matched_bib"].apply(bersih)
cocok["ocr"] = cocok["ocr_text_raw"].apply(bersih)
cocok = cocok[cocok["bib"] != ""]

# Pisahkan jalur exact match (skor 100) dan fuzzy
cocok["jalur"] = cocok["match_score"].apply(
    lambda s: "exact" if float(s) >= 100 else "fuzzy")

print("=" * 78)
print("SIMULASI AMBANG MINIMAL KEMUNCULAN (MULTI-FRAME VALIDATION)")
print("=" * 78)
print(f"Total baris log            : {len(df)}")
print(f"Berhasil cocok ke database : {len(cocok)}")
print(f"  - via exact match        : {(cocok['jalur']=='exact').sum()}")
print(f"  - via fuzzy match        : {(cocok['jalur']=='fuzzy').sum()}")
print(f"Ground truth               : {len(gt)} bib")
print()

# Hitung kemunculan teks OCR identik per bib
# (multi-frame validation di sistem menghitung berdasarkan teks OCR)
jumlah_teks = Counter(cocok["ocr"])

# Petakan tiap bib ke jumlah kemunculan teks OCR terbanyak yang mengarah ke bib itu
bib_kemunculan = {}
for bib, grup in cocok.groupby("bib"):
    maks = max(jumlah_teks[t] for t in grup["ocr"].unique())
    bib_kemunculan[bib] = maks

hasil = []
for ambang in AMBANG_UJI:
    lolos = {b for b, n in bib_kemunculan.items() if n >= ambang}
    tp = lolos & gt
    fp = lolos - gt
    p = len(tp) / len(lolos) * 100 if lolos else 0
    r = len(tp) / len(gt) * 100
    f = 2 * p * r / (p + r) if (p + r) else 0
    hasil.append({
        "ambang": ambang, "total": len(lolos), "tp": len(tp), "fp": len(fp),
        "p": p, "r": r, "f": f, "set_lolos": lolos, "set_fp": fp
    })

print(f"{'Ambang':<12}{'Total':<9}{'TP':<6}{'FP':<6}"
      f"{'Precision':<13}{'Recall':<12}{'F1'}")
print("-" * 78)
for h in hasil:
    tanda = "  <- sekarang" if h["ambang"] == 2 else ""
    print(f"{h['ambang']:<12}{h['total']:<9}{h['tp']:<6}{h['fp']:<6}"
          f"{h['p']:<12.1f}%{h['r']:<11.1f}%{h['f']:.1f}%{tanda}")
print()

terbaik_f1 = max(hasil, key=lambda h: h["f"])
terbaik_p = max(hasil, key=lambda h: h["p"])
print(f"F1 tertinggi        : ambang {terbaik_f1['ambang']} ({terbaik_f1['f']:.1f}%)")
print(f"Precision tertinggi : ambang {terbaik_p['ambang']} ({terbaik_p['p']:.1f}%)")
print()

dasar = hasil[0]
print("-" * 78)
print("PERUBAHAN DIBANDING AMBANG 2 (KONFIGURASI SEKARANG)")
print("-" * 78)
for h in hasil[1:]:
    hilang_benar = (dasar["set_lolos"] & gt) - h["set_lolos"]
    hilang_salah = dasar["set_fp"] - h["set_fp"]
    print(f"Ambang {h['ambang']}:")
    print(f"   Precision {h['p']-dasar['p']:+.1f} poin   "
          f"Recall {h['r']-dasar['r']:+.1f} poin   F1 {h['f']-dasar['f']:+.1f} poin")
    print(f"   FP tersaring ({len(hilang_salah)}): "
          f"{', '.join(sorted(hilang_salah)) or '-'}")
    print(f"   TP ikut hilang ({len(hilang_benar)}): "
          f"{', '.join(sorted(hilang_benar)) or '-'}")
    print()

print("-" * 78)
print("FALSE POSITIVE PADA KONFIGURASI SEKARANG (AMBANG 2)")
print("-" * 78)
for b in sorted(dasar["set_fp"]):
    print(f"  {b:<8} muncul {bib_kemunculan[b]}x")