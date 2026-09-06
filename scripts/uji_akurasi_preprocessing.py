"""
uji_akurasi_preprocessing.py
=============================
Menguji AKURASI (bukan sekadar tingkat baca) dari tiap konfigurasi
preprocessing, dengan membandingkan hasil OCR terhadap daftar ground
truth Video 1.

Logika:
  - Teks OCR yang PERSIS SAMA dengan salah satu bib di ground truth
    dianggap BENAR
  - Teks yang tidak ada di ground truth dianggap SALAH (bacaan sampah)
"""

import pandas as pd

CSV_UJI = r"D:\bib - detection\output\uji_preprocessing.csv"

# Ground truth Video 1 - gabungan dari 3 pengamat (varian mayoritas)
GROUND_TRUTH = [
    "5377","5231","10062","5198","5059","6377","6070","5934","5480","5473",
    "10426","6319","12044","10021","5244","10610","5271","5187","10573","6085",
    "5260","5257","10271","5990","6156","10237","5843","5845","10386","5569",
    "5567","6462","10499","7141","10483","6406","6429","5708","5535","10023",
    "6155","6154","5193","6049","6149","5280","10373",
]

VARIAN = ["A_sekarang", "B_tanpa_histeq", "C_crop_ketat", "D_ketat_nohisteq"]


def bersih(x):
    s = str(x).strip()
    if s in ("nan", "-", ""):
        return ""
    return s.replace(".0", "")


gt = set(GROUND_TRUTH)
df = pd.read_csv(CSV_UJI)

print("=" * 82)
print("UJI AKURASI KONFIGURASI PREPROCESSING")
print("=" * 82)
print(f"Total deteksi diproses : {len(df)}")
print(f"Ground truth Video 1   : {len(gt)} bib")
print()

rekap = []
for v in VARIAN:
    if v not in df.columns:
        continue

    teks = df[v].apply(bersih)
    terbaca = teks[teks != ""]

    benar = terbaca[terbaca.isin(gt)]
    salah = terbaca[~terbaca.isin(gt)]

    bib_unik_benar = set(benar.unique())

    rekap.append({
        "varian": v,
        "terbaca": len(terbaca),
        "benar": len(benar),
        "salah": len(salah),
        "akurasi": len(benar) / len(terbaca) * 100 if len(terbaca) else 0,
        "bib_unik": len(bib_unik_benar),
        "cakupan": len(bib_unik_benar) / len(gt) * 100,
        "set_unik": bib_unik_benar,
    })

print(f"{'Konfigurasi':<20}{'Terbaca':<10}{'Benar':<9}{'Salah':<9}"
      f"{'Akurasi':<11}{'Bib Unik':<11}{'Cakupan GT'}")
print("-" * 82)
for r in rekap:
    print(f"{r['varian']:<20}{r['terbaca']:<10}{r['benar']:<9}{r['salah']:<9}"
          f"{r['akurasi']:>6.1f}%    {r['bib_unik']:<11}{r['cakupan']:>6.1f}%")
print()

# ---------- kesimpulan ----------
print("=" * 82)
print("KESIMPULAN")
print("=" * 82)

best_akurasi = max(rekap, key=lambda r: r["akurasi"])
best_cakupan = max(rekap, key=lambda r: r["bib_unik"])
sekarang = next((r for r in rekap if r["varian"] == "A_sekarang"), None)

print(f"Akurasi tertinggi  : {best_akurasi['varian']} ({best_akurasi['akurasi']:.1f}%)")
print(f"Cakupan tertinggi  : {best_cakupan['varian']} ({best_cakupan['bib_unik']} bib unik)")
print()

if sekarang:
    print(f"Dibandingkan konfigurasi sekarang (A):")
    for r in rekap:
        if r["varian"] == "A_sekarang":
            continue
        d_ak = r["akurasi"] - sekarang["akurasi"]
        d_bib = r["bib_unik"] - sekarang["bib_unik"]
        tanda_ak = "+" if d_ak >= 0 else ""
        tanda_bib = "+" if d_bib >= 0 else ""
        print(f"  {r['varian']:<20} akurasi {tanda_ak}{d_ak:>5.1f} poin   "
              f"bib unik {tanda_bib}{d_bib:>3}")
    print()

# ---------- bib yang hanya ditemukan varian tertentu ----------
if sekarang:
    print("-" * 82)
    print("BIB YANG DITEMUKAN VARIAN LAIN TAPI TIDAK OLEH KONFIGURASI SEKARANG")
    print("-" * 82)
    for r in rekap:
        if r["varian"] == "A_sekarang":
            continue
        tambahan = r["set_unik"] - sekarang["set_unik"]
        hilang = sekarang["set_unik"] - r["set_unik"]
        print(f"{r['varian']}:")
        print(f"   tambahan ({len(tambahan)}): {', '.join(sorted(tambahan)) or '-'}")
        print(f"   hilang   ({len(hilang)}): {', '.join(sorted(hilang)) or '-'}")
        print()