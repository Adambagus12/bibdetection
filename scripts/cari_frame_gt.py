import pandas as pd

CSV_PATH = r"D:\bib - detection\scripts\output\ocr_raw_log.csv"

df = pd.read_csv(CSV_PATH)
df["area"] = (df["x2"] - df["x1"]) * (df["y2"] - df["y1"])

print("=" * 70)
print("KANDIDAT KATEGORI MUDAH (area besar + confidence tinggi)")
print("=" * 70)
mudah = df[(df["confidence"] > 0.85) & (df["area"] > 3000)].nlargest(8, "area")
print(mudah[["frame", "ocr_text_raw", "matched_bib", "confidence", "area"]].to_string(index=False))

print()
print("=" * 70)
print("KANDIDAT KATEGORI SULIT (area kecil)")
print("=" * 70)
sulit = df[df["area"] < 1200].nsmallest(8, "area")
print(sulit[["frame", "ocr_text_raw", "matched_bib", "confidence", "area"]].to_string(index=False))

print()
print("=" * 70)
print("PENCARIAN BIB SPESIFIK (titik perbedaan antar pengamat)")
print("=" * 70)
for bib in ["6049", "5280", "6086", "6149", "10373"]:
    hasil = df[
        (df["matched_bib"].astype(str) == bib) |
        (df["ocr_text_raw"].astype(str).str.contains(bib, na=False))
    ]
    if len(hasil) > 0:
        baris = hasil.nlargest(1, "area").iloc[0]
        print(f"Bib {bib:<8}: DITEMUKAN di frame {int(baris['frame'])} "
              f"(area {int(baris['area'])}, conf {baris['confidence']:.3f})")
    else:
        print(f"Bib {bib:<8}: tidak ada di log (tidak pernah terdeteksi sistem)")