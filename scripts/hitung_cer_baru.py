"""
hitung_cer_baru.py
==================
Menghitung CER dengan dua cara:
  1. Cara LAMA  : OCR dibandingkan dengan hasil pencocokan sistem (matched_bib)
  2. Cara BARU  : OCR dibandingkan dengan nomor bib SEBENARNYA (ground truth)

Perbandingan keduanya menunjukkan seberapa besar bias pada metrik CER lama.
"""

import pandas as pd

CSV_ANOTASI = r"D:\bib - detection\output\anotasi_cer_video1.csv"


def hitung_cer(teks_asli, teks_ocr):
    """Levenshtein distance dibagi panjang teks referensi."""
    teks_asli = str(teks_asli).strip()
    teks_ocr = str(teks_ocr).strip()
    if not teks_asli:
        return None

    n, m = len(teks_asli), len(teks_ocr)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if teks_asli[i - 1] == teks_ocr[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[n][m] / n


def bersih(x):
    return str(x).replace(".0", "").strip()


df = pd.read_csv(CSV_ANOTASI)
df = df[df["status"] == "ok"].copy()

print("=" * 78)
print("PERHITUNGAN ULANG CER TERHADAP GROUND TRUTH")
print("=" * 78)
print(f"Sampel dianotasi : {len(df)}")
print()

hasil = []
for _, r in df.iterrows():
    ocr = bersih(r["ocr_text_raw"])
    asli = bersih(r["bib_asli"])
    matched = bersih(r["matched_bib"]) if pd.notna(r["matched_bib"]) else None

    cer_gt = hitung_cer(asli, ocr)
    cer_lama = hitung_cer(matched, ocr) if matched and matched != "nan" else None
    benar = (matched == asli) if matched and matched != "nan" else None

    hasil.append({
        "frame": int(r["frame"]), "ocr": ocr, "asli": asli,
        "matched": matched if matched != "nan" else "-",
        "cer_gt": cer_gt, "cer_lama": cer_lama, "identitas_benar": benar
    })

h = pd.DataFrame(hasil)

# ---------- CER cara lama ----------
lama = h[h["cer_lama"].notna()]
print("-" * 78)
print("CARA LAMA (OCR vs hasil pencocokan sistem)")
print("-" * 78)
print(f"  Jumlah kasus tercocokkan : {len(lama)}")
if len(lama):
    print(f"  CER rata-rata            : {lama['cer_lama'].mean()*100:.2f}%")
    print(f"  CER median               : {lama['cer_lama'].median()*100:.2f}%")
print()

# ---------- CER cara baru ----------
print("-" * 78)
print("CARA BARU (OCR vs nomor bib sebenarnya / ground truth)")
print("-" * 78)
print(f"  Jumlah sampel            : {len(h)}")
print(f"  CER rata-rata            : {h['cer_gt'].mean()*100:.2f}%")
print(f"  CER median               : {h['cer_gt'].median()*100:.2f}%")
print(f"  CER minimum              : {h['cer_gt'].min()*100:.2f}%")
print(f"  CER maksimum             : {h['cer_gt'].max()*100:.2f}%")
print()

# ---------- ketepatan identitas ----------
cocok = h[h["identitas_benar"].notna()]
if len(cocok):
    benar = cocok["identitas_benar"].sum()
    print("-" * 78)
    print("KETEPATAN IDENTITAS HASIL PENCOCOKAN DATABASE")
    print("-" * 78)
    print(f"  Tercocokkan ke database  : {len(cocok)}")
    print(f"  Identitas BENAR          : {benar} ({benar/len(cocok)*100:.1f}%)")
    print(f"  Identitas SALAH          : {len(cocok)-benar} ({(len(cocok)-benar)/len(cocok)*100:.1f}%)")
    print()

# ---------- kasus OCR salah tapi CER lama 0% ----------
tersembunyi = h[(h["cer_lama"] == 0) & (h["identitas_benar"] == False)]
if len(tersembunyi):
    print("-" * 78)
    print(f"KASUS TERSEMBUNYI: CER lama 0% tetapi identitas SALAH ({len(tersembunyi)} kasus)")
    print("-" * 78)
    print(f"{'Frame':<8}{'OCR baca':<12}{'Dilaporkan':<12}{'Bib asli':<12}{'CER benar':<10}")
    for _, r in tersembunyi.iterrows():
        print(f"{r['frame']:<8}{r['ocr']:<12}{r['matched']:<12}{r['asli']:<12}{r['cer_gt']*100:>6.1f}%")
    print()

# ---------- ringkasan ----------
print("=" * 78)
print("RINGKASAN UNTUK LAPORAN")
print("=" * 78)
if len(lama):
    print(f"  CER cara lama (bias)     : {lama['cer_lama'].mean()*100:.2f}%")
print(f"  CER terhadap ground truth: {h['cer_gt'].mean()*100:.2f}%")
if len(lama):
    selisih = h['cer_gt'].mean()*100 - lama['cer_lama'].mean()*100
    print(f"  Selisih                  : {selisih:+.2f} poin persentase")
print("=" * 78)

h.to_csv(r"D:\bib - detection\output\hasil_cer_perbandingan.csv", index=False)
print("\nRincian tersimpan di: D:\\bib - detection\\output\\hasil_cer_perbandingan.csv")