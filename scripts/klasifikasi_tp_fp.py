"""
klasifikasi_tp_fp.py
=====================
Skrip otomatis untuk mengklasifikasikan setiap nomor bib pada hasil_bib.csv
(keluaran akhir sistem) sebagai True Positive (TP) atau False Positive (FP),
dengan membandingkannya terhadap daftar ground truth manual (bib yang
benar-benar melintas kamera, dicatat oleh peneliti melalui pengamatan
visual langsung terhadap video).

LOGIKA KLASIFIKASI:
  - Untuk setiap bib pada hasil_bib.csv:
      - Jika bib tersebut ADA di daftar ground truth -> TP
      - Jika bib tersebut TIDAK ADA di daftar ground truth -> FP
  - False Negative (FN) dihitung sebagai bib pada ground truth yang TIDAK
    muncul sama sekali pada hasil_bib.csv.

Skrip ini menghasilkan:
  1. Tabel rincian per bib (status TP/FP) -> dipakai untuk Tabel rincian_v1/v2
  2. Ringkasan TP, FP, FN, Precision, Recall, F1-Score
  3. Daftar bib yang termasuk FN (ground truth yang tidak berhasil dilaporkan)

CARA PAKAI:
  1. Sesuaikan CSV_PATH ke lokasi hasil_bib.csv Anda.
  2. Isi GROUND_TRUTH dengan daftar nomor bib ground truth (hasil pengamatan
     manual Anda terhadap video), sebagai list string.
  3. Jalankan: python klasifikasi_tp_fp.py
  4. Salin tabel & ringkasan yang tercetak ke Bab 4 (Tabel rincian_v1/v2
     dan Tabel hasil_akhir).
"""

import pandas as pd
import os

# ============================================================
# KONFIGURASI - SESUAIKAN INI
# ============================================================

# Path ke hasil_bib.csv (keluaran akhir sistem untuk satu video)
CSV_PATH = r"D:\bib - detection\scripts\output\hasil_bib.csv"

# Daftar ground truth manual (nomor bib yang benar-benar melintas kamera,
# dicatat lewat pengamatan langsung terhadap video). Video 2 (21 bib):
GROUND_TRUTH = [
    "6264", "6404", "5490", "5760", "10452", "5808", "5516", "10058",
    "5944", "5048", "10444", "7156", "6125", "12070", "10012", "12074",
    "6011", "6124", "5471", "10777", "5262",
]

# Nama/label video, hanya untuk keperluan judul cetak
NAMA_VIDEO = "Video 2"

# ============================================================
# FUNGSI UTAMA
# ============================================================

def klasifikasi(csv_path, ground_truth, nama_video):
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Berkas tidak ditemukan:\n{csv_path}")

    df = pd.read_csv(csv_path, dtype={"Bib": str})

    # Bersihkan format bib (hilangkan .0 jika terbaca sebagai float, spasi, dst)
    df["Bib"] = (
        df["Bib"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
    )

    ground_truth_bersih = [str(b).strip() for b in ground_truth]

    bib_hasil = df["Bib"].tolist()

    # ------------------------------------------------------------
    # KLASIFIKASI TP / FP untuk setiap bib pada hasil_bib.csv
    # ------------------------------------------------------------
    rincian = []
    for _, row in df.iterrows():
        bib = row["Bib"]
        status = "TP" if bib in ground_truth_bersih else "FP"
        rincian.append({
            "Bib": bib,
            "Nama": row.get("Nama", ""),
            "Kategori": row.get("Kategori", ""),
            "Status": status,
        })

    df_rincian = pd.DataFrame(rincian)

    # ------------------------------------------------------------
    # HITUNG FN: ground truth yang TIDAK ADA di hasil_bib.csv
    # ------------------------------------------------------------
    bib_fn = [b for b in ground_truth_bersih if b not in bib_hasil]

    # ------------------------------------------------------------
    # RINGKASAN METRIK
    # ------------------------------------------------------------
    tp = (df_rincian["Status"] == "TP").sum()
    fp = (df_rincian["Status"] == "FP").sum()
    fn = len(bib_fn)
    total_gt = len(ground_truth_bersih)
    total_dilaporkan = len(df_rincian)

    precision = tp / total_dilaporkan if total_dilaporkan > 0 else 0
    recall = tp / total_gt if total_gt > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0
    )

    # ------------------------------------------------------------
    # CETAK HASIL
    # ------------------------------------------------------------
    print("=" * 70)
    print(f"KLASIFIKASI TP/FP - {nama_video}")
    print("=" * 70)
    print()
    print(f"Sumber hasil sistem : {csv_path}")
    print(f"Total ground truth  : {total_gt} bib")
    print(f"Total dilaporkan    : {total_dilaporkan} bib")
    print()

    print("-" * 70)
    print("RINCIAN PER BIB (untuk Tabel rincian_v1 / rincian_v2)")
    print("-" * 70)
    print(df_rincian.to_string(index=False))
    print()

    print("-" * 70)
    print(f"DAFTAR FALSE NEGATIVE ({fn} bib ground truth tidak dilaporkan)")
    print("-" * 70)
    if bib_fn:
        for b in bib_fn:
            print(f"  - Bib {b}")
    else:
        print("  (tidak ada)")
    print()

    print("-" * 70)
    print("RINGKASAN METRIK (untuk Tabel hasil_akhir)")
    print("-" * 70)
    print(f"True Positive (TP)  : {tp}")
    print(f"False Positive (FP) : {fp}")
    print(f"False Negative (FN) : {fn}")
    print(f"Ground Truth Total  : {total_gt}")
    print(f"Precision           : {precision*100:.1f}%  (= TP / Total Dilaporkan = {tp}/{total_dilaporkan})")
    print(f"Recall              : {recall*100:.1f}%  (= TP / Total Ground Truth = {tp}/{total_gt})")
    print(f"F1-Score            : {f1*100:.1f}%")
    print("=" * 70)

    return df_rincian, bib_fn, {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
    }


if __name__ == "__main__":
    klasifikasi(CSV_PATH, GROUND_TRUTH, NAMA_VIDEO)