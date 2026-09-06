import os
import csv
import shutil

FILE_PENYARINGAN = r"D:\bib - detection\output\hasil_penyaringan.csv"
FOLDER_LABEL = r"D:\bib - detection\datasets\video_framesv8n\labels"
FOLDER_SUMBER = r"D:\bib - detection\datasets\video_framesv8n\images"
FOLDER_OUT = r"D:\bib - detection\output\layak_anotasi_final"

os.makedirs(FOLDER_OUT, exist_ok=True)

# Baca hasil penyaringan manual
kategori = {}
with open(FILE_PENYARINGAN, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        kategori[row["file"]] = row["kategori"]

# Cari gambar yang labelnya kosong
label_kosong = set()
for f in os.listdir(FOLDER_LABEL):
    if f.endswith(".txt") and os.path.getsize(os.path.join(FOLDER_LABEL, f)) == 0:
        label_kosong.add(f.replace(".txt", ".jpg"))

print("=" * 70)
print("ANALISIS SILANG: LABEL KOSONG vs PENILAIAN VISUAL")
print("=" * 70)
print(f"Total gambar berlabel kosong : {len(label_kosong)}")
print()

rekap = {}
layak = []
for img in sorted(label_kosong):
    k = kategori.get(img, "tidak_dinilai")
    rekap[k] = rekap.get(k, 0) + 1
    if k == "jelas":
        layak.append(img)
        src = os.path.join(FOLDER_SUMBER, img)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(FOLDER_OUT, img))

print("Rincian kategori dari gambar berlabel kosong:")
for k, v in sorted(rekap.items(), key=lambda x: -x[1]):
    print(f"  {k:<18}: {v}")
print()
print("=" * 70)
print(f"TEMUAN: {len(layak)} gambar berlabel kosong ternyata bib-nya JELAS terlihat")
print("=" * 70)
for i, f in enumerate(layak, 1):
    print(f"  {i:>3}. {f}")
print()
print(f"Disalin ke: {FOLDER_OUT}")