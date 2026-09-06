import os
import shutil

FOLDER_LABEL = r"D:\bib - detection\datasets\video_framesv8n\labels"
FOLDER_IMG = r"D:\bib - detection\datasets\video_framesv8n\images"
FOLDER_OUT = r"D:\bib - detection\output\gambar_dibuang"

os.makedirs(FOLDER_OUT, exist_ok=True)

kosong = []
berisi = []

for f in sorted(os.listdir(FOLDER_LABEL)):
    if not f.endswith(".txt"):
        continue
    path = os.path.join(FOLDER_LABEL, f)
    ukuran = os.path.getsize(path)
    if ukuran == 0:
        kosong.append(f)
    else:
        berisi.append(f)

print("=" * 70)
print("ANALISIS FILE LABEL HASIL PSEUDO-LABELING")
print("=" * 70)
print(f"Total file label       : {len(kosong) + len(berisi)}")
print(f"Label BERISI anotasi   : {len(berisi)}")
print(f"Label KOSONG (0 byte)  : {len(kosong)}")
print()

if kosong:
    print("Daftar gambar dengan label kosong (kandidat gambar yang dibuang):")
    disalin = 0
    for i, lbl in enumerate(kosong, 1):
        nama_img = lbl.replace(".txt", ".jpg")
        src = os.path.join(FOLDER_IMG, nama_img)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(FOLDER_OUT, nama_img))
            disalin += 1
            print(f"  {i:>3}. {nama_img}")
    print()
    print(f"{disalin} gambar disalin ke: {FOLDER_OUT}")
else:
    print("Tidak ada label kosong.")
    print("Kemungkinan pembersihan dilakukan dengan menghapus file, bukan mengosongkan.")