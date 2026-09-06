import os
import shutil

FOLDER_375 = r"D:\bib - detection\datasets\video_framesv8n\images"
FOLDER_FINAL = r"D:\bib - detection\datasets\final_dataset\train\images"
FOLDER_OUTPUT = r"D:\bib - detection\output\gambar_dibuang"

os.makedirs(FOLDER_OUTPUT, exist_ok=True)

nama_375 = set(os.listdir(FOLDER_375))
nama_final = set(os.listdir(FOLDER_FINAL))

# Gambar dari 375 yang TIDAK masuk ke dataset final
dibuang = sorted(nama_375 - nama_final)
masuk = sorted(nama_375 & nama_final)

print("=" * 70)
print("PENCARIAN GAMBAR YANG DIBUANG SAAT PEMBERSIHAN DATASET")
print("=" * 70)
print(f"Gambar hasil ekstraksi     : {len(nama_375)}")
print(f"Masuk ke dataset final     : {len(masuk)}")
print(f"Dibuang saat pembersihan   : {len(dibuang)}")
print()

if dibuang:
    print("Daftar gambar yang dibuang:")
    for i, f in enumerate(dibuang, 1):
        print(f"  {i:>3}. {f}")
        shutil.copy2(os.path.join(FOLDER_375, f),
                     os.path.join(FOLDER_OUTPUT, f))
    print()
    print(f"Semua gambar disalin ke: {FOLDER_OUTPUT}")
else:
    print("Tidak ada selisih. Semua gambar masuk ke dataset final.")
    print("Kemungkinan nama file berubah saat penggabungan dataset.")