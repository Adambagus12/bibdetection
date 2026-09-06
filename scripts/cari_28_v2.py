import os
import cv2
import hashlib
import shutil

FOLDER_375 = r"D:\bib - detection\datasets\video_framesv8n\images"
FOLDER_FINAL = r"D:\bib - detection\datasets\final_dataset\train\images"
FOLDER_OUTPUT = r"D:\bib - detection\output\gambar_dibuang"

os.makedirs(FOLDER_OUTPUT, exist_ok=True)


def hash_img(path):
    """Hash berdasarkan isi gambar setelah di-resize kecil (tahan perbedaan kompresi)."""
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.resize(img, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return hashlib.md5(img.tobytes()).hexdigest()


print("Tahap 1: menghitung hash 375 gambar lokal...")
hash_lokal = {}
for f in os.listdir(FOLDER_375):
    if f.lower().endswith((".jpg", ".png")):
        h = hash_img(os.path.join(FOLDER_375, f))
        if h:
            hash_lokal[h] = f
print(f"  {len(hash_lokal)} gambar diproses\n")

print("Tahap 2: memindai dataset final (8.168 gambar, mohon tunggu)...")
files_final = [f for f in os.listdir(FOLDER_FINAL)
               if f.lower().endswith((".jpg", ".png"))]

ditemukan = set()
for i, f in enumerate(files_final, 1):
    if i % 1000 == 0:
        print(f"  {i}/{len(files_final)}...")
    h = hash_img(os.path.join(FOLDER_FINAL, f))
    if h and h in hash_lokal:
        ditemukan.add(hash_lokal[h])

print(f"  selesai\n")

semua = set(hash_lokal.values())
dibuang = sorted(semua - ditemukan)

print("=" * 70)
print("HASIL PENCARIAN")
print("=" * 70)
print(f"Total gambar hasil ekstraksi : {len(semua)}")
print(f"Ditemukan di dataset final   : {len(ditemukan)}")
print(f"Tidak ditemukan (dibuang)    : {len(dibuang)}")
print()

if dibuang:
    for i, f in enumerate(dibuang, 1):
        print(f"  {i:>3}. {f}")
        shutil.copy2(os.path.join(FOLDER_375, f),
                     os.path.join(FOLDER_OUTPUT, f))
    print(f"\nDisalin ke: {FOLDER_OUTPUT}")