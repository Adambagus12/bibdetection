import cv2
import os
import shutil
import csv

FOLDER_SUMBER = r"D:\bib - detection\output\gambar_dibuang"
FOLDER_LAYAK = r"D:\bib - detection\output\gambar_layak_anotasi"
FILE_CATATAN = r"D:\bib - detection\output\hasil_penyaringan.csv"

os.makedirs(FOLDER_LAYAK, exist_ok=True)

files = sorted([f for f in os.listdir(FOLDER_SUMBER) if f.lower().endswith(".jpg")])

print("=" * 70)
print("PENYARINGAN GAMBAR")
print("=" * 70)
print("Tekan tombol berikut saat gambar tampil:")
print("  1 = Bib terlihat JELAS (layak dianotasi)")
print("  2 = Bib ada tapi kecil/jauh")
print("  3 = Bib blur / tertutup")
print("  4 = Tidak ada pelari / tidak ada bib")
print("  q = Keluar")
print("=" * 70)
print()

hasil = []
for i, f in enumerate(files, 1):
    path = os.path.join(FOLDER_SUMBER, f)
    img = cv2.imread(path)
    if img is None:
        continue

    tampil = img.copy()
    cv2.putText(tampil, f"{i}/{len(files)}  {f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(tampil, "1=jelas  2=kecil  3=blur  4=tidak ada  q=keluar",
                (10, tampil.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("Penyaringan", tampil)
    key = cv2.waitKey(0) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('1'):
        kategori = "jelas"
        shutil.copy2(path, os.path.join(FOLDER_LAYAK, f))
    elif key == ord('2'):
        kategori = "kecil_jauh"
    elif key == ord('3'):
        kategori = "blur_tertutup"
    elif key == ord('4'):
        kategori = "tidak_ada_bib"
    else:
        kategori = "belum_dinilai"

    hasil.append({"file": f, "kategori": kategori})
    print(f"  {i:>3}/{len(files)}  {f:<20} -> {kategori}")

cv2.destroyAllWindows()

with open(FILE_CATATAN, "w", newline="", encoding="utf-8") as fp:
    w = csv.DictWriter(fp, fieldnames=["file", "kategori"])
    w.writeheader()
    w.writerows(hasil)

print()
print("=" * 70)
print("RINGKASAN")
print("=" * 70)
for k in ["jelas", "kecil_jauh", "blur_tertutup", "tidak_ada_bib", "belum_dinilai"]:
    n = sum(1 for h in hasil if h["kategori"] == k)
    if n:
        print(f"  {k:<18}: {n}")
print()
print(f"Gambar layak anotasi disalin ke: {FOLDER_LAYAK}")
print(f"Catatan tersimpan di: {FILE_CATATAN}")