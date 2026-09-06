import cv2
import os

KANDIDAT = [
    r"D:\2025-10-19 07-00-04.mkv",
]

INTERVAL = 0.5
TARGET = 375

print("=" * 80)
print("CEK KANDIDAT VIDEO SUMBER DATASET")
print("=" * 80)
print(f"Target: video yang menghasilkan {TARGET} gambar pada interval {INTERVAL} detik")
print()

for path in KANDIDAT:
    if not os.path.isfile(path):
        print(f"[?] TIDAK DITEMUKAN: {path}")
        continue

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[!] GAGAL DIBUKA : {os.path.basename(path)}")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    durasi = total_frame / fps if fps > 0 else 0

    frame_interval = int(fps * INTERVAL)
    perkiraan = 0
    if frame_interval > 0:
        perkiraan = len(range(0, total_frame, frame_interval))

    selisih = abs(perkiraan - TARGET)
    if selisih == 0:
        status = "<<< COCOK PERSIS"
    elif selisih <= 3:
        status = "<<< SANGAT MUNGKIN"
    elif selisih <= 15:
        status = "<-- mungkin"
    else:
        status = ""

    print(f"{os.path.basename(path)}")
    print(f"   Durasi          : {durasi:.1f} detik ({durasi/60:.1f} menit)")
    print(f"   FPS             : {fps:.1f}")
    print(f"   Total frame     : {total_frame}")
    print(f"   Perkiraan hasil : {perkiraan} gambar  {status}")
    print()

    cap.release()

print("=" * 80)
print("Video yang benar adalah yang perkiraan hasilnya mendekati 375 gambar.")