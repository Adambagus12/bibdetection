import cv2
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# =========================
# PILIH VIDEO (POPUP)
# =========================
Tk().withdraw()  # sembunyikan window utama
video_path = askopenfilename(title="Pilih Video")

if not video_path:
    print("Tidak ada video dipilih!")
    exit()

# =========================
# KONFIGURASI
# =========================
output_folder = "../datasets/video_framesv8n/images"
interval = 0.5  # detik

# =========================
# BUAT FOLDER OUTPUT
# =========================
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# =========================
# BACA VIDEO
# =========================
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(fps * interval)

count = 0
saved = 0

print(f"Memproses video: {video_path}")
print("Mulai ekstrak frame...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if count % frame_interval == 0:
        filename = os.path.join(output_folder, f"frame_{saved:04d}.jpg")
        cv2.imwrite(filename, frame)
        saved += 1

    count += 1

cap.release()

print(f"Selesai! Total frame tersimpan: {saved}")
print(f"Lokasi folder: {os.path.abspath(output_folder)}")