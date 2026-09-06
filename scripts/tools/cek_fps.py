import cv2

videos = [
    ("Video 1", r"D:\bib - detection\vidio\uji1.mp4"),
    ("Video 2", r"D:\bib - detection\vidio\uji2.mp4"),  # sesuaikan nama file video 2
]

for nama, path in videos:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"{nama}: GAGAL membuka file di {path}")
        continue
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps if fps > 0 else 0
    print(f"{nama}:")
    print(f"  FPS        : {fps}")
    print(f"  Frame count: {int(frame_count)}")
    print(f"  Durasi     : {duration:.2f} detik")
    print()
    cap.release()