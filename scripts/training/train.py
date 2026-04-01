from ultralytics import YOLO

# =========================
# LOAD MODEL AWAL
# =========================
# Gunakan model yang lebih akurat
model = YOLO("../models/yolov8s.pt")  # rekomendasi untuk RAM 16GB

# =========================
# TRAINING
# =========================
model.train(
    data="../datasets/roboflow_dataset/data.yaml",  # path dataset kamu
    epochs=75,                 # jumlah training
    imgsz=640,                 # ukuran gambar
    batch=16,                  # cocok untuk RAM 16GB
    patience=20,               # stop kalau tidak improve
    name="bib_model",          # nama folder hasil
    workers=4                  # percepat loading data
)