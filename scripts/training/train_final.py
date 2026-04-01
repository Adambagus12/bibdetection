from ultralytics import YOLO

model = YOLO("../models/yolov8s.pt")

model.train(
    data="../datasets/final_dataset/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    name="bib_final"
)