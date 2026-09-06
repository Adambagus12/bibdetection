from ultralytics import YOLO

model = YOLO("models/best_v8n.pt")  # sesuaikan path relatif ke lokasi file ini
print("IoU aktif:", model.overrides.get("iou", "TIDAK DISET (pakai default Ultralytics)"))
print("Confidence aktif:", model.overrides.get("conf", "TIDAK DISET (pakai default Ultralytics)"))