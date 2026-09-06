from ultralytics import YOLO
import os
import cv2

# =========================
# LOAD MODEL
# =========================
model = YOLO("../../models/best_roboflow_v8n.pt")

# =========================
# FOLDER
# =========================
image_folder = "../../datasets/video_framesv8n/images"
label_folder = "../../datasets/video_framesv8n/labels"

if not os.path.exists(label_folder):
    os.makedirs(label_folder)

# =========================
# AUTO LABEL
# =========================
for image_name in os.listdir(image_folder):
    if image_name.endswith(".jpg"):
        image_path = os.path.join(image_folder, image_name)

        results = model.predict(image_path, conf=0.5)

        for r in results:
            boxes = r.boxes

            label_path = os.path.join(label_folder, image_name.replace(".jpg", ".txt"))

            with open(label_path, "w") as f:
                for box in boxes:
                    cls = int(box.cls[0])
                    x, y, w, h = box.xywhn[0]

                    f.write(f"{cls} {x} {y} {w} {h}\n")

print("Auto-label selesai!")