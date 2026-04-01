from ultralytics import YOLO

class Detector:
    def __init__(self, model_path, confidence=0.45):
        self.model = YOLO(model_path)
        self.confidence = confidence  # 🔥 SIMPAN CONFIDENCE

    def detect(self, frame):
        boxes = []

        # 🔥 PERBAIKAN: PAKAI CONFIDENCE
        results = self.model(frame, conf=self.confidence)

        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                boxes.append((x1, y1, x2, y2))

        return boxes