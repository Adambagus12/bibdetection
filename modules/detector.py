from ultralytics import YOLO

class Detector:
    def __init__(self, model_path, confidence=0.45, iou=0.45):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.iou = iou

    def detect(self, frame):
        boxes = []

        results = self.model(frame, conf=self.confidence, iou=self.iou, verbose=False)

        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # PERBAIKAN: ambil confidence asli dari hasil deteksi YOLO,
                # bukan nilai konstan 1.0 seperti sebelumnya
                conf = float(box.conf[0])

                boxes.append((x1, y1, x2, y2, conf))

        return boxes