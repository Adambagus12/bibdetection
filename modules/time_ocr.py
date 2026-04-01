import cv2
import pytesseract
import re

class TimeOCR:
    def __init__(self):
        self.last_time = ""

    def extract_timer_area(self, frame):
        # =========================
        # ANTI ERROR (frame None)
        # =========================
        if frame is None:
            return None

        h, w = frame.shape[:2]

        # =========================
        # AREA TIMER (KIRI ATAS)
        # =========================
        timer_crop = frame[0:int(h * 0.12), 0:int(w * 0.30)]

        return timer_crop

    def preprocess(self, img):
        if img is None or img.size == 0:
            return None

        # =========================
        # RESIZE (biar OCR lebih jelas)
        # =========================
        img = cv2.resize(img, None, fx=3, fy=3)

        # =========================
        # DETEKSI WARNA MERAH 🔴
        # =========================
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        lower_red1 = (0, 120, 70)
        upper_red1 = (10, 255, 255)

        lower_red2 = (170, 120, 70)
        upper_red2 = (180, 255, 255)

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

        mask = mask1 | mask2

        # =========================
        # INVERT → angka jadi putih
        # =========================
        result = cv2.bitwise_not(mask)

        return result

    def read_time(self, frame):
        # =========================
        # VALIDASI FRAME
        # =========================
        if frame is None:
            return self.last_time

        # =========================
        # AMBIL AREA TIMER
        # =========================
        timer_crop = self.extract_timer_area(frame)

        if timer_crop is None:
            return self.last_time

        # =========================
        # PREPROCESS
        # =========================
        processed = self.preprocess(timer_crop)

        if processed is None:
            return self.last_time

        # =========================
        # OCR
        # =========================
        text = pytesseract.image_to_string(
            processed,
            config='--psm 7 -c tessedit_char_whitelist=0123456789:.'
        )

        # =========================
        # CLEAN TEXT
        # =========================
        text = text.strip().replace(" ", "").replace("\n", "")

        # =========================
        # REGEX (AMBIL SAMPAI MILLISECOND)
        # =========================
        match = re.search(r"\d{1,2}:\d{2}:\d{2}\.\d{2,3}", text)

        if match:
            self.last_time = match.group()

        return self.last_time