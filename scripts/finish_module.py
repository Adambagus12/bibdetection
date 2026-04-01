class FinishTimeSystem:
    def __init__(self):
        # simpan hasil terbaik
        self.best_detection = {}

        # counter untuk stabilisasi OCR
        self.counter = {}

    def process(self, matched_bib, text, x1, y1, x2, y2, conf, frame, timestamp):

        # =========================
        # VALIDASI FRAME
        # =========================
        if frame is None:
            return

        frame_h, frame_w = frame.shape[:2]
        cx = (x1 + x2) // 2

        # =========================
        # ZONA FINISH (DIPERLUAS)
        # =========================
        if not (frame_w * 0.2 < cx < frame_w * 0.8):
            return

        # =========================
        # MULTI FRAME (LEBIH RINGAN)
        # =========================
        self.counter[text] = self.counter.get(text, 0) + 1

        if self.counter[text] < 1:
            return

        # =========================
        # HITUNG SCORE
        # =========================
        area = (x2 - x1) * (y2 - y1)
        score = area * conf

        # =========================
        # FIX UTAMA 🔥
        # =========================

        # 1. Kalau belum ada → simpan langsung
        if matched_bib not in self.best_detection:
            self.best_detection[matched_bib] = {
                "score": score,
                "time": timestamp
            }

        else:
            # 2. Update jika lebih bagus
            if score > self.best_detection[matched_bib]["score"]:
                self.best_detection[matched_bib] = {
                    "score": score,
                    "time": timestamp
                }

            # 3. FALLBACK (kalau waktu kosong tapi sekarang ada)
            elif not self.best_detection[matched_bib]["time"] and timestamp:
                self.best_detection[matched_bib]["time"] = timestamp

    def get_results(self):
        return self.best_detection