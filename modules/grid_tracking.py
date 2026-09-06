class GridTrackingSystem:
    """
    Implementasi tracking berbasis posisi grid sesuai desain Bab 2/3.

    Cara kerja (2 fase, sesuai kerangka pemikiran):
    - FASE 1 (selama loop video): setiap deteksi yang punya hasil OCR
      dipetakan ke sel grid berdasarkan titik pusat bounding box, lalu
      disimpan sementara (belum dicocokkan ke database).
    - FASE 2 (setelah loop selesai / finalize()): tiap grup grid yang
      punya minimal `min_frame_valid` kemunculan dianggap valid, dipilih
      satu deteksi terbaik (skor area x confidence tertinggi), BARU
      dicocokkan ke database sekali per grup.
    """

    def __init__(self, grid_size=120, min_frame_valid=2, finish_zone=(0.2, 0.8)):
        self.grid_size = grid_size
        self.min_frame_valid = min_frame_valid
        self.finish_zone = finish_zone
        self.tracks = {}  # grid_id -> list of records

    def add_detection(self, text, x1, y1, x2, y2, conf, frame, timestamp):
        """Dipanggil di dalam loop utama, untuk setiap deteksi dengan OCR text valid."""
        if not text or frame is None:
            return

        frame_h, frame_w = frame.shape[:2]
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # Tetap pakai filter zona finish sebagai penyaring awal, konsisten
        # dengan asumsi kamera statis di area finish line
        if not (frame_w * self.finish_zone[0] < cx < frame_w * self.finish_zone[1]):
            return

        grid_id = (cx // self.grid_size, cy // self.grid_size)

        area = (x2 - x1) * (y2 - y1)
        score = area * conf

        record = {"text": text, "score": score, "timestamp": timestamp}
        self.tracks.setdefault(grid_id, []).append(record)

    def finalize(self, database, cer_func=None):
        """
        Dipanggil SEKALI setelah loop video selesai.

        Return dict berformat sama dengan FinishTimeSystem.get_results():
        { matched_bib: {"score": ..., "time": ..., "cer": ...} }
        """
        results = {}

        for grid_id, records in self.tracks.items():
            # Syarat multi-frame validation: minimal N kemunculan di grid yang sama
            if len(records) < self.min_frame_valid:
                continue

            # Best detection scoring: pilih skor tertinggi dalam grup grid ini
            best_record = max(records, key=lambda r: r["score"])
            text = best_record["text"]

            # Database matching: HANYA SEKALI per grup grid (bukan per frame)
            matched_bib, runner, _, _ = database.get_runner(text)
            if not runner:
                continue

            entry = {"score": best_record["score"], "time": best_record["timestamp"]}
            if cer_func:
                entry["cer"] = cer_func(str(matched_bib), str(text))

            # Kalau bib yang sama ternyata muncul di lebih dari satu sel grid
            # (misalnya karena pergeseran lateral saat berlari), simpan yang
            # skornya lebih tinggi
            if matched_bib not in results or entry["score"] > results[matched_bib]["score"]:
                results[matched_bib] = entry

        return results