import pandas as pd
from rapidfuzz import process

class RunnerDatabase:
    def __init__(self, excel_path):
        self.data = pd.read_excel(excel_path)
        self.data.columns = [c.strip() for c in self.data.columns]

        self.db = {}
        self.bib_list = []

        for _, row in self.data.iterrows():
            bib = row["BIB"]

            if pd.notna(bib):
                bib = str(int(bib))
            else:
                continue

            self.db[bib] = {
                "nama": str(row["Nama"]).strip(),
                "kategori": str(row["Kategori"]).strip()
            }

            self.bib_list.append(bib)

    def get_runner(self, bib, min_score=80, ambiguity_margin=10):
        """
        min_score        : skor minimal fuzzy match agar dianggap kandidat valid
        ambiguity_margin : jika selisih skor kandidat #1 dan #2 kurang dari
                            nilai ini, match dianggap AMBIGU dan ditolak
                            (lebih baik tidak match daripada salah match)

        Return: (matched_bib, runner_dict, best_score, second_score)
        - matched_bib / runner_dict bernilai None kalau tidak ada match
          yang valid (baik karena skor terlalu rendah maupun karena ambigu)
        - best_score  : skor kandidat teratas (None kalau tidak ada kandidat)
        - second_score: skor kandidat kedua (None kalau tidak ada kandidat kedua,
                         atau kalau match-nya exact)
        """
        bib = str(bib).strip()

        # =========================
        # MATCH EXACT (selalu prioritas, tidak pernah ambigu)
        # =========================
        if bib in self.db:
            return bib, self.db[bib], 100, None

        # =========================
        # FUZZY MATCH DENGAN CEK AMBIGUITAS
        # =========================
        # Ambil 2 kandidat teratas, bukan cuma 1
        candidates = process.extract(bib, self.bib_list, limit=2)

        if not candidates:
            return None, None, None, None

        best_match, best_score, _ = candidates[0]

        if best_score <= min_score:
            return None, None, best_score, None

        # Kalau ada kandidat kedua, cek apakah terlalu berdekatan skornya
        second_score = None
        if len(candidates) >= 2:
            second_match, second_score, _ = candidates[1]

            if (best_score - second_score) < ambiguity_margin:
                # Ambigu: dua kandidat sama-sama masuk akal, sistem
                # tidak cukup yakin -> tolak daripada salah pilih
                return None, None, best_score, second_score

        return best_match, self.db[best_match], best_score, second_score