import os
import re
import pandas as pd
from rapidfuzz import process

# Margin ambiguitas dapat diatur lewat variabel lingkungan untuk keperluan
# eksperimen. Nilai 0 berarti deteksi ambiguitas dinonaktifkan.
AMBIGUITY_MARGIN = int(os.environ.get("AMBIGUITY_MARGIN", "10"))


class RunnerDatabase:
    def __init__(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".csv":
            self.data = pd.read_csv(file_path, dtype=str)
        elif ext in (".xlsx", ".xls"):
            self.data = pd.read_excel(file_path, dtype=str)
        else:
            raise ValueError(
                f"Format database tidak didukung: '{ext}'. "
                "Gunakan file .csv atau .xlsx."
            )

        self.data.columns = [c.strip() for c in self.data.columns]

        kolom_wajib = ["BIB", "Nama", "Kategori"]
        kolom_hilang = [k for k in kolom_wajib if k not in self.data.columns]
        if kolom_hilang:
            raise ValueError(
                f"Database tidak valid: kolom wajib berikut tidak ditemukan: "
                f"{', '.join(kolom_hilang)}. Pastikan berkas memiliki kolom "
                f"'BIB', 'Nama', dan 'Kategori'."
            )

        self.db = {}
        self.bib_list = []

        for _, row in self.data.iterrows():
            bib = row["BIB"]
            if pd.isna(bib):
                continue

            bib = str(bib).strip()
            # Buang akhiran ".0" yang muncul ketika berkas Excel menyimpan
            # nomor bib sebagai bilangan pecahan (contoh: "10004.0").
            if bib.endswith(".0"):
                bib = bib[:-2]

            # Abaikan entri yang bukan angka murni maupun yang panjangnya
            # di luar rentang wajar nomor bib, karena sebagian berkas peserta
            # memuat baris non-bib seperti nomor urut atau kode kategori.
            if not bib.isdigit() or len(bib) < 4 or len(bib) > 6:
                continue

            self.db[bib] = {
                "nama": str(row["Nama"]).strip(),
                "kategori": str(row["Kategori"]).strip()
            }
            self.bib_list.append(bib)

    def get_runner(self, bib, min_score=80, ambiguity_margin=None):
        """
        min_score        : skor minimal fuzzy match agar dianggap kandidat valid
        ambiguity_margin : jika selisih skor kandidat #1 dan #2 kurang dari
                           nilai ini, match dianggap AMBIGU dan ditolak

        Return: (matched_bib, runner_dict, best_score, second_score)
        """
        if ambiguity_margin is None:
            ambiguity_margin = AMBIGUITY_MARGIN

        bib = str(bib).strip()

        if bib in self.db:
            return bib, self.db[bib], 100, None

        # Pencocokan berbasis awalan untuk menangani perbedaan format:
        # sebagian event menuliskan bib sebagai 2062.1 pada bib fisik,
        # namun tercatat sebagai 206201 pada basis data.
        if len(bib) == 5 and bib.isdigit():
            kandidat_awalan = [b for b in self.bib_list
                               if len(b) == 6
                               and b.startswith(bib[:4])
                               and b.endswith(bib[4])]
            if len(kandidat_awalan) == 1:
                hasil = kandidat_awalan[0]
                return hasil, self.db[hasil], 95, None

        candidates = process.extract(bib, self.bib_list, limit=2)

        if not candidates:
            return None, None, None, None

        best_match, best_score, _ = candidates[0]

        if best_score <= min_score:
            return None, None, best_score, None

        second_score = None
        if len(candidates) >= 2:
            second_match, second_score, _ = candidates[1]

            if ambiguity_margin > 0 and (best_score - second_score) < ambiguity_margin:
                return None, None, best_score, second_score

        return best_match, self.db[best_match], best_score, second_score