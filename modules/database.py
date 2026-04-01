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

    def get_runner(self, bib):
        bib = str(bib).strip()

        # =========================
        # MATCH EXACT
        # =========================
        if bib in self.db:
            return bib, self.db[bib]

        # =========================
        # FUZZY MATCH 🔥
        # =========================
        match, score, _ = process.extractOne(bib, self.bib_list)

        if score > 80:
            return match, self.db[match]

        return None, None