import re
import pytesseract

# =========================================================
# KONFIGURASI TESSERACT — sesuaikan path dengan instalasi
# Tesseract di komputer kamu
# =========================================================
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def read_text_tesseract(image):
    """
    Versi Tesseract dari read_text() di modules/ocr.py.
    Postprocessing SAMA PERSIS dengan versi EasyOCR (gabung teks,
    buang non-digit, hapus leading zero, filter panjang 4-5 digit)
    supaya perbandingan kedua engine adil.
    """
    if image is None:
        return ""

    try:
        # PSM 7 = anggap citra sebagai satu baris teks (cocok untuk
        # region bib yang sudah di-crop)
        custom_config = r"--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789"

        combined = pytesseract.image_to_string(image, config=custom_config)

        if not combined:
            return ""

        # Hapus semua non-digit
        text = re.sub(r'\D', '', combined)

        # Hilangkan leading nol
        text = text.lstrip("0")

        # Jika kosong setelah hapus nol
        if not text:
            return ""

        # Hanya 4-5 digit
        if len(text) < 4 or len(text) > 5:
            return ""

        return text

    except Exception:
        return ""