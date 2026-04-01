import pytesseract
import re

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def read_text(image):
    if image is None:
        return ""

    text = pytesseract.image_to_string(
        image,
        config='--psm 6 -c tessedit_char_whitelist=0123456789'
    )

    text = text.strip()
    text = re.sub(r'\D', '', text)

    # =========================
    # NORMALISASI
    # =========================

    # hilangkan leading nol
    text = text.lstrip("0")

    # jika kosong setelah hapus nol
    if not text:
        return ""

    # hanya 4-5 digit
    if len(text) < 4 or len(text) > 5:
        return ""

    return text