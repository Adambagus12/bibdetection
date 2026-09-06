import re
import easyocr

# Inisialisasi EasyOCR — hanya sekali saat import
_reader = easyocr.Reader(['en'], gpu=False, verbose=False)

def read_text(image):
    if image is None:
        return ""

    try:
        result = _reader.readtext(image, detail=0, allowlist='0123456789')

        if not result:
            return ""

        # Gabungkan semua teks yang terdeteksi
        combined = "".join(result)

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