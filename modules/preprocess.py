import cv2

def preprocess_image(crop):
    if crop is None or crop.size == 0:
        return None

    # convert grayscale
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # perbesar gambar
    gray = cv2.resize(gray, None, fx=3, fy=3)

    # Histogram equalization dihilangkan berdasarkan hasil pengujian
    # komparatif: tahap ini memperkuat noise dan teks sponsor pada bib,
    # sehingga mengganggu pemisahan digit pada Otsu thresholding.
    # Akurasi pembacaan OCR naik dari 30,6% menjadi 45,4% tanpa tahap ini.

    # blur ringan
    blur = cv2.GaussianBlur(gray, (3,3), 0)

    # threshold OTSU
    _, thresh = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return thresh