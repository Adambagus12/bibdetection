import cv2

def preprocess_image(crop):
    if crop is None or crop.size == 0:
        return None

    # convert grayscale
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # perbesar gambar
    gray = cv2.resize(gray, None, fx=3, fy=3)

    # tingkatkan kontras
    gray = cv2.equalizeHist(gray)

    # blur ringan
    blur = cv2.GaussianBlur(gray, (3,3), 0)

    # threshold OTSU
    _, thresh = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return thresh