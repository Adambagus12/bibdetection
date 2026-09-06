import cv2
import numpy as np


def sharpen_image(image, amount=1.5, radius=1.0):
    """
    Menerapkan teknik unsharp masking untuk menajamkan citra yang
    terdampak motion blur ringan, sebelum tahap preprocessing lain
    dijalankan.

    Cara kerja: citra asli dikurangi versi citra yang diperhalus
    (Gaussian blur dengan radius kecil), lalu selisihnya (detail
    tepi/tekstur) dikuatkan kembali ke citra asli. Ini BERBEDA dari
    Gaussian Blur pada preprocessing yang sudah ada -- Gaussian Blur
    di preprocessing MENGHALUSKAN citra untuk redam noise sebelum
    Otsu thresholding, sedangkan fungsi ini MENAJAMKAN citra untuk
    memperjelas tepi digit yang buram akibat motion blur.

    Parameter
    ---------
    image : numpy.ndarray
        Citra crop bib (bisa berwarna atau grayscale).
    amount : float
        Kekuatan penajaman. Semakin besar, semakin agresif penajamannya.
    radius : float
        Radius Gaussian blur internal yang dipakai untuk menghitung
        selisih ketajaman.

    Return
    ------
    numpy.ndarray
        Citra yang sudah ditajamkan.
    """
    if image is None or image.size == 0:
        return image

    try:
        blurred = cv2.GaussianBlur(image, (0, 0), radius)
        sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
        return sharpened
    except Exception:
        return image