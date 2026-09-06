import cv2
import numpy as np


def correct_rotation(crop, max_angle=25):
    """
    Mendeteksi kemiringan region bib dan meluruskannya sebelum diproses
    lebih lanjut. Ditempatkan SEBELUM tahap preprocessing yang sudah ada
    (resize 2x, grayscale, dst) -- beroperasi pada citra crop berwarna asli.

    Parameter
    ---------
    crop : numpy.ndarray
        Citra hasil crop region bib (berwarna, BGR), sebelum preprocessing lain.
    max_angle : float
        Batas maksimal sudut koreksi (derajat). Jika sudut yang terdeteksi
        melebihi batas ini, dianggap noise/salah deteksi dan crop dikembalikan
        tanpa koreksi (mencegah rotasi yang salah pada kasus ambigu).

    Return
    ------
    numpy.ndarray
        Citra crop yang sudah diluruskan (atau citra asli jika tidak ada
        koreksi yang perlu/aman dilakukan).
    """
    if crop is None or crop.size == 0:
        return crop

    try:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # Threshold untuk memisahkan area bib (foreground) dari latar
        _, thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        coords = cv2.findNonZero(thresh)
        if coords is None or len(coords) < 10:
            # Tidak cukup titik untuk menghitung orientasi secara andal
            return crop

        rect = cv2.minAreaRect(coords)
        angle = rect[-1]

        # Normalisasi sudut (OpenCV minAreaRect mengembalikan rentang
        # yang tidak selalu intuitif, perlu disesuaikan)
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        # Jika sudut terlalu ekstrem, kemungkinan besar deteksi kontur
        # keliru (misalnya menangkap bagian pakaian, bukan bib itu sendiri)
        # -- lebih aman tidak melakukan koreksi daripada merotasi secara keliru
        if abs(angle) > max_angle or abs(angle) < 0.5:
            return crop

        (h, w) = crop.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        rotated = cv2.warpAffine(
            crop, rotation_matrix, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )

        return rotated

    except Exception:
        # Kalau terjadi error apapun saat deteksi/koreksi, kembalikan
        # crop asli -- lebih aman daripada proses berhenti total
        return crop