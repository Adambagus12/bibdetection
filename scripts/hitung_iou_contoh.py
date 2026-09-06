"""
Skrip untuk menghasilkan contoh perhitungan IoU (worked example) untuk Bab 4.
VERSI 2 - otomatis membaca ukuran gambar dan mengonversi label YOLO (normalized)
ke koordinat piksel yang benar, sehingga tidak perlu hardcode ukuran gambar.

CARA PAKAI:
1. Sesuaikan MODEL_PATH, IMAGE_PATH, dan LABEL_PATH di bawah
2. Jalankan: python hitung_iou_contoh.py
3. Salin output ke Bab 4 sebagai worked example IoU
"""

from ultralytics import YOLO
from PIL import Image

# ======== SESUAIKAN TIGA BARIS INI ========
MODEL_PATH = r"D:\bib - detection\models\best_v8n.pt"
IMAGE_PATH = r"D:\bib - detection\datasets\roboflow_dataset\valid\images\0d1d7036-good_484A4976_JPG.rf.fca1b680dd410384ce61face856de1c1.jpg"
LABEL_PATH = r"D:\bib - detection\datasets\roboflow_dataset\valid\labels\0d1d7036-good_484A4976_JPG.rf.fca1b680dd410384ce61face856de1c1.txt"
# ============================================


def baca_label_yolo_ke_piksel(label_path, lebar_gambar, tinggi_gambar):
    """Baca file label YOLO (format: class x_center y_center width height,
    semua dinormalisasi 0-1) dan konversi ke piksel (x1,y1,x2,y2)."""
    kotak_kotak = []
    with open(label_path, "r") as f:
        for baris in f:
            baris = baris.strip()
            if not baris:
                continue
            bagian = baris.split()
            kelas = int(bagian[0])
            x_center_norm, y_center_norm, w_norm, h_norm = map(float, bagian[1:5])

            x_center_px = x_center_norm * lebar_gambar
            y_center_px = y_center_norm * tinggi_gambar
            w_px = w_norm * lebar_gambar
            h_px = h_norm * tinggi_gambar

            x1 = x_center_px - w_px / 2
            y1 = y_center_px - h_px / 2
            x2 = x_center_px + w_px / 2
            y2 = y_center_px + h_px / 2

            kotak_kotak.append({
                "kelas": kelas, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "x_center_norm": x_center_norm, "y_center_norm": y_center_norm,
                "w_norm": w_norm, "h_norm": h_norm,
            })
    return kotak_kotak


def hitung_iou(box_a, box_b):
    """Hitung IoU antara dua bounding box dalam format dict {x1,y1,x2,y2}."""
    x_kiri = max(box_a["x1"], box_b["x1"])
    y_atas = max(box_a["y1"], box_b["y1"])
    x_kanan = min(box_a["x2"], box_b["x2"])
    y_bawah = min(box_a["y2"], box_b["y2"])

    if x_kanan <= x_kiri or y_bawah <= y_atas:
        area_overlap = 0.0
    else:
        area_overlap = (x_kanan - x_kiri) * (y_bawah - y_atas)

    area_a = (box_a["x2"] - box_a["x1"]) * (box_a["y2"] - box_a["y1"])
    area_b = (box_b["x2"] - box_b["x1"]) * (box_b["y2"] - box_b["y1"])
    area_union = area_a + area_b - area_overlap
    iou = area_overlap / area_union if area_union > 0 else 0.0

    return {
        "x_kiri": x_kiri, "y_atas": y_atas, "x_kanan": x_kanan, "y_bawah": y_bawah,
        "area_overlap": area_overlap, "area_a": area_a, "area_b": area_b,
        "area_union": area_union, "iou": iou,
    }


def main():
    # 1. Cek ukuran gambar asli
    img = Image.open(IMAGE_PATH)
    lebar, tinggi = img.size
    print(f"Ukuran gambar asli: {lebar} x {tinggi} piksel")
    print()

    # 2. Baca ground truth dari file label, konversi ke piksel pakai ukuran asli
    gt_list = baca_label_yolo_ke_piksel(LABEL_PATH, lebar, tinggi)
    print("=" * 60)
    print("GROUND TRUTH (dari file label, dikonversi ke piksel):")
    for i, gt in enumerate(gt_list):
        print(f"  GT #{i+1}: normalized(x_c={gt['x_center_norm']:.6f}, "
              f"y_c={gt['y_center_norm']:.6f}, w={gt['w_norm']:.6f}, h={gt['h_norm']:.6f})")
        print(f"          -> piksel: x1={gt['x1']:.2f}, y1={gt['y1']:.2f}, "
              f"x2={gt['x2']:.2f}, y2={gt['y2']:.2f}")
        print(f"          -> luas GT = {(gt['x2']-gt['x1'])*(gt['y2']-gt['y1']):.2f} piksel^2")
    print()

    # 3. Jalankan deteksi
    model = YOLO(MODEL_PATH)
    results = model(IMAGE_PATH, conf=0.45, iou=0.45)
    boxes = results[0].boxes

    if len(boxes) == 0:
        print("Model tidak mendeteksi bib apapun pada gambar ini dengan conf=0.45.")
        return

    print("=" * 60)
    print("HASIL DETEKSI & PERHITUNGAN IoU TERHADAP GROUND TRUTH:")
    print()

    for i, box in enumerate(boxes):
        xyxy = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        pred_box = {"x1": xyxy[0], "y1": xyxy[1], "x2": xyxy[2], "y2": xyxy[3]}

        print(f"--- Deteksi #{i+1} (confidence={conf:.4f}) ---")
        print(f"  Predicted box: x1={pred_box['x1']:.2f}, y1={pred_box['y1']:.2f}, "
              f"x2={pred_box['x2']:.2f}, y2={pred_box['y2']:.2f}")

        for j, gt in enumerate(gt_list):
            hasil = hitung_iou(pred_box, gt)
            print(f"  -> vs GT #{j+1}:")
            print(f"     Luas predicted = {hasil['area_a']:.2f} piksel^2, "
                  f"Luas GT = {hasil['area_b']:.2f} piksel^2")
            print(f"     Titik potong (overlap): x=[{hasil['x_kiri']:.2f}, {hasil['x_kanan']:.2f}], "
                  f"y=[{hasil['y_atas']:.2f}, {hasil['y_bawah']:.2f}]")
            print(f"     Luas overlap = {hasil['area_overlap']:.2f} piksel^2")
            print(f"     Luas union = {hasil['area_a']:.2f} + {hasil['area_b']:.2f} - "
                  f"{hasil['area_overlap']:.2f} = {hasil['area_union']:.2f} piksel^2")
            print(f"     IoU = {hasil['area_overlap']:.2f} / {hasil['area_union']:.2f} = "
                  f"{hasil['iou']:.4f}")
            status = "TP (valid, IoU > 0.45)" if hasil["iou"] > 0.45 else "IoU <= 0.45"
            print(f"     Status pada threshold IoU 0.45: {status}")
        print()

    print("=" * 60)


if __name__ == "__main__":
    main()