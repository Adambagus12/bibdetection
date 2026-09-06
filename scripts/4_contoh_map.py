"""
Skrip 4: Ekstraksi Data untuk Contoh Perhitungan Manual AP/mAP (Subbab Metrik Evaluasi)
==========================================================================================
Tujuan: Mengambil sejumlah kecil deteksi NYATA dari data validasi Roboflow
(524 gambar), diurutkan berdasarkan confidence, lalu menghitung status
TP/FP tiap deteksi terhadap ground truth pada IoU threshold 0.5. Data ini
dipakai untuk membuat CONTOH PERHITUNGAN MANUAL AP (Average Precision)
mengikuti Persamaan eq:bab2_ap di Bab 2, dengan angka nyata -- bukan
ilustrasi karangan -- sama seperti gaya IoU dan CER yang sudah ada.

Catatan: mAP asli pada Tabel bab4_model_compare dihitung Ultralytics
menggunakan seluruh 524 gambar pada berbagai ambang IoU (0.5 hingga 0.95).
Skrip ini TIDAK menghitung ulang seluruh itu (tidak realistis untuk
ditulis manual di laporan) -- ia mengambil SATU KELAS objek (bib) dari
SATU GAMBAR VALIDASI dengan beberapa deteksi, sebagai ilustrasi
langkah-demi-langkah cara kerja rumus AP, dengan hasil akhir yang
konsisten arahnya dengan angka mAP keseluruhan yang sudah dilaporkan.

Cara pakai:
  1. Sesuaikan DATASET_YAML dan MODEL_PATH.
  2. Jalankan: python 4_contoh_map.py
  3. Skrip akan mencari gambar validasi dengan JUMLAH GROUND TRUTH BIB
     TERBANYAK (agar contoh AP lebih kaya, tidak cuma 1-2 objek), lalu
     mencetak tabel deteksi terurut confidence beserta status TP/FP,
     precision, recall, dan interpolasi AP -- tinggal disalin ke laporan.
"""

import os
from ultralytics import YOLO

# ==== SESUAIKAN INI ====
DATASET_YAML = r"D:\bib - detection\datasets\final_dataset\data.yaml"
MODEL_PATH = r"D:\bib - detection\models\best_v8n.pt"
IOU_THRESHOLD_EVAL = 0.5   # ambang IoU untuk menentukan TP (standar mAP@0.5)
CONF_MIN = 0.001            # ambil semua deteksi (confidence rendah pun disertakan)
# ========================

import cv2
import numpy as np
import yaml


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def yolo_to_xyxy(cx, cy, w, h, img_w, img_h):
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return [x1, y1, x2, y2]


with open(DATASET_YAML, "r") as f:
    data_cfg = yaml.safe_load(f)

base_path = data_cfg.get("path", os.path.dirname(DATASET_YAML))
val_images_dir = os.path.join(base_path, data_cfg["val"])
val_labels_dir = val_images_dir.replace("images", "labels")

model = YOLO(MODEL_PATH)

# Cari gambar validasi dengan jumlah ground truth bib terbanyak
best_image = None
best_gt_count = 0
label_files = [f for f in os.listdir(val_labels_dir) if f.endswith(".txt")]

for lf in label_files:
    with open(os.path.join(val_labels_dir, lf), "r") as f:
        lines = [l for l in f.readlines() if l.strip()]
    if len(lines) > best_gt_count:
        best_gt_count = len(lines)
        best_image = lf.replace(".txt", "")

print(f"Gambar terpilih (GT terbanyak): {best_image} ({best_gt_count} objek bib)")

# Cari file gambar yang sesuai (ekstensi bisa .jpg/.png)
img_path = None
for ext in [".jpg", ".jpeg", ".png"]:
    candidate = os.path.join(val_images_dir, best_image + ext)
    if os.path.exists(candidate):
        img_path = candidate
        break

if img_path is None:
    raise FileNotFoundError(f"Gambar untuk {best_image} tidak ditemukan di {val_images_dir}")

img = cv2.imread(img_path)
img_h, img_w = img.shape[:2]

# Baca ground truth
gt_boxes = []
with open(os.path.join(val_labels_dir, best_image + ".txt"), "r") as f:
    for line in f.readlines():
        if not line.strip():
            continue
        parts = line.strip().split()
        cx, cy, w, h = map(float, parts[1:5])
        gt_boxes.append(yolo_to_xyxy(cx, cy, w, h, img_w, img_h))

# Jalankan deteksi dengan confidence sangat rendah supaya semua kandidat muncul
results = model(img_path, conf=CONF_MIN, iou=0.7, verbose=False)
boxes = results[0].boxes

detections = []
for box in boxes:
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    conf = float(box.conf[0])
    detections.append({"box": [x1, y1, x2, y2], "conf": conf})

# Urutkan deteksi berdasarkan confidence menurun (syarat utama perhitungan AP)
detections.sort(key=lambda d: d["conf"], reverse=True)

# Tentukan TP/FP: setiap GT hanya boleh dipasangkan sekali (greedy, highest conf first)
gt_matched = [False] * len(gt_boxes)
tp_list = []
fp_list = []

for det in detections:
    best_iou = 0
    best_gt_idx = -1
    for i, gt in enumerate(gt_boxes):
        if gt_matched[i]:
            continue
        iou = compute_iou(det["box"], gt)
        if iou > best_iou:
            best_iou = iou
            best_gt_idx = i

    if best_iou >= IOU_THRESHOLD_EVAL and best_gt_idx != -1:
        tp_list.append(1)
        fp_list.append(0)
        gt_matched[best_gt_idx] = True
    else:
        tp_list.append(0)
        fp_list.append(1)

# Hitung precision & recall kumulatif tiap deteksi (urut confidence menurun)
total_gt = len(gt_boxes)
cum_tp, cum_fp = 0, 0
print(f"\n{'No':<4}{'Conf':<8}{'IoU>=0.5?':<11}{'CumTP':<7}{'CumFP':<7}{'Precision':<11}{'Recall':<8}")
rows = []
for idx, (det, tp, fp) in enumerate(zip(detections, tp_list, fp_list), start=1):
    cum_tp += tp
    cum_fp += fp
    precision = cum_tp / (cum_tp + cum_fp)
    recall = cum_tp / total_gt if total_gt > 0 else 0
    status = "TP" if tp == 1 else "FP"
    rows.append((idx, det["conf"], status, cum_tp, cum_fp, precision, recall))
    print(f"{idx:<4}{det['conf']:<8.4f}{status:<11}{cum_tp:<7}{cum_fp:<7}{precision:<11.4f}{recall:<8.4f}")

# Hitung AP dengan interpolasi 11-titik sederhana (mengikuti pola Persamaan eq:bab2_ap)
recalls = [r[6] for r in rows]
precisions = [r[5] for r in rows]

# Precision interpolated: p_interp(r) = max precision untuk recall >= r
recall_levels = np.linspace(0, 1, 11)
ap = 0
print(f"\n--- Interpolasi 11 titik recall (untuk Persamaan AP) ---")
for r_level in recall_levels:
    p_interp = max([p for p, r in zip(precisions, recalls) if r >= r_level], default=0)
    ap += p_interp / 11
    print(f"Recall >= {r_level:.1f}: Precision maks = {p_interp:.4f}")

print(f"\nAP (interpolasi 11 titik) untuk gambar {best_image} = {ap:.4f} ({ap*100:.2f}%)")
print(f"\nCatatan: Karena hanya 1 kelas (bib) dan 1 gambar dipakai sebagai contoh,")
print(f"mAP = AP kelas ini = {ap*100:.2f}% -- gunakan angka ini sebagai ILUSTRASI")
print(f"cara kerja rumus, sertakan nama file gambar ({best_image}) sebagai sumber data.")