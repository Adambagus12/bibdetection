from ultralytics import YOLO

DATA_YAML = r"D:\bib - detection\datasets\roboflow_dataset\data.yaml"

MODEL_BASE_PATH = r"D:\bib - detection\models\best_roboflow_v8n.pt"
MODEL_FINAL_PATH = r"D:\bib - detection\models\best_v8n.pt"

print("=" * 60)
print("EVALUASI MODEL BASE - TEST SET (349 gambar)")
print("=" * 60)
model_base = YOLO(MODEL_BASE_PATH)
hasil_base = model_base.val(data=DATA_YAML, split="test")
print()
print(f"Precision    : {hasil_base.box.mp*100:.2f}%")
print(f"Recall       : {hasil_base.box.mr*100:.2f}%")
print(f"mAP@0.5      : {hasil_base.box.map50*100:.2f}%")
print(f"mAP@0.5:0.95 : {hasil_base.box.map*100:.2f}%")

print()
print("=" * 60)
print("EVALUASI MODEL FINAL - TEST SET (349 gambar)")
print("=" * 60)
model_final = YOLO(MODEL_FINAL_PATH)
hasil_final = model_final.val(data=DATA_YAML, split="test")
print()
print(f"Precision    : {hasil_final.box.mp*100:.2f}%")
print(f"Recall       : {hasil_final.box.mr*100:.2f}%")
print(f"mAP@0.5      : {hasil_final.box.map50*100:.2f}%")
print(f"mAP@0.5:0.95 : {hasil_final.box.map*100:.2f}%")

print()
print("=" * 60)
print("RINGKASAN PERBANDINGAN")
print("=" * 60)
print(f"{'Metrik':<15}{'Base':<12}{'Final':<12}")
print(f"{'Precision':<15}{hasil_base.box.mp*100:<11.2f}{hasil_final.box.mp*100:<11.2f}")
print(f"{'Recall':<15}{hasil_base.box.mr*100:<11.2f}{hasil_final.box.mr*100:<11.2f}")
print(f"{'mAP@0.5':<15}{hasil_base.box.map50*100:<11.2f}{hasil_final.box.map50*100:<11.2f}")
print(f"{'mAP@0.5:0.95':<15}{hasil_base.box.map*100:<11.2f}{hasil_final.box.map*100:<11.2f}")