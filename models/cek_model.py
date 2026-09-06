from ultralytics import YOLO

model = YOLO("best_v8n.pt")

if hasattr(model, 'ckpt') and model.ckpt is not None:
    train_args = model.ckpt.get('train_args', None)
    if train_args:
        print("Dataset yang dipakai training:", train_args.get('data', 'tidak diketahui'))
        print("Epoch:", train_args.get('epochs', 'tidak diketahui'))
        print("Pretrained awal:", train_args.get('model', 'tidak diketahui'))
    else:
        print("Tidak ada train_args tersimpan di checkpoint")