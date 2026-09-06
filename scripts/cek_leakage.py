import os

# SESUAIKAN path ini ke lokasi folder dataset Roboflow Anda
BASE_PATH = r"D:\bib - detection\datasets\roboflow_dataset"

train_files = set(os.listdir(os.path.join(BASE_PATH, "train", "images")))
valid_files = set(os.listdir(os.path.join(BASE_PATH, "valid", "images")))
test_files  = set(os.listdir(os.path.join(BASE_PATH, "test", "images")))

overlap_train_valid = train_files & valid_files
overlap_train_test  = train_files & test_files
overlap_valid_test  = valid_files & test_files

print("Jumlah gambar train:", len(train_files))
print("Jumlah gambar valid:", len(valid_files))
print("Jumlah gambar test :", len(test_files))
print()
print("Overlap train-valid:", len(overlap_train_valid), overlap_train_valid)
print("Overlap train-test :", len(overlap_train_test), overlap_train_test)
print("Overlap valid-test :", len(overlap_valid_test), overlap_valid_test)

if not (overlap_train_valid or overlap_train_test or overlap_valid_test):
    print("\nAMAN - tidak ada data yang bocor antar subset.")
else:
    print("\nADA KEBOCORAN DATA - perlu penanganan lebih lanjut.")