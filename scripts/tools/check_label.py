import cv2
import os

image_folder = "../datasets/video_frames/images"
label_folder = "../datasets/video_frames/labels"

# jumlah gambar yang ingin dicek
max_check = 20  

count = 0

for image_name in os.listdir(image_folder):
    if image_name.endswith(".jpg"):
        img_path = os.path.join(image_folder, image_name)
        label_path = os.path.join(label_folder, image_name.replace(".jpg", ".txt"))

        img = cv2.imread(img_path)
        h, w, _ = img.shape

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    cls, x, y, bw, bh = map(float, line.split())

                    # convert YOLO → pixel
                    x1 = int((x - bw/2) * w)
                    y1 = int((y - bh/2) * h)
                    x2 = int((x + bw/2) * w)
                    y2 = int((y + bh/2) * h)

                    cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)

        cv2.imshow("Check Label", img)

        key = cv2.waitKey(0)

        # tekan q untuk keluar
        if key == ord('q'):
            break

        count += 1
        if count >= max_check:
            break

cv2.destroyAllWindows()