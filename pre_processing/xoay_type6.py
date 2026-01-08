import os
import cv2
import numpy as np
from tqdm import tqdm
import albumentations as A

# Đường dẫn thư mục ảnh gốc tone_6
INPUT_DIR = "D:/KLTN/FINAL_SKINTONE/public/dataset_14/type_3_filtered/Type_3"
OUTPUT_DIR ="D:/KLTN/FINAL_SKINTONE/public/dataset_14/augmented_dataset/Type_3"
TARGET_TOTAL = 1050

# Tạo output nếu chưa có
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Bộ augment
augmentations = [
    A.HorizontalFlip(p=1),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1),
    A.GaussianBlur(blur_limit=(3, 5), p=1),
    A.Rotate(limit=5, p=1),
]

# Đọc toàn bộ ảnh gốc
image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
current_total = len(image_files)
augment_needed = TARGET_TOTAL - current_total

print(f" Số ảnh gốc: {current_total}")
print(f"Cần augment thêm: {augment_needed}")

aug_count = 0
idx = 0

with tqdm(total=augment_needed) as pbar:
    while aug_count < augment_needed:
        file_name = image_files[idx % current_total]
        img_path = os.path.join(INPUT_DIR, file_name)
        img = cv2.imread(img_path)

        for aug in augmentations:
            if aug_count >= augment_needed:
                break

            transformed = aug(image=img)["image"]
            out_name = f"{os.path.splitext(file_name)[0]}_aug{aug_count}.jpg"
            cv2.imwrite(os.path.join(OUTPUT_DIR, out_name), transformed)
            aug_count += 1
            pbar.update(1)

        idx += 1

print(" Augmentation hoàn tất!")
