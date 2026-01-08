import os
import cv2
import numpy as np
from tqdm import tqdm
from skimage.color import rgb2lab

# ==== CẤU HÌNH ====
INPUT_FOLDER = "D:/KLTN/FINAL_SKINTONE/public/data_3/dataset_cheeks_skin/Type_4"          # Thư mục chứa ảnh đầu vào
OUTPUT_FOLDER = "public/data_4/Type_4"  # Thư mục lưu ảnh thuộc tone Type_4

# Tạo thư mục đầu ra nếu chưa tồn tại
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==== NGƯỠNG LAB SIẾT CHẶT CHO TONE 4 ====
def is_type4(lab_color):
    l, a, b = lab_color
    return (
       59.5 <= l <= 63.5 and
        5.8 <= a <= 8.0 and
        24.5 <= b <= 29.0
    )

# ==== HÀM TÍNH GIÁ TRỊ LAB TRUNG BÌNH ====
def calculate_average_lab(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_lab = rgb2lab(image_rgb)
    return np.mean(image_lab.reshape(-1, 3), axis=0)

# ==== LỌC TONE 4 ====
total = 0
kept = 0

for filename in tqdm(os.listdir(INPUT_FOLDER)):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img_path = os.path.join(INPUT_FOLDER, filename)
    image = cv2.imread(img_path)

    if image is None:
        continue

    avg_lab = calculate_average_lab(image)

    if is_type4(avg_lab):
        out_path = os.path.join(OUTPUT_FOLDER, filename)
        cv2.imwrite(out_path, image)
        kept += 1

    total += 1

print(f"\n🔍 Đã kiểm tra {total} ảnh.")
print(f"✅ Lưu {kept} ảnh thuộc Tone 4 (Type_4) vào: {OUTPUT_FOLDER}")
