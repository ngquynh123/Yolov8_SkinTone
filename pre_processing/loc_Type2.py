import os
import cv2
import numpy as np
from skimage.color import rgb2lab, deltaE_cie76
from tqdm import tqdm
from PIL import Image

# ========== Cấu hình ========== #
SAMPLE_DIR = "public/skin tone values/Type_2"            # Thư mục chứa nhiều ảnh mẫu tone 2
INPUT_DIR = "public/dataset_14/dataset_cheeks_skin/Type_2"   # Ảnh đã gán nhãn là Type_2
OUTPUT_DIR = "public/dataset_14/type_2_filtered/Type_2"      # Thư mục lưu ảnh sau khi lọc
THRESHOLD = 21.5                                            # Ngưỡng độ chênh lệch màu

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== Hàm tính trung bình màu LAB của ảnh ========== #
def compute_avg_lab(img_path):
    img = Image.open(img_path).convert("RGB").resize((100, 100))
    img_np = np.array(img) / 255.0
    lab = rgb2lab(img_np)
    avg_lab = np.mean(lab.reshape(-1, 3), axis=0)
    return avg_lab

# ========== Tính trung bình màu LAB của các ảnh mẫu tone 2 ========== #
print(" Đang tính trung bình Lab từ ảnh mẫu tone 2...")
sample_files = [os.path.join(SAMPLE_DIR, f) for f in os.listdir(SAMPLE_DIR)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
sample_lab_values = [compute_avg_lab(f) for f in sample_files]
avg_sample_lab = np.mean(sample_lab_values, axis=0)

#  In ra kết quả trung bình LAB của mẫu tone 2
print(f" Trung bình LAB của tone 2: L={avg_sample_lab[0]:.2f}, A={avg_sample_lab[1]:.2f}, B={avg_sample_lab[2]:.2f}")

# ========== Lọc ảnh trong thư mục Type_2 ========== #
print(" Đang lọc ảnh gần với tone 2...")
image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

for fname in tqdm(image_files):
    img_path = os.path.join(INPUT_DIR, fname)
    try:
        img_lab = compute_avg_lab(img_path)
        distance = deltaE_cie76(avg_sample_lab, img_lab)

        if distance < THRESHOLD:
            dst_path = os.path.join(OUTPUT_DIR, fname)
            cv2.imwrite(dst_path, cv2.imread(img_path))
    except Exception as e:
        print(f"Lỗi với ảnh {fname}: {e}")

print(f"\n Hoàn tất lọc tone 2. Kết quả: {len(os.listdir(OUTPUT_DIR))} ảnh tại {OUTPUT_DIR}")
