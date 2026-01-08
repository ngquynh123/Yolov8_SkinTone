import os
import cv2
import numpy as np
from skimage.color import rgb2lab, deltaE_cie76
from tqdm import tqdm
from PIL import Image

# ========== Cấu hình ==========
SAMPLE_DIR = "public/skin tone values/Type_4"            # Ảnh mẫu tone 4
INPUT_DIR = "public/dataset_14/dataset_cheeks_skin/Type_4"   # Ảnh đã gán nhãn Type_4
OUTPUT_DIR = "public/dataset_14/type_4_filtered/Type_4"      # Kết quả lọc
THRESHOLD = 23                                            # Độ nới tone (càng lớn lọc càng rộng)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== Hàm tính trung bình Lab ==========
def compute_avg_lab(img_path):
    img = Image.open(img_path).convert("RGB").resize((100, 100))
    img_np = np.array(img) / 255.0
    lab = rgb2lab(img_np)
    avg_lab = np.mean(lab.reshape(-1, 3), axis=0)
    return avg_lab

# ========== Lấy trung bình Lab của ảnh mẫu ==========
print("📥 Đang tính trung bình Lab từ ảnh mẫu tone 4...")
sample_files = [os.path.join(SAMPLE_DIR, f) for f in os.listdir(SAMPLE_DIR)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
sample_lab_values = [compute_avg_lab(f) for f in sample_files]
avg_sample_lab = np.mean(sample_lab_values, axis=0)

# ========== Lọc ảnh trong thư mục Type_4 ==========
print("🔍 Đang lọc ảnh gần với tone 4...")
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

print(f"\n✅ Hoàn tất lọc tone 4. Kết quả: {len(os.listdir(OUTPUT_DIR))} ảnh tại {OUTPUT_DIR}")
