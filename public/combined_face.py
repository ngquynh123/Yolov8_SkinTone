import os
import cv2
import numpy as np
from tqdm import tqdm
from skimage.color import rgb2lab, deltaE_cie76

# ====== Cấu hình đường dẫn ======
input_left = "pre_processing/data_12/output_crop_batch_face_skin/mid-light/left"
input_right = "pre_processing/data_12/output_crop_batch_face_skin/mid-light/right"
input_chin = "pre_processing/data_12/output_crop_batch_face_skin/mid-light/chin"
output_combined = "pre_processing/data_12/combined_face/mid-light"

os.makedirs(output_combined, exist_ok=True)

# === Hàm tính màu trung bình trong không gian Lab ===
def get_mean_lab(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    lab = rgb2lab(img_rgb)
    return lab.mean(axis=(0, 1))

# === Hàm tính khoảng cách màu DeltaE ===
def deltaE(img1, img2):
    lab1 = get_mean_lab(img1)
    lab2 = get_mean_lab(img2)
    return deltaE_cie76(lab1, lab2)

# === Bắt đầu xử lý ảnh ===
for f in tqdm(os.listdir(input_left), desc="🔄 Đang xử lý"):
    base_name = f.replace(".jpg", "")
    path_left = os.path.join(input_left, f)
    path_right = os.path.join(input_right, f)
    path_chin = os.path.join(input_chin, f)

    img_left = cv2.imread(path_left) if os.path.exists(path_left) else None
    img_right = cv2.imread(path_right) if os.path.exists(path_right) else None
    img_chin = cv2.imread(path_chin) if os.path.exists(path_chin) else None

    if img_left is None and img_right is None:
        print(f" Không có má nào: {base_name}")
        continue
    if img_chin is None:
        print(f" Thiếu ảnh cằm: {base_name}")
        continue

    # Resize về 112x112
    if img_left is not None:
        img_left = cv2.resize(img_left, (112, 112))
    if img_right is not None:
        img_right = cv2.resize(img_right, (112, 112))
    img_chin = cv2.resize(img_chin, (112, 112))  # Giữ nguyên hướng cằm


    parts_to_concat = []

    if img_left is not None and img_right is not None:
        delta_lr = deltaE(img_left, img_right)

        if delta_lr < 10:
            delta_lc = deltaE(img_left, img_chin)
            delta_rc = deltaE(img_right, img_chin)

            if delta_lc > 10 and delta_rc > 10:
                # Bỏ cằm nếu lệch với cả hai
                parts_to_concat = [img_left, img_right]
            else:
                parts_to_concat = [img_left, img_chin, img_right]
        else:
            delta_lc = deltaE(img_left, img_chin)
            delta_rc = deltaE(img_right, img_chin)

            if delta_lc < delta_rc and delta_lc < 10:
                parts_to_concat = [img_left, img_chin]
            elif delta_rc < 10:
                parts_to_concat = [img_chin, img_right]
            else:
                parts_to_concat = [img_left] if delta_lc < delta_rc else [img_right]
    else:
        # Chỉ có 1 má
        if img_left is not None:
            delta_lc = deltaE(img_left, img_chin)
            if delta_lc < 10:
                parts_to_concat = [img_left, img_chin]
            else:
                parts_to_concat = [img_left]
        elif img_right is not None:
            delta_rc = deltaE(img_right, img_chin)
            if delta_rc < 10:
                parts_to_concat = [img_chin, img_right]
            else:
                parts_to_concat = [img_right]

    if len(parts_to_concat) < 2:
        print(f" Không đủ vùng hợp lệ để ghép: {base_name}")
        continue

    try:
        combined = cv2.hconcat(parts_to_concat)
        final_img = cv2.resize(combined, (224, 224))
        save_path = os.path.join(output_combined, base_name + "_filtered.jpg")
        cv2.imwrite(save_path, final_img)
    except Exception as e:
        print(f" Lỗi khi ghép ảnh: {base_name} | {e}")

print(" Đã hoàn tất ghép ảnh theo chiều ngang tại:", output_combined)
