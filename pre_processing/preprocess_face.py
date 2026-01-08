import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logging.getLogger('mediapipe').setLevel(logging.ERROR)
import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

def preprocess_for_yolo(image_path, output_path=None, resize_size=(640, 640), padding=40):
    image = cv2.imread(image_path)
    if image is None:
        tqdm.write(f" Không đọc được ảnh: {image_path}")
        return None

    h, w, _ = image.shape
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mp_face_mesh = mp.solutions.face_mesh
    mp_selfie_segmentation = mp.solutions.selfie_segmentation

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh, \
         mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_seg:

        face_results = face_mesh.process(img_rgb)
        if not face_results.multi_face_landmarks:
            tqdm.write(f"  Không phát hiện khuôn mặt: {image_path}")
            return None

        landmarks = face_results.multi_face_landmarks[0]

        face_contour = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397,
                        365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172,
                        58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        face_points = np.array([[int(landmarks.landmark[i].x * w),
                                 int(landmarks.landmark[i].y * h)]
                                for i in face_contour], dtype=np.int32)

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [face_points], 255)

        regions = {
            "left_eye": [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7],
            "right_eye": [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382],
            "mouth": [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314,
                      17, 84, 181, 91, 146, 61]
        }
        for region in regions.values():
            pts = np.array([[int(landmarks.landmark[i].x * w),
                             int(landmarks.landmark[i].y * h)]
                            for i in region], dtype=np.int32)
            cv2.fillPoly(mask, [pts], 0)

        seg = selfie_seg.process(img_rgb)
        seg_mask = (seg.segmentation_mask > 0.9).astype(np.uint8) * 255
        seg_mask = cv2.GaussianBlur(seg_mask, (7, 7), sigmaX=3)
        seg_mask = (seg_mask > 127).astype(np.uint8) * 255

        comb_mask = cv2.bitwise_and(mask, seg_mask)
        masked = cv2.bitwise_and(img_rgb, img_rgb, mask=comb_mask)

        gray_bg = np.full_like(img_rgb, fill_value=180)
        img_with_gray_bg = np.where(comb_mask[:, :, None] == 255, masked, gray_bg)

        x0, y0 = face_points.min(axis=0)
        x1, y1 = face_points.max(axis=0)
        x0 = max(x0 - padding, 0)
        y0 = max(y0 - padding, 0)
        x1 = min(x1 + padding, w)
        y1 = min(y1 + padding, h)
        face_crop = img_with_gray_bg[y0:y1, x0:x1]

        if face_crop.size == 0:
            tqdm.write(f"❌ Lỗi crop khuôn mặt: {image_path}")
            return None

        resized_face = cv2.resize(face_crop, resize_size)

        if output_path:
            cv2.imwrite(output_path, cv2.cvtColor(resized_face, cv2.COLOR_RGB2BGR))

        return resized_face

def preprocess_directory(input_root_dir, output_root_dir):
    categories = ['light', 'mid-light', 'mid-dark', 'dark']

    for category in categories:
        input_dir = os.path.join(input_root_dir, category)
        output_dir = os.path.join(output_root_dir, category)
        os.makedirs(output_dir, exist_ok=True)

        image_files = [f for f in os.listdir(input_dir)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        print(f"\n Đang xử lý thư mục: {category} ({len(image_files)} ảnh)")
        for filename in tqdm(image_files, desc=f"📸 {category}", unit="ảnh"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            processed = preprocess_for_yolo(input_path, output_path)
            if processed is None:
                tqdm.write(f" Bỏ qua ảnh lỗi: {filename}")

# === GỌI HÀM ===
input_root = "D:/KLTN/FINAL_SKINTONE/public/data_skintone"
output_root = "D:/KLTN/FINAL_SKINTONE/pre_processing/face_yolo"

preprocess_directory(input_root, output_root)
