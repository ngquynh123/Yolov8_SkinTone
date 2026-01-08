from ultralytics import YOLO
import cv2
import os

# Load model YOLOv8 đã huấn luyện
model = YOLO("runs/detect/cheek_chin_yolov8_skin/weights/best.pt")

# Thư mục gốc chứa các nhóm tone da
input_root = "public/data_skintone"
output_crop_root = "pre_processing/data_12/output_crop_batch_face_skin"
output_bbox_root = "pre_processing/data_12/output_bbox_visualize"
os.makedirs(output_crop_root, exist_ok=True)
os.makedirs(output_bbox_root, exist_ok=True)

# Shrink tỷ lệ riêng theo từng vùng
shrink_ratio_map = {
    "left": 0.2,
    "right": 0.2,
    "chin": 0.1
}

# Log kết quả (tùy chọn)
log_file = open("log_crop_result.csv", "w", encoding="utf-8")
log_file.write("filename,tone,label,x1_s,y1_s,x2_s,y2_s\n")

# Duyệt qua các tone da
for tone_dir in os.listdir(input_root):
    tone_path = os.path.join(input_root, tone_dir)
    if not os.path.isdir(tone_path):
        continue

    for filename in os.listdir(tone_path):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(tone_path, filename)
        image = cv2.imread(img_path)
        if image is None:
            print(f" Không đọc được ảnh: {img_path}")
            continue

        h_img, w_img, _ = image.shape
        results = model(image, conf=0.2)[0]
        best_boxes = {}

        for i, box in enumerate(results.boxes.xyxy):
            x1, y1, x2, y2 = map(int, box)
            conf = float(results.boxes.conf[i])
            cls_id = int(results.boxes.cls[i])
            raw_label = model.names[cls_id]
            cx = (x1 + x2) // 2

            if raw_label in ["left", "right"]:
                label = "left" if cx < w_img // 2 else "right"
            elif raw_label == "chin":
                label = "chin"
            else:
                continue

            if label not in best_boxes or conf > best_boxes[label]["conf"]:
                best_boxes[label] = {
                    "conf": conf,
                    "coords": (x1, y1, x2, y2)
                }

        image_bbox = image.copy()

        for label, box_data in best_boxes.items():
            x1, y1, x2, y2 = map(int, box_data["coords"])
            box_w = x2 - x1
            box_h = y2 - y1

            if box_w < 0.05 * w_img or box_h < 0.05 * h_img:
                print(f"⚠️ Bỏ qua box quá nhỏ trong ảnh: {filename} - {label}")
                continue

            # Dời vùng cằm xuống (theo tỷ lệ chiều cao box)
            if label == "chin":
                chin_shift = int(0.35 * box_h)
                y1 = min(y1 + chin_shift, h_img)
                y2 = min(y2 + chin_shift, h_img)

            # Shrink vùng crop
            shrink = shrink_ratio_map.get(label, 0.1)
            x1_s = max(int(x1 + shrink * box_w / 2), 0)
            y1_s = max(int(y1 + shrink * box_h / 2), 0)
            x2_s = min(int(x2 - shrink * box_w / 2), w_img)
            y2_s = min(int(y2 - shrink * box_h / 2), h_img)

            # Crop ảnh
            crop = image[y1_s:y2_s, x1_s:x2_s]

            # Kiểm tra crop có hợp lệ không
            if crop.shape[0] < 20 or crop.shape[1] < 20:
                print(f" Bỏ qua crop quá nhỏ: {filename} - {label}")
                continue

            # Lưu ảnh crop
            save_crop_dir = os.path.join(output_crop_root, tone_dir, label)
            os.makedirs(save_crop_dir, exist_ok=True)
            save_crop_path = os.path.join(save_crop_dir, os.path.splitext(filename)[0] + ".jpg")
            cv2.imwrite(save_crop_path, crop)

            # Ghi log
            log_file.write(f"{filename},{tone_dir},{label},{x1_s},{y1_s},{x2_s},{y2_s}\n")

            # Vẽ hộp
            cv2.rectangle(image_bbox, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_bbox, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Lưu ảnh bbox
        save_bbox_dir = os.path.join(output_bbox_root, tone_dir)
        os.makedirs(save_bbox_dir, exist_ok=True)
        save_bbox_path = os.path.join(save_bbox_dir, filename)
        cv2.imwrite(save_bbox_path, image_bbox)

        print(f" Đã xử lý: {tone_dir}/{filename}")

log_file.close()
