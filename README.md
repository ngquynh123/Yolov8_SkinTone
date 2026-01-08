# Phát Hiện và Phân Loại Sắc Độ Da Sử Dụng YOLOv8

Một dự án machine learning toàn diện để phát hiện và phân loại sắc độ da sử dụng mô hình YOLOv8 phát hiện đối tượng và MobileNetV2 phân loại. Dự án này bao gồm phát hiện khuôn mặt, trích xuất vùng da và phân loại sắc độ da đa danh mục (tối, sáng, trung bình tối, trung bình sáng).

## 📋 Tổng Quan Dự Án

Dự án này được thiết kế để:

- **Phát hiện các vùng khuôn mặt** (mặt, má, cằm) sử dụng YOLOv8
- **Trích xuất vùng da** từ các khuôn mặt được phát hiện
- **Phân loại sắc độ da** thành nhiều danh mục sử dụng MobileNetV2
- **Xử lý và tăng cường dữ liệu** để cải thiện hiệu suất mô hình
- **Tạo dự đoán và nhật ký** để phân tích và đánh giá

## 📁 Cấu Trúc Dự Án

```
FINAL_SKINTONE/
├── crop_yolov8.py              # Cắt và trích xuất vùng da từ khuôn mặt
├── yolo.py                     # Script huấn luyện mô hình YOLOv8
├── pre_processing/
│   ├── preprocess_face.py      # Quy trình tiền xử lý khuôn mặt
│   ├── augment_images.py       # Mô-đun tăng cường dữ liệu
│   ├── train_val_test.py       # Tiện ích chia tập dữ liệu
│   ├── loc_Type2-6.py          # Scripts định vị vùng da
│   └── data_*/                 # Tập dữ liệu đã xử lý với chia train/val/test
├── public/
│   ├── mobilenetV2.py          # Huấn luyện và suy luận mô hình MobileNetV2
│   ├── combined_face.py        # Xử lý khuôn mặt kết hợp
│   ├── tone_labeler.py         # Tiện ích gán nhãn sắc độ da
│   ├── check_yolov8.ipynb      # Notebook đánh giá YOLOv8
│   └── data_skintone/          # Tập dữ liệu sắc độ da được tổ chức
├── models/
│   ├── yolov8n.pt             # Mô hình YOLOv8 nano được huấn luyện trước
│   ├── mobilenetv2_*.pth      # Mô hình MobileNetV2 đã huấn luyện (5 biến thể)
│   └── runs/                   # Kết quả huấn luyện và trọng số
├── notebooks/
│   ├── test.ipynb             # Notebook kiểm tra và xác thực
│   ├── yolo8.ipynb            # Notebook huấn luyện YOLOv8
│   └── yolov8_seg.ipynb       # Notebook phân đoạn YOLOv8
└── test_images*/              # Các thư mục ảnh kiểm tra
```

## 🎯 Các Thành Phần Chính

### 1. **Phát Hiện Vùng Da YOLOv8** (`crop_yolov8.py`)

- Phát hiện các vùng khuôn mặt cụ thể: má trái, má phải, cằm
- Trích xuất các vùng được cắt với tỷ lệ co lại có thể cấu hình
- Ghi nhật ký kết quả vào CSV để theo dõi
- Hỗ trợ nhiều danh mục sắc độ da

### 2. **Tiền Xử Lý Dữ Liệu** (`pre_processing/`)

- **preprocess_face.py**: Chuẩn hóa hình ảnh khuôn mặt cho đầu vào mô hình
- **augment_images.py**: Áp dụng các kỹ thuật tăng cường dữ liệu
- **train_val_test.py**: Chia tập dữ liệu thành tập train/validation/test
- **loc_Type2-6.py**: Định vị chuyên biệt cho các vùng khuôn mặt khác nhau

### 3. **Phân Loại MobileNetV2** (`public/mobilenetV2.py`)

- CNN nhẹ để phân loại sắc độ da
- Huấn luyện trên nhiều danh mục sắc độ da
- 5 biến thể mô hình khác nhau có sẵn
- Triển khai học chuyển giao

## 🏷️ Danh Mục Sắc Độ Da

Dự án phân loại sắc độ da thành 4 danh mục chính:

- **Tối**: Sắc độ da tối hơn
- **Sáng**: Sắc độ da sáng hơn
- **Trung Bình Tối**: Sắc độ da trung bình tối
- **Trung Bình Sáng**: Sắc độ da trung bình sáng

Đường dẫn tập dữ liệu: `public/data_skintone/` và `public/dataset_*/`

## 📦 Mô Hình Bao Gồm

### Mô Hình YOLOv8

- **yolov8n.pt**: Mô hình nano (nhẹ, nhanh)
- Mô hình phát hiện được huấn luyện trước trong `runs/detect/`

### Mô Hình MobileNetV2

Nhiều biến thể đã huấn luyện:

- `mobilenetv2_best_yolov8_tuned_1.pth`
- `mobilenetv2_best_yolov8_tuned_2.pth`
- `mobilenetv2_best_yolov8_tuned_3.pth`
- `mobilenetv2_best_yolov8_tuned_4.pth`
- `mobilenetv2_best_yolov8_tuned_5.pth`
- `mobilenetv2_best_yolov8_unfreeze.pth`

## 🚀 Bắt Đầu Nhanh

### Các Yêu Cầu Tiên Quyết

```bash
pip install ultralytics opencv-python pytorch torchvision numpy pandas albumentations tqdm
```

### 1. Phát Hiện Vùng Da YOLOv8

```bash
python crop_yolov8.py
```

### 3. Huấn Luyện Mô Hình YOLOv8

```bash
python yolo.py
```

### 2. Huấn Luyện/Đánh Giá MobileNetV2

```bash
python public/mobilenetV2.py
```

### 4. Chạy Notebook để Phân Tích Tương Tác

```bash
jupyter notebook test.ipynb
jupyter notebook yolo8.ipynb
jupyter notebook yolov8_seg.ipynb
```

## 📊 Tệp Đầu Ra

- **log_crop_result.csv**: Nhật ký các vùng được cắt với tọa độ
- **test_predictions_albu.csv**: Dự đoán mô hình với kết quả tăng cường
- **runs/detect/**: Kết quả phát hiện YOLOv8 và hình ảnh trực quan hóa
- **runs/segment/**: Kết quả phân đoạn YOLOv8
- Tập dữ liệu đã xử lý trong `pre_processing/data_*/output_crop_batch_face_skin/`

## 🔧 Cấu Hình & Thông Số

Các thông số chính có thể được điều chỉnh trong các scripts:

**Trong `crop_yolov8.py`:**

```python
shrink_ratio_map = {
    "left": 0.2,      # Tỷ lệ co lại cho má trái
    "right": 0.2,     # Tỷ lệ co lại cho má phải
    "chin": 0.1       # Tỷ lệ co lại cho cằm
}
conf = 0.2           # Ngưỡng độ tin cậy
```

**Trong `preprocess_face.py`:**

```python
resize_size = (640, 640)  # Kích thước ảnh đầu ra
padding = 40              # Khoảng đệm xung quanh vùng khuôn mặt
```

## 📈 Số Liệu Hiệu Suất

Kết quả được lưu trữ trong:

- Nhật ký CSV với tọa độ phát hiện và điểm số độ tin cậy
- Dự đoán mô hình được lưu trong `test_predictions_albu.csv`
- Hình ảnh trực quan hóa trong các thư mục đầu ra

## 🔄 Quy Trình Công Việc

```
Ảnh Thô
    ↓
[Phát Hiện Khuôn Mặt] (MediaPipe/YOLOv8)
    ↓
[Trích Xuất & Tiền Xử Lý Khuôn Mặt]
    ↓
[Định Vị Vùng Da] (YOLOv8)
    ↓
[Phân Loại Sắc Độ Da] (MobileNetV2)
    ↓
Kết Quả & Nhật Ký (CSV, Hình Ảnh)
```

## 📝 Tăng Cường Dữ Liệu

Dự án hỗ trợ nhiều kỹ thuật tăng cường:

- Xoay, lật, điều chỉnh độ sáng
- Tích hợp thư viện Albumentations
- Quy trình tăng cường tự động trong tiền xử lý

Xem `pre_processing/augment_images.py` để biết chi tiết.

## 🛠️ Ghi Chú Phát Triển

- **Ngôn Ngữ**: Nhận xét tiếng Việt trong code
- **Framework**: PyTorch để học sâu
- **Phát Hiện**: Ultralytics YOLOv8
- **Tăng Cường**: Thư viện Albumentations

## 📚 Jupyter Notebook

- **test.ipynb**: Kiểm tra và đánh giá chính
- **yolo8.ipynb**: Huấn luyện và trực quan hóa YOLOv8
- **yolov8_seg.ipynb**: Thử nghiệm phân đoạn cá thể
- **public/check_yolov8.ipynb**: Kiểm tra YOLOv8 bổ sung

## ⚠️ Lưu Ý

- Đảm bảo hình ảnh ở định dạng JPG/JPEG/PNG
- Điều chỉnh đường dẫn đầu vào/đầu ra theo hệ thống của bạn
- GPU được khuyến nghị để xử lý nhanh hơn (hỗ trợ CUDA)
- Mô hình yêu cầu đủ VRAM để xử lý batch

## 📄 Giấy Phép

Dự án này được phát triển cho mục đích phân loại và phân tích sắc độ da.

## 👥 Đóng Góp

Không gian làm việc dự án: `d:\FINAL_SKINTONE_YOLOv8\FINAL_SKINTONE\`

---

Nếu có câu hỏi hoặc vấn đề, vui lòng tham khảo các nhận xét trong script riêng lẻ hoặc các Jupyter notebook để xem ví dụ chi tiết.
