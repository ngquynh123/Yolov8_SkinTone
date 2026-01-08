from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # hoặc yolov8s.pt cho nhẹ

model.train(
    data='dataset_yolo/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='yolov8_face_parts',
    device='cuda'  # hoặc 'cpu'
)
