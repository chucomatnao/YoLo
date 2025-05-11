import sys
import os

# Thêm đường dẫn đến thư mục yolov5 vào sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'yolov5'))

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression
from yolov5.utils.torch_utils import select_device
import cv2
import numpy as np
import torch

def load_yolo_model():
    device = select_device('cpu')  # Sử dụng CPU, thay '0' nếu dùng GPU
    model_path = os.path.join(os.path.dirname(__file__), 'yolov5', 'yolov5s.pt')  # Đường dẫn tuyệt đối
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found. Please download yolov5s.pt and place it in the yolov5 directory.")
    model = DetectMultiBackend(model_path, device=device, dnn=False)
    model.eval()
    return model

# Load model một lần khi module được import
model = load_yolo_model()

def process_image(image, output_path=None):
    try:
        # Kiểm tra xem image là đường dẫn (string) hay mảng numpy
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise ValueError("Không thể đọc ảnh từ đường dẫn: " + image)
        elif isinstance(image, np.ndarray):
            img = image  # Sử dụng trực tiếp mảng numpy
        else:
            raise ValueError("Đầu vào không hợp lệ: image phải là đường dẫn (str) hoặc mảng numpy")

        if img is None:
            raise ValueError("Ảnh không hợp lệ")

        # Chuẩn bị ảnh cho YOLO (chuyển sang RGB, resize)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (640, 640))  # Kích thước mặc định của YOLOv5
        img_tensor = np.transpose(img_resized, (2, 0, 1))  # CHW format
        img_tensor = np.expand_dims(img_tensor, axis=0).astype(np.float32) / 255.0

        # Chuyển sang định dạng PyTorch
        img_tensor = torch.from_numpy(img_tensor).to(next(model.parameters()).device)

        # Nhận diện với YOLO
        with torch.no_grad():
            pred = model(img_tensor)
            pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)[0]

        # Chuyển đổi kết quả thành danh sách detections
        detections = []
        if pred is not None and len(pred):
            for *xyxy, conf, cls in pred:
                label = model.names[int(cls)]  # Tên lớp
                confidence = conf.item()  # Độ tin cậy
                detections.append({
                    'label': label,
                    'confidence': float(confidence)
                })

        # Nếu có output_path, lưu ảnh với bounding boxes
        if output_path and detections:
            for *xyxy, conf, cls in pred:
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"{model.names[int(cls)]}: {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imwrite(output_path, img)

        print(f"Detections: {detections}")  # Log để debug
        return detections
    except Exception as e:
        print(f"Error in process_image: {str(e)}")
        return []

if __name__ == "__main__":
    # Test hàm
    img = cv2.imread("path/to/test/image.jpg")
    detections = process_image(img, None)
    print(detections)