# yolo.py
import cv2
import numpy as np
from ultralytics import YOLO

# Khởi tạo mô hình YOLO với đường dẫn thực tế
model = YOLO("D:\\Nam3\\hocky2\\TGMT\\tgmt\\yolo_web_app\\yolov8n.pt")  # Thay bằng đường dẫn của bạn

def process_image(image_path, output_path=None):
    # Đọc ảnh
    img = cv2.imread(image_path) if isinstance(image_path, str) else image_path
    if img is None:
        raise ValueError("Không thể đọc ảnh từ đường dẫn hoặc dữ liệu đầu vào.")

    # Thực hiện nhận diện
    results = model(img)
    
    # Lấy danh sách đối tượng được phát hiện
    detections = []
    for result in results:
        for box in result.boxes:
            label = result.names[int(box.cls[0])]
            confidence = float(box.conf[0])
            detections.append({"label": label, "confidence": confidence})

    # Lưu ảnh đã xử lý nếu có output_path
    if output_path:
        annotated_img = results[0].plot()  # Vẽ bounding boxes
        cv2.imwrite(output_path, annotated_img)

    return detections

def process_video(video_path, output_path):
    # Đọc video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Không thể mở video.")

    # Lấy thông tin video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Khởi tạo VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Xử lý frame với YOLO
        results = model(frame)
        annotated_frame = results[0].plot()  # Vẽ bounding boxes
        out.write(annotated_frame)
        frame_count += 1
        print(f"Frame {frame_count}: Wrote frame to {output_path}")

    cap.release()
    out.release()
    return []  # Trả về danh sách rỗng (có thể mở rộng để lưu detections nếu cần)

if __name__ == "__main__":
    pass