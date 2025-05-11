# Ứng dụng Web Nhận diện Đối tượng bằng YOLO

Ứng dụng web này sử dụng mô hình YOLO (You Only Look Once) để nhận diện đối tượng trong ảnh và video. Người dùng có thể đăng nhập, tải lên ảnh/video, chụp ảnh/ghi video từ camera, xem lịch sử kết quả, và xem chi tiết các đối tượng được phát hiện (với hộp giới hạn và độ tin cậy). Gần đây, ứng dụng đã được nâng cấp để hỗ trợ nhận diện trực tiếp từ camera và sử dụng mô hình YOLOv5 thay vì YOLOv3.

## Tính năng chính

- **Đăng nhập/Đăng ký**: Người dùng cần đăng nhập để sử dụng ứng dụng.
- **Menu slide-in**: Hiển thị sau đăng nhập, bao gồm Dashboard, Camera, History, Logout.
- **Dashboard**: Tải lên ảnh (.jpg, .png) hoặc video (.mp4, .avi, .webm) để nhận diện.
- **Camera**: 
  - Chụp ảnh hoặc ghi video trực tiếp từ webcam, sau đó nhận diện.
  - Nhận diện trực tiếp từ luồng video camera, hiển thị kết quả theo thời gian thực.
- **History**: Xem danh sách các ảnh/video đã xử lý (tên file, loại, thời gian, liên kết xem chi tiết).
- **Result**: Hiển thị ảnh/video đã nhận diện trong khung 640x480px, cùng danh sách đối tượng (ví dụ: person: 0.8744).
- **Quản lý file**: Ảnh/video được lưu vào thư mục `static/uploads` với cấu trúc:
  - `images/original`: Ảnh gốc.
  - `images/processed`: Ảnh đã nhận diện.
  - `videos/original`: Video gốc.
  - `videos/processed`: Video đã nhận diện.

## Cấu trúc thư mục và giải thích file
yolo_web_app/
├── app.py                    # File chính, chạy ứng dụng Flask, xử lý route và logic
├── yolo.py                   # Hàm xử lý nhận diện ảnh bằng YOLOv5
├── yolo_video.py             # Hàm xử lý nhận diện video bằng YOLOv5
├── .gitignore                # Loại bỏ file lớn (yolov5s.pt, uploads/*) khỏi Git
├── README.md                 # Hướng dẫn cài đặt và chạy ứng dụng
├── requirements.txt          # Danh sách thư viện Python cần cài đặt
├── static/
│   ├── css/
│   │   └── style.css         # File CSS tùy chỉnh giao diện (menu, khung 640x480px)
│   └── uploads/              # Thư mục lưu ảnh/video (không commit vào Git)
│       ├── images/
│       │   ├── original/     # Ảnh gốc
│       │   └── processed/    # Ảnh đã nhận diện
│       ├── videos/
│       │   ├── original/     # Video gốc
│       │   └── processed/    # Video đã nhận diện
│       └── metadata.json     # Lưu metadata (tên file, loại, thời gian xử lý)
├── templates/
│   ├── base.html             # Template cơ sở, chứa menu slide-in
│   ├── login.html            # Trang đăng nhập
│   ├── register.html         # Trang đăng ký
│   ├── dashboard.html        # Trang tải lên ảnh/video
│   ├── camera.html           # Trang chụp ảnh/ghi video và nhận diện trực tiếp
│   ├── history.html          # Trang hiển thị lịch sử kết quả
│   └── result.html           # Trang hiển thị kết quả nhận diện
├── yolo-coco/                # (Đã thay thế bằng YOLOv5, có thể xóa)
│   ├── coco.names            # (Không còn sử dụng)
│   ├── yolov3.cfg            # (Không còn sử dụng)
│   └── yolov3.weights        # (Không còn sử dụng, thay bằng yolov5s.pt)
├── yolov5/                   # Thư mục chứa mã nguồn YOLOv5 (tải từ GitHub)
│   ├── models/               # Chứa mô hình YOLOv5
│   ├── utils/                # Tiện ích YOLOv5
│   ├── requirements.txt      # Yêu cầu thư viện cho YOLOv5
│   └── yolov5s.pt            # File trọng số YOLOv5 (cần tải riêng)
└── venv/                     # Môi trường ảo Python (không commit)

text

Sao chép

### Mô tả chi tiết

- **`app.py`**: Chứa logic Flask, quản lý route (`/login`, `/dashboard`, `/camera`, `/history`, `/result`, `/process_frame`), lưu file vào `static/uploads`, và lưu metadata vào `metadata.json`.
- **`yolo.py`, `yolo_video.py`**: Chứa hàm nhận diện đối tượng cho ảnh và video, sử dụng OpenCV và mô hình YOLOv5.
- **`.gitignore`**: Ngăn commit các file lớn (`yolov5s.pt`, `uploads/*`, `metadata.json`) và file tạm (`venv/`, `__pycache__`).
- **`static/css/style.css`**: CSS tùy chỉnh cho menu slide-in, khung 640x480px, và giao diện responsive.
- **`templates/*`**: Các template HTML sử dụng Tailwind CSS:
  - `base.html`: Khung giao diện chính, chứa menu slide-in (chỉ hiển thị sau đăng nhập).
  - `login.html`, `register.html`: Giao diện đăng nhập/đăng ký.
  - `dashboard.html`: Form tải lên ảnh/video.
  - `camera.html`: Giao diện chụp ảnh/ghi video từ webcam và nhận diện trực tiếp.
  - `history.html`: Bảng danh sách kết quả đã xử lý (tên file, loại, thời gian, liên kết xem chi tiết).
  - `result.html`: Hiển thị ảnh/video đã nhận diện và danh sách đối tượng.
- **`static/uploads/`**: Lưu ảnh/video gốc và đã xử lý, cùng `metadata.json` (không commit vào Git).
- **`yolo-coco/`**: Thư mục cũ chứa file YOLOv3 (có thể xóa, đã thay thế bằng YOLOv5).
- **`yolov5/`**: Thư mục chứa mã nguồn và mô hình YOLOv5, bao gồm file `yolov5s.pt` (cần tải riêng).

## Yêu cầu để chạy code

### 1. Phần mềm và công cụ
- **Python**: Phiên bản 3.8 trở lên.
- **Git**: Để clone repository.
- **Trình duyệt**: Chrome, Firefox (hỗ trợ webcam cho tính năng Camera).
- **VLC Media Player (tùy chọn)**: Để xem video đã nhận diện trong `static/uploads/videos/processed`.

### 2. Clone repository
Clone dự án từ GitHub:
```bash
git clone https://github.com/chucomatnao/yolo_web.git
cd yolo_web
3. Tải file mô hình YOLO
Ứng dụng hiện sử dụng YOLOv5 thay vì YOLOv3. File yolov5s.pt không có trong repository do kích thước lớn (~80 MB). Tải từ nguồn chính thức:

Truy cập: https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt
Tải thủ công:
Mở trình duyệt, dán link trên, tải file yolov5s.pt.
Đặt vào thư mục dự án: yolo_web_app/yolov5/.
4. Cài đặt YOLOv5
Clone mã nguồn YOLOv5:
bash

Sao chép
git clone https://github.com/ultralytics/yolov5.git
mv yolov5/* yolo_web_app/yolov5/
Cài đặt thư viện cho YOLOv5:
bash

Sao chép
cd yolo_web_app/yolov5
pip install -r requirements.txt
5. Cài đặt môi trường ảo
Tạo và kích hoạt môi trường ảo:

Trên Windows:
bash

Sao chép
python -m venv venv
venv\Scripts\activate
Trên Linux/MacOS:
bash

Sao chép
python -m venv venv
source venv/bin/activate
6. Cài đặt thư viện
Cài các thư viện cần thiết từ requirements.txt:

bash

Sao chép
pip install -r requirements.txt
Nội dung requirements.txt (cập nhật cho YOLOv5):

text

Sao chép
flask==2.0.1
flask-login==0.5.0
opencv-python==4.11.0
numpy==1.21.0
werkzeug==2.0.1
torch==2.0.0
torchvision==0.15.0
Nếu chưa có requirements.txt, tạo file với nội dung trên hoặc cài trực tiếp:

bash

Sao chép
pip install flask flask-login opencv-python numpy werkzeug torch torchvision
7. Chạy ứng dụng
Khởi động ứng dụng Flask:

bash

Sao chép
python app.py
Truy cập: http://127.0.0.1:5000

8. Chạy với HTTPS (tùy chọn, cho Camera trên mạng)
Cài flask[async] để hỗ trợ HTTPS:

bash

Sao chép
pip install flask[async]
Chạy với chứng chỉ tự ký:

bash

Sao chép
python -m flask run --cert=adhoc
Truy cập: 

Hướng dẫn sử dụng
Đăng ký/Đăng nhập
Truy cập http://127.0.0.1:5000, tạo tài khoản và đăng nhập.
Menu slide-in xuất hiện với các mục: Dashboard, Camera, History, Logout.
Dashboard
Tải lên ảnh (.jpg, .png) hoặc video (.mp4, .avi, .webm).
Kết quả được lưu vào static/uploads/images/processed hoặc static/uploads/videos/processed.
Camera
Chụp ảnh hoặc ghi video từ webcam.
Kết quả lưu tương tự như Dashboard.
Nhận diện trực tiếp: Click "Nhận diện trực tiếp" để xem kết quả theo thời gian thực trong <ul id="detectionList">.
History
Xem danh sách các file đã xử lý (tên, loại, thời gian, liên kết xem chi tiết).
Click "Xem" để chuyển đến trang Result.
Result
Hiển thị ảnh/video trong khung 640x480px.
Liệt kê các đối tượng được phát hiện (ví dụ: person: 0.8744).
Lưu ý
File lớn: Không commit yolov5s.pt, static/uploads/*, hoặc metadata.json vào Git. Đã cấu hình trong .gitignore.
Camera trên mạng: Cần HTTPS (--cert=adhoc) để webcam hoạt động trên các trình duyệt hiện đại.
Debug:
Nếu thiếu yolov5s.pt, ứng dụng sẽ báo lỗi. Tải và đặt đúng thư mục yolo_web_app/yolov5/.
Kiểm tra static/uploads/metadata.json để xem metadata của các file đã xử lý.
Mở static/uploads/images/processed/output_*.jpg hoặc static/uploads/videos/processed/output_*.mp4 để kiểm tra kết quả.
Cấu hình thư mục lưu trữ
Ảnh và video được lưu trong static/uploads/:

Ảnh gốc: static/uploads/images/original/<tên_ảnh>.jpg
Ảnh đã nhận diện: static/uploads/images/processed/output_<tên_ảnh>.jpg
Video gốc: static/uploads/videos/original/<tên_video>.mp4
Video đã nhận diện: static/uploads/videos/processed/output_<tên_video>.mp4
Metadata: static/uploads/metadata.json (tên file, loại, thời gian).
Liên hệ
Nếu gặp vấn đề, liên hệ qua GitHub Issues: https://github.com/chucomatnao/yolo_web/issues