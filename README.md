Ứng dụng Web Nhận diện Đối tượng bằng YOLO
Ứng dụng web này sử dụng mô hình YOLO (You Only Look Once) để nhận diện đối tượng trong ảnh và video. Người dùng có thể đăng nhập, tải lên ảnh/video, chụp ảnh/ghi video từ camera, xem lịch sử kết quả, và xem chi tiết các đối tượng được phát hiện (với hộp giới hạn và độ tin cậy).
Tính năng chính

Đăng nhập/Đăng ký: Người dùng cần đăng nhập để sử dụng ứng dụng.
Menu slide-in: Hiển thị sau đăng nhập, bao gồm Dashboard, Camera, History, Logout.
Dashboard: Tải lên ảnh (.jpg, .png) hoặc video (.mp4, .avi, .webm) để nhận diện.
Camera: Chụp ảnh hoặc ghi video trực tiếp từ webcam, sau đó nhận diện.
History: Xem danh sách các ảnh/video đã xử lý (tên file, loại, thời gian, liên kết xem chi tiết).
Result: Hiển thị ảnh/video đã nhận diện trong khung 640x480px, cùng danh sách đối tượng (ví dụ: person: 0.8744).
Quản lý file: Ảnh/video được lưu vào thư mục static/uploads với cấu trúc:
images/original: Ảnh gốc.
images/processed: Ảnh đã nhận diện.
videos/original: Video gốc.
videos/processed: Video đã nhận diện.



Cấu trúc thư mục và giải thích file
yolo_web_app/
├── app.py                    # File chính, chạy ứng dụng Flask, xử lý route và logic
├── yolo.py                   # Hàm xử lý nhận diện ảnh bằng YOLO
├── yolo_video.py             # Hàm xử lý nhận diện video bằng YOLO
├── .gitignore                # Loại bỏ file lớn (yolov3.weights, uploads/*) khỏi Git
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
│   ├── camera.html           # Trang chụp ảnh/ghi video
│   ├── history.html          # Trang hiển thị lịch sử kết quả
│   └── result.html           # Trang hiển thị kết quả nhận diện
├── yolo-coco/
│   ├── coco.names            # Danh sách lớp đối tượng (person, car, ...)
│   ├── yolov3.cfg            # Cấu hình mô hình YOLO
│   └── yolov3.weights        # File mô hình YOLO (không commit, cần tải riêng)
└── venv/                     # Môi trường ảo Python (không commit)

Mô tả chi tiết

app.py: Chứa logic Flask, quản lý route (/login, /dashboard, /camera, /history, /result), lưu file vào static/uploads, và lưu metadata vào metadata.json.
yolo.py, yolo_video.py: Chứa hàm nhận diện đối tượng cho ảnh và video, sử dụng OpenCV và mô hình YOLO.
.gitignore: Ngăn commit các file lớn (yolov3.weights, uploads/*, metadata.json) và file tạm (venv/, __pycache__/).
static/css/style.css: CSS tùy chỉnh cho menu slide-in, khung 640x480px, và giao diện responsive.
templates/*: Các template HTML sử dụng Tailwind CSS:
base.html: Khung giao diện chính, chứa menu slide-in (chỉ hiển thị sau đăng nhập).
login.html, register.html: Giao diện đăng nhập/đăng ký.
dashboard.html: Form tải lên ảnh/video.
camera.html: Giao diện chụp ảnh/ghi video từ webcam.
history.html: Bảng danh sách kết quả đã xử lý (tên file, loại, thời gian, liên kết xem).
result.html: Hiển thị ảnh/video đã nhận diện và danh sách đối tượng.


static/uploads/: Lưu ảnh/video gốc và đã xử lý, cùng metadata.json (không commit vào Git).
yolo-coco/: Chứa file cấu hình và mô hình YOLO. File yolov3.weights (~237 MB) không commit, cần tải riêng.

Yêu cầu để chạy code
1. Phần mềm và công cụ

Python: Phiên bản 3.8 trở lên.
Git: Để clone repository.
Trình duyệt: Chrome, Firefox (hỗ trợ webcam cho tính năng Camera).
VLC Media Player (tùy chọn): Để xem video đã nhận diện trong static/uploads/videos/processed.

2. Clone repository
Clone dự án từ GitHub:
git clone https://github.com/chucomatnao/yolo_web.git
cd yolo_web

3. Tải file mô hình YOLO
File yolo-coco/yolov3.weights không có trong repository do kích thước lớn (~237 MB). Tải từ nguồn chính thức:

Truy cập: https://pjreddie.com/darknet/yolo/
Tải file yolov3.weights.
Đặt vào thư mục yolo-coco/:

yolo_web/
└── yolo-coco/
    └── yolov3.weights

Hoặc dùng lệnh (nếu có wget):
wget https://pjreddie.com/media/files/yolov3.weights -O yolo-coco/yolov3.weights

4. Cài đặt môi trường ảo
Tạo và kích hoạt môi trường ảo:
python -m venv venv


Trên Windows:

venv\Scripts\activate


Trên Linux/MacOS:

source venv/bin/activate

5. Cài đặt thư viện
Cài các thư viện cần thiết từ requirements.txt:
pip install -r requirements.txt

Nội dung requirements.txt:
flask==2.0.1
flask-login==0.5.0
opencv-python==4.5.3
numpy==1.21.0
werkzeug==2.0.1

Nếu chưa có requirements.txt, tạo file với nội dung trên hoặc cài trực tiếp:
pip install flask flask-login opencv-python numpy werkzeug

6. Chạy ứng dụng
Khởi động ứng dụng Flask:
python app.py

Truy cập: http://127.0.0.1:5000
7. Chạy với HTTPS (tùy chọn, cho Camera trên mạng)
Cài flask[async] để hỗ trợ HTTPS:
pip install flask[async]

Chạy với chứng chỉ tự ký:
python -m flask run --cert=adhoc

Truy cập: https://<IP>:5000
Hướng dẫn sử dụng

Đăng ký/Đăng nhập:
Truy cập http://127.0.0.1:5000, tạo tài khoản và đăng nhập.
Menu slide-in xuất hiện với các mục: Dashboard, Camera, History, Logout.


Dashboard:
Tải lên ảnh (.jpg, .png) hoặc video (.mp4, .avi, .webm).
Kết quả được lưu vào static/uploads/images/processed hoặc static/uploads/videos/processed.


Camera:
Chụp ảnh hoặc ghi video từ webcam.
Kết quả lưu tương tự như Dashboard.


History:
Xem danh sách các file đã xử lý (tên, loại, thời gian, liên kết xem chi tiết).
Click "Xem" để chuyển đến trang Result.


Result:
Hiển thị ảnh/video trong khung 640x480px.
Liệt kê các đối tượng được phát hiện (ví dụ: person: 0.8744).



Lưu ý

File lớn: Không commit yolo-coco/yolov3.weights, static/uploads/*, hoặc metadata.json vào Git. Đã cấu hình trong .gitignore.
Camera trên mạng: Cần HTTPS (--cert=adhoc) để webcam hoạt động trên các trình duyệt hiện đại.
Debug:
Nếu thiếu yolov3.weights, ứng dụng sẽ báo lỗi. Tải và đặt đúng thư mục.
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
