from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, session, jsonify
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import os
import cv2
import numpy as np
import json
from datetime import datetime
from yolo import process_image
from yolo_video import process_video

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['IMAGE_ORIGINAL'] = os.path.join(app.config['UPLOAD_FOLDER'], 'images/original')
app.config['IMAGE_PROCESSED'] = os.path.join(app.config['UPLOAD_FOLDER'], 'images/processed')
app.config['VIDEO_ORIGINAL'] = os.path.join(app.config['UPLOAD_FOLDER'], 'videos/original')
app.config['VIDEO_PROCESSED'] = os.path.join(app.config['UPLOAD_FOLDER'], 'videos/processed')
app.config['METADATA_FILE'] = os.path.join(app.config['UPLOAD_FOLDER'], 'metadata.json')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'webm'}

# Thiết lập Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Lớp User giả lập cơ sở dữ liệu
class User(UserMixin):
    def __init__(self, id, username, password_hash):
        self.id = id
        self.username = username
        self.password_hash = password_hash

# Giả lập cơ sở dữ liệu người dùng
users = {}

@login_manager.user_loader
def load_user(user_id):
    return users.get(user_id)

# Kiểm tra định dạng file
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Tạo các thư mục nếu chưa tồn tại
def ensure_directories():
    os.makedirs(app.config['IMAGE_ORIGINAL'], exist_ok=True)
    os.makedirs(app.config['IMAGE_PROCESSED'], exist_ok=True)
    os.makedirs(app.config['VIDEO_ORIGINAL'], exist_ok=True)
    os.makedirs(app.config['VIDEO_PROCESSED'], exist_ok=True)

# Lưu metadata
def save_metadata(filename, file_type, detections):
    metadata = {}
    if os.path.exists(app.config['METADATA_FILE']):
        with open(app.config['METADATA_FILE'], 'r') as f:
            metadata = json.load(f)
    metadata[filename] = {
        'type': file_type,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'detections': detections or []
    }
    with open(app.config['METADATA_FILE'], 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata updated for {filename}: {metadata[filename]}")  # Debug

@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = next((u for u in users.values() if u.username == username), None)
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        flash('Tên đăng nhập hoặc mật khẩu không đúng.', 'error')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if any(u.username == username for u in users.values()):
            flash('Tên đăng nhập đã tồn tại.', 'error')
            return render_template('register.html')
        user_id = str(len(users) + 1)
        password_hash = generate_password_hash(password)
        users[user_id] = User(user_id, username, password_hash)
        flash('Đăng ký thành công! Vui lòng đăng nhập.', 'success')
        session.clear()
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    ensure_directories()
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('Không có file được chọn.', 'error')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('Không có file được chọn.', 'error')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = file.filename
            is_image = filename.lower().endswith(('.png', '.jpg', '.jpeg'))
            original_folder = app.config['IMAGE_ORIGINAL'] if is_image else app.config['VIDEO_ORIGINAL']
            processed_folder = app.config['IMAGE_PROCESSED'] if is_image else app.config['VIDEO_PROCESSED']
            filepath = os.path.join(original_folder, filename)
            file.save(filepath)
            output_filename = 'output_' + filename.rsplit('.', 1)[0] + ('.jpg' if is_image else '.mp4')
            output_path = os.path.join(processed_folder, output_filename)

            try:
                if is_image:
                    detections = process_image(filepath, output_path)
                else:
                    detections = process_video(filepath, output_path)
                    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                        raise ValueError("File video đầu ra không được tạo hoặc bị hỏng")
                print(f"Stored detections: {detections}")  # Debug
                session['detections'] = detections or []
                session['file_type'] = 'image' if is_image else 'video'
                save_metadata(output_filename, session['file_type'], detections)
                return redirect(url_for('result', filename=output_filename))
            except Exception as e:
                print(f"Error processing file: {str(e)}")  # Debug
                flash(f'Lỗi khi xử lý file: {str(e)}', 'error')
                return redirect(request.url)
        else:
            flash('Định dạng file không được hỗ trợ.', 'error')
    return render_template('dashboard.html')

@app.route('/camera', methods=['GET', 'POST'])
@login_required
def camera():
    ensure_directories()
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'Không có file được gửi'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'Không có file được chọn'}), 400
        if file and allowed_file(file.filename):
            filename = file.filename
            is_image = filename.lower().endswith(('.png', '.jpg', '.jpeg'))
            original_folder = app.config['IMAGE_ORIGINAL'] if is_image else app.config['VIDEO_ORIGINAL']
            processed_folder = app.config['IMAGE_PROCESSED'] if is_image else app.config['VIDEO_PROCESSED']
            filepath = os.path.join(original_folder, filename)
            file.save(filepath)
            output_filename = 'output_' + filename.rsplit('.', 1)[0] + ('.jpg' if is_image else '.mp4')
            output_path = os.path.join(processed_folder, output_filename)

            try:
                if is_image:
                    detections = process_image(filepath, output_path)
                else:
                    detections = process_video(filepath, output_path)
                    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                        raise ValueError("File video đầu ra không được tạo hoặc bị hỏng")
                print(f"Stored detections: {detections}")  # Debug
                session['detections'] = detections or []
                session['file_type'] = 'image' if is_image else 'video'
                save_metadata(output_filename, session['file_type'], detections)
                return jsonify({'success': True, 'filename': output_filename})
            except Exception as e:
                print(f"Error processing file: {str(e)}")  # Debug
                return jsonify({'success': False, 'error': str(e)}), 500
        else:
            return jsonify({'success': False, 'error': 'Định dạng file không được hỗ trợ'}), 400
    return render_template('camera.html')

@app.route('/history')
@login_required
def history():
    results = []
    if os.path.exists(app.config['METADATA_FILE']):
        with open(app.config['METADATA_FILE'], 'r') as f:
            metadata = json.load(f)
        for filename, info in metadata.items():
            results.append({
                'filename': filename,
                'type': info.get('type', 'unknown'),
                'timestamp': info.get('timestamp', 'N/A'),
                'detections': info.get('detections', [])
            })
    results.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    print(f"History results: {results}")  # Debug
    return render_template('history.html', results=results)

@app.route('/result/<filename>')
@login_required
def result(filename):
    file_type = None
    detections = []
    if os.path.exists(app.config['METADATA_FILE']):
        with open(app.config['METADATA_FILE'], 'r') as f:
            metadata = json.load(f)
        if filename in metadata:
            file_type = metadata[filename].get('type', 'image')
            detections = metadata[filename].get('detections', [])
    print(f"Retrieved detections from metadata for {filename}: {detections}")  # Debug
    return render_template('result.html', filename=filename, detections=detections, file_type=file_type)

@app.route('/uploads/<file_type>/<filename>')
@login_required
def uploaded_file(file_type, filename):
    folder = app.config['IMAGE_PROCESSED'] if file_type == 'image' else app.config['VIDEO_PROCESSED']
    return send_from_directory(folder, filename)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

if __name__ == '__main__':
    ensure_directories()
    app.run(debug=True)