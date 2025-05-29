# app.py
import os
import uuid
from flask import Flask, request, jsonify, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from train_model import train_model

app = Flask(__name__)

# Cấu hình thư mục lưu file upload
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Chỉ cho phép file CSV
ALLOWED_EXTENSIONS = {'csv'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Kiểm tra file đã được gửi lên hay chưa
        if 'file' not in request.files:
            return render_template('index.html', error="Không tìm thấy file upload.")
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error="Không có file nào được chọn.")
        if file and allowed_file(file.filename):
            # Lưu file tạm vào thư mục uploads
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4().hex}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)
            
            # Lấy tham số gửi từ form, mặc định là regression
            model_type = request.form.get("model_type", "regression")
            
            # Gọi hàm huấn luyện từ train_model.py
            result = train_model(filepath, model_type)
            
            # Xóa file tạm sau khi xử lý
            os.remove(filepath)
            
            # Render kết quả trả về qua giao diện result.html
            return render_template('result.html',
                                   table_html=result['table_html'],
                                   plot_base64=result['plot_base64'],
                                   model_type=model_type)
        else:
            return render_template('index.html', error="File không hợp lệ. Xin hãy upload file CSV.")
    return render_template('index.html')

# API REST: endpoint tiếp nhận POST request (ví dụ dùng AJAX/Postman)
@app.route('/api/train', methods=['POST'])
def api_train():
    if 'file' not in request.files:
        return jsonify({"error": "Không tìm thấy file upload."}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Không có file được chọn."}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        model_type = request.form.get("model_type", "regression")
        result = train_model(filepath, model_type)
        os.remove(filepath)
        return jsonify(result)
    else:
        return jsonify({"error": "File không hợp lệ. Chỉ chấp nhận file CSV."}), 400

if __name__ == '__main__':
    app.run(debug=True)
