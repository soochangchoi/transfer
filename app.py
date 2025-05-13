from flask import Flask, render_template, request
import os
from torchvision import transforms, models
import torch
import torch.nn as nn
from PIL import Image
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from urllib.parse import quote_plus

# Flask 앱 설정
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# MySQL 연결 설정
DB_USER = 'user01'  # 본인 아이디
DB_PASSWORD = quote_plus('1234')  # 본인 비밀번호
DB_HOST = '192.168.2.170'   # 서버 IP
DB_PORT = '3306'
DB_NAME = 'monkeydb'

app.config['SQLALCHEMY_DATABASE_URI'] = f'mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# monkeybase 테이블 모델
class Monkeybase(db.Model):
    __tablename__ = 'monkeybase'
    
    id = db.Column(db.Integer, primary_key=True)
    path = db.Column(db.String(100))
    label = db.Column(db.String(30))

# 클래스 이름 설정
class_names = ['etc', 'n8', 'n9']  # 본인 train_dataset.classes 순서에 맞춰야 함

# EfficientNet 모델 로드
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
model.load_state_dict(torch.load('../Monkey/best_efficientnet.pth', map_location=device))
model = model.to(device)
model.eval()

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 메인 페이지 (GET)
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# 이미지 업로드 및 예측 처리 (POST)
@app.route('/', methods=['POST'])
def predict():
    file = request.files['file']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # 이미지 열기 및 전처리
    image = Image.open(filepath).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0).to(device)

    # 예측
    with torch.no_grad():
        outputs = model(image)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        _, pred = torch.max(outputs, 1)

    pred_class = class_names[pred.item()]
    confidence = probs[0][pred.item()].item() * 100

    # 예측 결과 DB에 저장
    new_record = Monkeybase(
        path=filepath,  # 업로드된 파일 경로
        label=pred_class  # 예측된 클래스 이름
    )
    db.session.add(new_record)
    db.session.commit()

    return render_template('index.html', filename=file.filename, prediction=pred_class, confidence=confidence)

# 업로드한 파일 표시용 (사용 안 해도 무방)
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return os.path.join(app.config['UPLOAD_FOLDER'], filename)

# DB 기록 조회 페이지
@app.route('/result')
def result():
    records = Monkeybase.query.all()
    return render_template('result.html', records=records)

# 서버 실행
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
