# 🐵 Monkey Species Classification (Transfer Learning & CNN)

EfficientNet-B0 및 Custom CNN을 활용하여 원숭이 10종 분류 모델을 학습 및 평가한 프로젝트입니다.  
Flask 웹과 MySQL을 연동하여 이미지 업로드 및 결과 관리 기능까지 제공합니다.

---

## 📂 프로젝트 구조

| 파트                        | 주요 기능                          | 스크립트              | 모델 파일                |
|---------------------------|---------------------------------|---------------------|----------------------|
| Custom CNN 원숭이 분류        | SimpleCNN을 통한 원숭이 10종 분류 학습    | `monekycnn.py`       | `best_simplecnn.pth` |
| EfficientNet 전이학습 원숭이 분류 | EfficientNet-B0 전이학습을 통한 원숭이 10종 분류 | `monkey.py`          | `best_efficientnet.pth` |
| Flask 웹 서비스 및 DB 연동    | 업로드 이미지 분류 + 결과 기록 + DB 저장  | `app.py`             | `best_efficientnet.pth` |
| 데이터 준비 및 전처리 노트북   | 이미지셋 정리 및 검증셋 생성             | `makeDS.ipynb`, `makeDS valid.ipynb` | - |

---

## 🛠 사용 기술

- Python 3.9
- PyTorch, torchvision
- EfficientNet (torchvision pretrained)
- Flask
- MySQL (SQLAlchemy)
- OpenCV, matplotlib
- Grad-CAM 분석 포함

---

## ▶ 실행 방법

### 1. Custom CNN 학습 (SimpleCNN)
```bash
python monekycnn.py
데이터셋: ../Monkey/training/training

모델: SimpleCNN

결과: best_simplecnn.pth

2. EfficientNet 전이학습 (추천)

python monkey.py
모델: EfficientNet-B0 (Pretrained on ImageNet)

결과: best_efficientnet.pth

추가: Confusion Matrix, Grad-CAM 분석 및 F1 Score 출력

3. Flask 웹 서비스 실행

python app.py
URL: http://localhost:5000/

기능: 이미지 업로드 → 분류 → DB 기록 (monkeybase 테이블)

DB 조회: http://localhost:5000/result

4. 데이터 전처리
makeDS.ipynb : 데이터셋 생성

makeDS valid.ipynb : 검증셋 준비

🏷 클래스 레이블 정보 (monkey_labels.txt)
ID	라틴명	일반명
n0	alouatta_palliata	mantled_howler
n1	erythrocebus_patas	patas_monkey
n2	cacajao_calvus	bald_uakari
n3	macaca_fuscata	japanese_macaque
n4	cebuella_pygmea	pygmy_marmoset
n5	cebus_capucinus	white_headed_capuchin
n6	mico_argentatus	silvery_marmoset
n7	saimiri_sciureus	common_squirrel_monkey
n8	aotus_nigriceps	black_headed_night_monkey
n9	trachypithecus_johnii	nilgiri_langur

💡 주요 특징
✅ EfficientNet 전이학습 기반 고성능 모델

✅ Grad-CAM으로 AI의 예측 이유 해석 가능

✅ Flask 웹 & MySQL 연동 통한 결과 관리 시스템

✅ Softmax 기반 confidence score 표시
