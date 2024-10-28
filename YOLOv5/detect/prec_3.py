import torch
import pandas as pd

# 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # 또는 다른 모델 선택

# 이미지 로드 및 전처리
img_path = 'path/to/your/image.jpg'  # 예측할 이미지 경로
img = Image.open(img_path)

# 예측 수행
results = model(img)

# 예측 결과 가져오기
predictions = results.xyxy[0]  # (x1, y1, x2, y2, confidence, class)

# PredictionString 생성
prediction_strings = []
for pred in predictions:
    x1, y1, x2, y2, conf, cls = pred
    prediction_string = f"{cls} {conf:.6f} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}"
    prediction_strings.append(prediction_string)

# 최종 PredictionString
final_prediction_string = " ".join(prediction_strings)

# 결과 출력
print(final_prediction_string)
