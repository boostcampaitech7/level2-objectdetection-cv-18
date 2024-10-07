import os
import torch
import cv2
import numpy as np
import csv
import json
from utils.dataloaders import LoadImages
from utils.general import non_max_suppression

def detect(weights, img_size=640, conf_thres=0.25, iou_thres=0.45, json_file=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights, force_reload=True)
    model.eval()

    if json_file is not None:
        with open(json_file, 'r') as f:
            data = json.load(f)
        images = data['images']
    else:
        raise ValueError("No JSON file provided.")

    with open('output_1.csv', mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['PredictionString', 'image_id'])

        for img_info in images:
            image_id = img_info['id']
            relative_path = img_info['file_name']
            img_path = os.path.join('/data/ephemeral/home/dataset', relative_path)

            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Failed to load image {img_path}")
                continue

            # 이미지를 (B, C, H, W) 형태로 변환 (C: Channels, H: Height, W: Width)
            img = torch.from_numpy(img).to(device).float() / 255.0
            img = img.permute(2, 0, 1).unsqueeze(0)  # (H, W, C) -> (C, H, W), 그리고 batch 차원 추가
            # Inference
            pred = model(img, augment=False)[0]

            # 예측 결과 차원 확인
            #print(f"Prediction shape: {pred.shape}")

            # 예측 결과가 예상대로 3D 텐서인지 확인
            if pred.ndim == 3:
                try:
                    # 속성 개수 확인 (클래스 수 + 기타 속성)
                    if pred.shape[2] >= 15:  # 속성 개수가 15 이상인지 확인
                        pred = pred[0]  # 첫 번째 배치 항목 선택
                    else:
                        print(f"Unexpected prediction shape: {pred.shape}")
                        continue  # 예상하지 못한 형식일 경우 넘어감
                except IndexError as e:
                    print(f"Error processing prediction for image {img_path}: {e}")
                    continue  # 오류 발생 시 해당 이미지는 건너뜀
            elif pred.ndim == 2:
                pred = pred.unsqueeze(0)  # 차원 추가 (필요한 경우)

            # Apply NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres)

            pred_str = ""

            for det in pred:
                if det is not None and len(det):
                    for *xyxy, conf, cls in reversed(det):
                        xyxy = [coord.item() for coord in xyxy]
                        pred_str += f"{int(cls)} {conf} {xyxy[0]} {xyxy[1]} {xyxy[2]} {xyxy[3]} "
            
            formatted_image_id = f"test/{str(image_id).zfill(4)}.jpg"

            if pred_str:
                csv_writer.writerow([pred_str.strip(), formatted_image_id])
            else:
                print(f"No predictions for image {img_path}")
                csv_writer.writerow([0, formatted_image_id])

if __name__ == "__main__":
    detect(weights='/data/ephemeral/home/jiwan/level2-objectdetection-cv-18/yolov5/runs/train/exp12/weights/best.pt',
            json_file='/data/ephemeral/home/dataset/test.json',
            conf_thres=0.05,  # Confidence threshold 설정 (예시: 0.1)
            iou_thres=0.55)   # IOU threshold 설정 (예시: 0.5))

