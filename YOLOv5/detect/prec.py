import torch
import cv2
import numpy as np
import csv
from utils.dataloaders import LoadImages  # LoadImages 임포트
from utils.general import non_max_suppression  # non_max_suppression 임포트

def detect(source, weights, img_size=640, conf_thres=0.25, iou_thres=0.45):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 자동으로 장치 선택

    # Load model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights, force_reload=True)
    model.eval()

    # Load source images (e.g., video, webcam, image files)
    dataset = LoadImages(source, img_size=img_size)

    # CSV 파일 열기
    with open('output.csv', mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # 헤더 작성
        csv_writer.writerow(['PredictionString', 'image_id'])

        for data in dataset:
            path = data[0]  # 이미지 경로
            img = None  # img 변수를 None으로 초기화
            print(path)

            # 5개 반환하는 경우에 대한 처리
            if len(data) == 5:
                img = data[1]  # 이미지 데이터

            if img is not None:  # img가 None이 아닐 때만 처리
                img = torch.from_numpy(img).to(device).float() / 255.0  # Normalize the image

                if len(img.shape) == 3:  # 3D일 경우
                    img = img.unsqueeze(0)  # Add batch dimension

                # Inference
                pred = model(img, augment=False)[0]  # Perform inference

                # Apply Non-Maximum Suppression (NMS)
                pred = non_max_suppression(pred, conf_thres, iou_thres)

                # Process detections
                pred_str = ""  # PredictionString 초기화
                image_id = path.split('/')[-1]  # 이미지 ID 가져오기

                for det in pred:  # Detections per image
                    if det is not None and len(det):
                        for *xyxy, conf, cls in reversed(det):
                            # 예측 결과를 PredictionString 형식으로 저장
                            xyxy = [coord.item() for coord in xyxy]  # Convert coordinates to float
                            pred_str += f"{int(cls)} {conf:.6f} {xyxy[0]} {xyxy[1]} {xyxy[2]} {xyxy[3]} "

                # 결과를 CSV 파일에 기록
                if pred_str:
                    csv_writer.writerow([pred_str.strip(), image_id])  # PredictionString과 image_id를 기록
            else:
                print("Warning: No image data to process.")

if __name__ == "__main__":
    detect(source='/data/ephemeral/home/dataset/test',
           weights='/data/ephemeral/home/jiwan/level2-objectdetection-cv-18/yolov5/runs/train/exp11/weights/best.pt')
