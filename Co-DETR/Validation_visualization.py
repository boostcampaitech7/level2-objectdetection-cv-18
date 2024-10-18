from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import numpy as np
import cv2
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
import json
import os
from pycocotools.coco import COCO
from mmcv import Config
import matplotlib.pyplot as plt
import torch
import gc
from sklearn.metrics import precision_recall_curve, average_precision_score

def visualize_image_and_pr_curves(img_path, wrong_predictions, correct_bboxes, precision_per_class, recall_per_class, avg_precision_per_class):
    classes = ["General trash", "Paper", "Paper pack", "Metal", "Glass", 
               "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"]
    
    img = cv2.imread(img_path)
    
    # 틀린 예측에 대한 bounding box 그리기 (빨간색으로 표시)
    for bbox in wrong_predictions:
        x_min, y_min, x_max, y_max = map(int, bbox[:4])
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
        class_id = int(bbox[4])
        cv2.putText(img, f'GT: {classes[class_id]}', (x_min, y_min - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # 정답 bounding box 그리기 (파란색으로 표시)
    for bbox in correct_bboxes:
        x_min, y_min, x_max, y_max = map(int, bbox[1:5])
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        class_id = int(bbox[0])
        cv2.putText(img, f'GT: {classes[class_id]}', (x_min, y_min - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # 시각화: 한쪽은 이미지, 한쪽은 PR 곡선
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    
    # 이미지 출력
    ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # BGR을 RGB로 변환하여 출력
    ax[0].axis('off')  # 이미지 부분에서 축 제거
    ax[0].set_title("Prediction & Ground Truth")
    
    # 각 클래스에 대한 PR 곡선 시각화
    for class_id in range(len(classes)):
        ax[1].step(recall_per_class[class_id], precision_per_class[class_id], where='post', label=f'{classes[class_id]} (AP={avg_precision_per_class[class_id]:.2f})')
    
    ax[1].set_xlabel('Recall')
    ax[1].set_ylabel('Precision')
    ax[1].set_title('PR Curves for All Classes')
    ax[1].legend(loc='lower left')
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()
    plt.savefig(f'./work_dirs/visualization/{os.path.basename(img_path)}_pr_curves.png')  # 파일로 저장
    plt.close()  # 현재 figure 닫기

def compute_pr_curve_per_class(gt_bboxes, pred_bboxes, num_classes=10, iou_threshold=0.5):
    precision_per_class = {}
    recall_per_class = {}
    avg_precision_per_class = {}
    
    for class_id in range(num_classes):
        # 해당 클래스에 해당하는 GT와 예측 바운딩 박스 필터링
        gt_bboxes_class = [bbox for bbox in gt_bboxes if bbox[0] == class_id]
        pred_bboxes_class = [bbox for bbox in pred_bboxes if bbox[5] == class_id]  # 5번 인덱스는 클래스 ID
        
        # GT 및 예측 바운딩 박스가 없으면 스킵
        if len(gt_bboxes_class) == 0 or len(pred_bboxes_class) == 0:
            continue

        # Precision-Recall 계산
        precision, recall, avg_precision = compute_pr_curve(np.array(gt_bboxes_class), np.array(pred_bboxes_class))
        
        precision_per_class[class_id] = precision
        recall_per_class[class_id] = recall
        avg_precision_per_class[class_id] = avg_precision

    return precision_per_class, recall_per_class, avg_precision_per_class

def compute_pr_curve(gt_bboxes, pred_bboxes, iou_threshold=0.5):
    # pred_bboxes와 gt_bboxes에 대해 IoU 계산
    iou_matrix = bbox_overlaps(pred_bboxes[:, :4], gt_bboxes[:, 1:5])  # 예측은 [x_min, y_min, x_max, y_max], GT는 [class_id, x_min, y_min, x_max, y_max]

    # 각 예측에 대해 최대 IoU와 해당 GT 인덱스를 찾음
    max_ious = iou_matrix.max(axis=1)
    max_iou_indices = iou_matrix.argmax(axis=1)
    
    y_true = []
    y_scores = []

    # 예측된 바운딩 박스를 기준으로 TP/FP 계산
    for i, iou in enumerate(max_ious):
        if iou >= iou_threshold:
            # IoU가 threshold 이상이면 TP로 간주
            y_true.append(1)
        else:
            # 그 외의 경우 FP로 간주
            y_true.append(0)
        
        # 해당 예측 바운딩 박스의 score 값을 y_scores에 추가
        y_scores.append(pred_bboxes[i, 4])  # pred_bboxes의 5번째 값은 score

    # 각 바운딩 박스에 대한 Precision-Recall 계산 (간단히 TP/FP/FN 계산 후 처리)
    precision, recall, _ = precision_recall_curve(y_true, y_scores)  # y_true는 GT, y_scores는 예측된 스코어
    
    avg_precision = np.mean(precision)  # 평균 Precision 값 (optional)

    return precision, recall, avg_precision

def main():
    # 모델 및 설정 파일 로드
    config_file = './projects/configs/co_dino/co_dino_5scale_lsj_swin_large_1x_coco.py'
    cfg = Config.fromfile(config_file)
    checkpoint_file = './work_dirs/co_dino_5scale_lsj_swin_large_3x_coco/epoch_12.pth' # 수정
    cfg.gpu_ids = [1]  # GPU ID 설정
    cfg.model.query_head.num_classes = 10
    cfg.model.roi_head[0].bbox_head.num_classes = 10
    cfg.model.bbox_head[0].num_classes = 10
    cfg.model.train_cfg = None  # 학습 관련 설정을 제거 (inference 전용)

    model = init_detector(cfg, checkpoint_file, device='cuda:0')

    # dataset 설정
    # COCO 형식의 JSON 파일 경로
    root = '/data/ephemeral/home/dataset/'  # root 경로 설정
    json_file = root + 'val_split_0.json'

    # COCO API 초기화
    coco = COCO(json_file)

    # 이미지 ID 리스트 가져오기
    image_ids = coco.getImgIds()

    # validation 데이터셋 생성
    validation_dataset = []

    for img_id in image_ids:
        # 이미지 메타데이터 가져오기
        img_info = coco.loadImgs(img_id)[0]
        img_path = root + img_info['file_name']
        
        # 해당 이미지의 annotation 가져오기
        ann_ids = coco.getAnnIds(imgIds=img_id)
        annotations = coco.loadAnns(ann_ids)
        
        # bounding box 및 class_id 저장
        gt_bboxes = []
        for ann in annotations:
            bbox = ann['bbox']  # [x_min, y_min, width, height]
            class_id = ann['category_id']  # class ID
            # COCO 형식에 맞게 [x_min, y_min, x_max, y_max] 형식으로 변환
            x_min = float(bbox[0])
            y_min = float(bbox[1])
            x_max = float(bbox[0]) + float(bbox[2])
            y_max = float(bbox[1]) + float(bbox[3])
            gt_bboxes.append([class_id, x_min, y_min, x_max, y_max])

        # validation 데이터셋에 이미지 경로 및 GT bbox 추가
        validation_dataset.append({'img_path': img_path, 'gt_bboxes': gt_bboxes})

    # 데이터셋 예측 및 시각화
    for item in validation_dataset:
        img_path = item['img_path']
        gt_bboxes = item['gt_bboxes']

        # 예측 실행
        result = inference_detector(model, img_path)
        
        # 스코어 기반 결과 필터링
        score_threshold = 0.3
        filtered_results = []
        for class_index, class_bboxes in enumerate(result):
            # 각 클래스의 bounding boxes에서 score가 threshold를 초과하는 것만 필터링
            filtered_bboxes = class_bboxes[class_bboxes[:, 4] > score_threshold]
            
            # 필터링된 bounding box에 클래스 정보 추가
            for bbox in filtered_bboxes:
                # [x_min, y_min, x_max, y_max, score, class_index] 형식으로 저장
                filtered_results.append(np.array([*bbox[:4], bbox[4], class_index]))

        # 클래스별 PR 곡선 계산
        precision_per_class, recall_per_class, avg_precision_per_class = compute_pr_curve(gt_bboxes, filtered_results)
        
        # 이미지 및 PR 곡선 시각화
        visualize_image_and_pr_curves(img_path, filtered_bboxes, gt_bboxes, precision_per_class, recall_per_class, avg_precision_per_class)


if __name__ == '__main__':
    gc.collect()  # 파이썬의 가비지 컬렉터 실행
    torch.cuda.empty_cache()  # 캐시 메모리 비우기
    main()