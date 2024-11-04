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

def visualize_image_and_pr_curves(img_path, predictions, correct_bboxes, precision_per_class, recall_per_class, avg_precision_per_class):
    classes = ["General trash", "Paper", "Paper pack", "Metal", "Glass", 
               "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"]
    
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print("img_size : ", img.shape)
    
    # 정답 bounding box 그리기 (빨간색으로 표시)
    for bbox in correct_bboxes:
        x_min, y_min, x_max, y_max = map(int, bbox[1:5])
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        class_id = int(bbox[0])
        cv2.putText(img, f'GT: {classes[class_id]}', (x_min, y_min + 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
    # 예측에 대한 bounding box 그리기 파란색으로 표시)
    for bbox in predictions:
        x_min, y_min, x_max, y_max = map(int, bbox[2:])
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
        class_id = int(bbox[0])
        cv2.putText(img, f'PR: {classes[class_id]}', (x_min, y_min + 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # 전체 figure 설정
    plt.figure(figsize=(15, 6))
    
    # 이미지 서브플롯
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title("Detection Results\nRed: Ground Truth, Blue: Predictions")
    
    # PR 커브 서브플롯
    plt.subplot(1, 2, 2)
    
    # 클래스별 PR 커브 그리기
    colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))
    mAP50 = 0
    valid_classes = 0
    
    for class_id in range(len(classes)):
        if class_id in precision_per_class and len(precision_per_class[class_id]) > 0:
            plt.plot(recall_per_class[class_id], precision_per_class[class_id],
                    color=colors[class_id], 
                    label=f'{classes[class_id]} (AP50={avg_precision_per_class[class_id]:.3f})')
            mAP50 += avg_precision_per_class[class_id]
            valid_classes += 1
    
    # mAP50 계산 및 표시
    if valid_classes > 0:
        mAP50 /= valid_classes
        plt.title(f'Precision-Recall Curves\nmAP50: {mAP50:.3f}')
    else:
        plt.title('Precision-Recall Curves')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim(-0.2, 1.2)  # y축 범위를 -0.2부터 1.2까지 설정
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

    plt.tight_layout()
    # plt.show()

    # 저장 경로가 존재하지 않으면 디렉토리 생성
    save_dir = './work_dirs/visualization/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.savefig(f'./work_dirs/visualization/{os.path.basename(img_path)}_pr_curves.png')  # 파일로 저장
    plt.close()  # 현재 figure 닫기

# IoU 계산 함수
def calculate_iou(box1, box2):
    """
    box: [x_min, y_min, x_max, y_max]
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_area = max(0, x2_inter - x1_inter + 1) * max(0, y2_inter - y1_inter + 1)
    
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    
    union_area = box1_area + box2_area - inter_area
    
    iou = inter_area / union_area
    return iou

def calculate_map50_with_class(gt_boxes, pred_boxes, num_classes=10, iou_threshold=0.5):
    """
    대회 방식의 mAP50 계산 함수
    
    Args:
        gt_boxes: Ground Truth boxes [class_id, x_min, y_min, x_max, y_max]
        pred_boxes: Predicted boxes [class_id, score, x_min, y_min, x_max, y_max]
        num_classes: 클래스 개수
        iou_threshold: IoU 임계값 (기본값 0.5)
    """
    precision_per_class = {}
    recall_per_class = {}
    avg_precision_per_class = {}
    
    # 이미지에 실제로 존재하는 클래스 확인
    existing_classes = set(box[0] for box in gt_boxes)
    
    for class_id in range(num_classes):
        # 현재 클래스의 GT와 예측 박스 필터링
        class_gt_boxes = [box for box in gt_boxes if box[0] == class_id]
        class_pred_boxes = [box for box in pred_boxes if box[0] == class_id]
        
        # GT 박스가 없는 경우 (현재 클래스가 이미지에 존재하지 않는 경우)
        if len(class_gt_boxes) == 0:
            if class_id in existing_classes:
                # GT는 있지만 예측이 없는 경우
                precision_per_class[class_id] = np.array([1.0])
                recall_per_class[class_id] = np.array([0.0])
                avg_precision_per_class[class_id] = 0.0
            continue
            
        # 예측이 없는 경우
        if len(class_pred_boxes) == 0:
            precision_per_class[class_id] = np.array([0.0])
            recall_per_class[class_id] = np.array([0.0])
            avg_precision_per_class[class_id] = 0.0
            continue
            
        # 예측 박스 점수 기준 내림차순 정렬
        class_pred_boxes = sorted(class_pred_boxes, key=lambda x: x[1], reverse=True)
        
        # TP, FP 배열 초기화
        num_gt = len(class_gt_boxes)
        num_pred = len(class_pred_boxes)
        tp = np.zeros(num_pred)
        fp = np.zeros(num_pred)
        gt_detected = set()  # 이미 매칭된 GT 박스 추적
        
        # 각 예측에 대해 TP/FP 판정
        for pred_idx, pred_box in enumerate(class_pred_boxes):
            max_iou = -1
            max_iou_gt_idx = -1
            
            # GT 박스들과 IoU 계산
            for gt_idx, gt_box in enumerate(class_gt_boxes):
                if gt_idx in gt_detected:
                    continue
                    
                iou = calculate_iou(pred_box[2:], gt_box[1:])
                if iou > max_iou:
                    max_iou = iou
                    max_iou_gt_idx = gt_idx
            
            # IoU threshold를 넘는 매칭이 있으면 TP
            if max_iou >= iou_threshold and max_iou_gt_idx not in gt_detected:
                tp[pred_idx] = 1
                gt_detected.add(max_iou_gt_idx)
            else:
                fp[pred_idx] = 1
        
        # 누적 TP, FP 계산
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        
        # Recall과 Precision 계산
        recall = cum_tp / num_gt
        precision = cum_tp / (cum_tp + cum_fp)
        
        # AP 계산 (모든 점에서의 interpolation)
        # COCO style AP 계산 방식 사용
        recall = np.concatenate(([0.], recall, [1.]))
        precision = np.concatenate(([0.], precision, [0.]))
        
        # precision을 우측에서 좌측으로 최대값으로 설정
        for i in range(precision.size - 1, 0, -1):
            precision[i - 1] = max(precision[i - 1], precision[i])
        
        # recall 변화가 있는 지점들을 찾아서 AP 계산
        i = np.where(recall[1:] != recall[:-1])[0]
        ap = np.sum((recall[i + 1] - recall[i]) * precision[i + 1])
        
        # 결과 저장
        precision_per_class[class_id] = precision[:-1]
        recall_per_class[class_id] = recall[:-1]
        avg_precision_per_class[class_id] = ap
    
    return precision_per_class, recall_per_class, avg_precision_per_class

def main():
    # 모델 및 설정 파일 로드
    config_file = './projects/configs/co_dino/co_dino_5scale_lsj_swin_large_1x_coco.py'
    cfg = Config.fromfile(config_file)
    checkpoint_file = './work_dirs/co_dino_5scale_lsj_swin_large_3x_coco/fold3_epoch_12.pth' # 수정
    cfg.gpu_ids = [1]  # GPU ID 설정
    cfg.model.query_head.num_classes = 10
    cfg.model.roi_head[0].bbox_head.num_classes = 10
    cfg.model.bbox_head[0].num_classes = 10
    cfg.model.train_cfg = None  # 학습 관련 설정을 제거 (inference 전용)

    model = init_detector(cfg, checkpoint_file, device='cuda:0')

    # dataset 설정
    # COCO 형식의 JSON 파일 경로
    root = '../../../project2/dataset/'  # root 경로 설정 /data/ephemeral/home/dataset/
    json_file = root + 'Stratified_Split_Train_data/val_split_3.json'

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
                # [class_index, score, x_min, y_min, x_max, y_max] 형식으로 저장
                bbox = np.array([class_index, bbox[4], *bbox[:4]])
                # print(class_index)
                filtered_results.append(bbox)
        
        # 클래스별 PR 곡선 계산
        precision_per_class, recall_per_class, avg_precision_per_class = calculate_map50_with_class(gt_bboxes, filtered_results)
        
        # 이미지 및 PR 곡선 시각화
        visualize_image_and_pr_curves(img_path, filtered_results, gt_bboxes, precision_per_class, recall_per_class, avg_precision_per_class)


if __name__ == '__main__':
    gc.collect()  # 파이썬의 가비지 컬렉터 실행
    torch.cuda.empty_cache()  # 캐시 메모리 비우기
    main()