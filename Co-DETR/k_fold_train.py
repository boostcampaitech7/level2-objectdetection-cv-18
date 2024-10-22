import os
import json
import logging
import torch
import heapq

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from mmcv import Config
from mmdet.apis import train_detector
from mmdet.models import build_detector
from mmdet.datasets import build_dataset
from mmdet.utils import get_device
from mmcv.runner import Hook
from mmcv.runner import HOOKS

@HOOKS.register_module()
class SaveTopModelsHook(Hook):
    """ 상위 3개 accuracy 및 loss 기준으로 모델을 저장하는 Hook """
    def __init__(self, work_dir):
        self.top3_val_acc = []  # accuracy 기준 상위 3개
        self.top3_val_loss = []  # loss 기준 상위 3개
        self.work_dir = work_dir

        # Logger 설정
        log_path = os.path.join(work_dir, 'train.log')
        logging.basicConfig(
            level=logging.INFO,  # 로그 레벨을 INFO로 설정
            format='%(asctime)s - %(levelname)s - %(message)s',  # 로그 형식
            handlers=[
                logging.FileHandler(log_path),  # 로그를 파일에 기록
                logging.StreamHandler()  # 로그를 콘솔에도 출력
            ]
        )
        self.logger = logging.getLogger()

    def save_top_model(self, metric_list, value, epoch, filename, mode='max'):
        """ 상위 3개 모델을 저장하고 관리하는 함수 """
        if mode == 'max':
            # 상위 3개 accuracy를 추적
            heapq.heappush(metric_list, (value, epoch, filename))
            if len(metric_list) > 3:
                # 상위 3개만 유지하고, 가장 낮은 값을 제거
                old_model = heapq.heappop(metric_list)
                os.remove(old_model[2])  # 파일 삭제
        else:
            # 상위 3개 loss를 추적 (loss는 작을수록 좋음)
            heapq.heappush(metric_list, (-value, epoch, filename))
            if len(metric_list) > 3:
                # 상위 3개만 유지하고, 가장 큰 값을 제거
                old_model = heapq.heappop(metric_list)
                os.remove(old_model[2])  # 파일 삭제

    def after_val_epoch(self, runner):
        # validation accuracy, loss, mAP50을 얻음
        val_acc = runner.log_buffer.output.get('accuracy', 0)
        val_loss = runner.log_buffer.output.get('loss', 0)
        mAP50 = runner.log_buffer.output.get('bbox_mAP_50', 0)  # mAP50 값
        epoch = runner.epoch + 1

        # 모델 파일명 생성 (val accuracy, mAP@50, epoch, val loss 기준)
        acc_filename = os.path.join(self.work_dir, f"checkpoint_acc_{val_acc:.4f}_mAP50_{mAP50:.4f}_epoch_{epoch}_loss_{val_loss:.4f}.pth")
        loss_filename = os.path.join(self.work_dir, f"checkpoint_loss_{val_loss:.4f}_mAP50_{mAP50:.4f}_epoch_{epoch}_acc_{val_acc:.4f}.pth")

        # 먼저 모델을 저장
        torch.save(runner.model.state_dict(), acc_filename)
        torch.save(runner.model.state_dict(), loss_filename)

        # 정확도 기준 상위 3개 가중치만 유지
        self.save_top_model(self.top3_val_acc, val_acc, epoch, acc_filename, mode='max')

        # 손실 기준 상위 3개 가중치만 유지
        self.save_top_model(self.top3_val_loss, val_loss, epoch, loss_filename, mode='min')

        # 훈련 및 검증 결과 로그 남기기
        train_loss = runner.log_buffer.output.get('loss', 0)  # 학습 중 loss는 runner.log_buffer에 저장됨
        train_acc = runner.log_buffer.output.get('accuracy', 0)  # accuracy도 log_buffer에서 가져올 수 있음
        self.logger.info(f"Epoch {epoch}/{runner.max_epochs}, "
                         f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}, "
                         f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}, "
                         f"Validation mAP@50: {mAP50:.4f}")

    def after_run(self, runner):
        # 학습 종료 후 필요한 정리 작업 수행
        self.logger.info("Training has ended.")


def split_train_and_val(n_splits):
    
    # load json
    data_root = '/data/ephemeral/home/dataset'
    with open(os.path.join(data_root, 'train.json')) as f:
        data = json.load(f)

    var = [(ann['image_id'], ann['category_id']) for ann in data['annotations']]

    X = np.ones((len(data['annotations']),1))   # train data, (n_samples, n_features) : annotation의 index로 구분하기 위함
    y = np.array([v[1] for v in var])           # target (n_samples,) : category_id
    groups = np.array([v[0] for v in var])      # group : image_id 이미지 별로 카테고리가 여러개 있는 것이기 때문

    # k-fold 크로스 밸리데이션 초기화, StratifiedGroupKFold:클래스 불균형을 해소하며, 이미지가 train/val set에 혼재하는 것 방지
    kf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=411)

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X, y, groups)):
        # Train과 Val image ids
        train_image_ids = groups[train_idx]
        val_image_ids = groups[val_idx]

        # 변경해야 하는 image 필드와 annotation 필드만 수정한다.
        # Train과 Val에 해당하는 images 필드 data 빌드
        train_images = [img for img in data['images'] if img['id'] in train_image_ids]
        val_images = [img for img in data['images'] if img['id'] in val_image_ids]

        # Train과 Val에 해당하는 annotations 필드 data 빌드
        train_annotations = [data['annotations'][idx] for idx in train_idx]
        val_annotations = [data['annotations'][idx] for idx in val_idx]

        # train fold json 생성
        train_data = data.copy()
        train_data['annotations'] = train_annotations
        train_data['images'] = train_images
        with open(os.path.join(data_root,f'train_{fold_idx}.json'), 'w') as f:
            json.dump(train_data, f)
        
        # validation fold json 생성
        val_data = data.copy()
        val_data['annotations'] = val_annotations
        val_data['images'] = val_images
        with open(os.path.join(data_root, f'val_{fold_idx}.json'), 'w') as f:
            json.dump(val_data, f)

def main(n_splits):
    """
    train_detector를 사용하여 모델을 훈련하는 메인 함수.
    """
    # Config 파일 로드 및 수정
    config_file_root = '/data/ephemeral/home/taehan/level2-objectdetection-cv-18/Co-DETR/projects/configs/co_dino/co_dino_5scale_lsj_swin_large_1x_coco.py'
    model_name = config_file_root.split('/')[-1][:-3]
    cfg = Config.fromfile(config_file_root)  # 모델 설정

    # Dataset 설정
    classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
               "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

    # 기본 학습 설정
    root = '/data/ephemeral/home/dataset'  # root 경로 설정
    cfg.data.train.classes = classes
    cfg.data.val.classes = classes
    cfg.data.samples_per_gpu = 2    


    cfg.model.backbone.use_checkpoint = True
    cfg.model.query_head.num_classes = 10
    cfg.model.roi_head[0].bbox_head.num_classes = 10
    cfg.model.bbox_head[0].num_classes = 10

    
    cfg.seed = 411                                                                  # 랜덤 시드 설정
    cfg.gpu_ids = [0]

    cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
    cfg.checkpoint_config = dict(max_keep_ckpts=3, interval=1)
    cfg.device = get_device()                # 디바이스 설정 (GPU 또는 CPU)


    # TensorBoard 로그 및 텍스트 로그 설정
    cfg.log_config = dict(
        interval=50,  # 로그 기록 간격 (배치 단위)
        hooks=[
            dict(type='TextLoggerHook'),  # 텍스트 기반 로그 출력 (콘솔용)
            dict(type='TensorboardLoggerHook')  # TensorBoard 로그 기록
        ]
    )

    # K-Fold를 위한 설정
    for fold_idx in range(n_splits):
        print(f"fold_idx : {fold_idx}")
        if fold_idx < 1:
            continue
        cfg.work_dir = f'./work_dirs/{model_name}_{fold_idx}'                                       # 로그/모델 저장 위치
        cfg.data.train.img_prefix = root
        cfg.data.train.ann_file = os.path.join(cfg.data.train.img_prefix,f'train_{fold_idx}.json')
        # cfg.data.train.pipeline[2]['img_scale'] = (512,512) # Resize

        cfg.data.val.img_prefix = root
        cfg.data.val.ann_file = os.path.join(cfg.data.val.img_prefix,f'val_{fold_idx}.json')
        # cfg.data.test.pipeline[1]['img_scale'] = (512,512) # Resize

        # Train 데이터셋 빌드
        train_dataset = build_dataset(cfg.data.train)

        # 모델 정의
        model = build_detector(cfg.model)
        model.init_weights()

        # work_dir 생성
        work_dir = cfg.work_dir
        os.makedirs(work_dir, exist_ok=True)

        # 모델 학습 시작
        train_detector(
            model=model,
            dataset=train_dataset,
            cfg=cfg,
            distributed=False,
            validate=True  # 검증 포함
        )

if __name__ == '__main__':
    n_splits = 5
    split_train_and_val(n_splits)
    main(n_splits)
