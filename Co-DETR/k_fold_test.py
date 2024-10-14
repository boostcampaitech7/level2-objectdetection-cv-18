import mmcv
from ensemble_boxes import *
from mmcv import Config
from mmdet.datasets import (build_dataloader, build_dataset)
from mmdet.models import build_detector
from mmdet.apis import init_detector, single_gpu_test
from mmcv.runner import load_checkpoint
import os
from mmcv.parallel import MMDataParallel
import pandas as pd
from pycocotools.coco import COCO

def set_base_config():
    classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
               "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

    # Config 파일 경로
    config_file = './configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
    cfg = Config.fromfile(config_file)
    model_name = config_file.split('/')[-1][:-3]
    
    # 경로 설정
    root = '/data/ephemeral/home/dataset'

    # Dataset 설정
    cfg.data.test.classes = classes
    cfg.data.test.img_prefix = root
    cfg.data.test.ann_file = os.path.join(root, 'test.json')
    cfg.data.test.pipeline[1]['img_scale'] = (512, 512)  # 이미지 Resize 설정
    cfg.data.test.test_mode = True

    # 하이퍼파라미터 설정
    cfg.data.samples_per_gpu = 4
    cfg.data.workers_per_gpu = 2
    cfg.seed = 42
    cfg.gpu_ids = [1]  # GPU ID 설정
    cfg.model.roi_head.bbox_head.num_classes = 10
    cfg.model.train_cfg = None  # 학습 관련 설정을 제거 (inference 전용)

    return cfg, model_name


def k_fold_test(n_splits):
    cfg, model_name = set_base_config()
    for fold_idx in range(n_splits):

        cfg.work_dir = f'./work_dirs/{model_name}_{fold_idx}'
        checkpoint_file = os.path.join(cfg.work_dir,'epoch_12.pth')
        
        # Dataset 및 DataLoader 생성
        dataset = build_dataset(cfg.data.test)
        data_loader = build_dataloader(
            dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False
        )

        # 모델 생성 및 체크포인트 로드
        model = init_detector(cfg, checkpoint_file)
        model.CLASSES = dataset.CLASSES
        model = MMDataParallel(model.cuda(), device_ids=[0])

        # Test 진행
        output = single_gpu_test(model, data_loader, show_score_thr=0.05) # class 별 output
        

        # COCO API를 사용하여 예측 결과 후처리
        prediction_strings = []
        file_names = []
        coco = COCO(cfg.data.test.ann_file)

        class_num = 10
        for i, out in enumerate(output):
            prediction_string = ''
            image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
            for j in range(class_num):
                for o in out[j]:
                    prediction_string += str(j) + ' ' + str(o[4]) + ' ' + str(o[0]) + ' ' + str(o[1]) + ' ' + str(
                        o[2]) + ' ' + str(o[3]) + ' '

            prediction_strings.append(prediction_string)
            file_names.append(image_info['file_name'])

        # 결과 저장
        submission = pd.DataFrame()
        submission['PredictionString'] = prediction_strings
        submission['image_id'] = file_names
        submission_file = os.path.join(cfg.work_dir, f'{model_name}_{fold_idx}_result.csv')
        submission.to_csv(submission_file, index=False)
        print(f"Submission file saved to {submission_file}")


if __name__ == '__main__':
    k_fold_test(2)