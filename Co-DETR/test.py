import mmcv
from mmcv import Config
from mmdet.datasets import (build_dataloader, build_dataset)
from mmdet.models import build_detector
from mmdet.apis import single_gpu_test
from mmcv.runner import load_checkpoint
import os
from mmcv.parallel import MMDataParallel
import pandas as pd
from pycocotools.coco import COCO

def main():
    classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
               "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

    # Config 파일 경로
    config_file = './projects/configs/co_dino/co_dino_5scale_lsj_swin_large_1x_coco.py'
    cfg = Config.fromfile(config_file)

    # 경로 설정
    root = '/data/ephemeral/home/dataset'
    cfg.work_dir = './work_dirs/co_dino_5scale_lsj_swin_large_1x_coco_1'
    checkpoint_file = './work_dirs/co_dino_5scale_lsj_swin_large_1x_coco_1/epoch_12.pth'

    # Dataset 설정
    cfg.data.test.classes = classes
    cfg.data.test.img_prefix = root
    cfg.data.test.ann_file = os.path.join(root, 'test.json')
    # cfg.data.test.pipeline[1]['img_scale'] = (1024, 1024)  # 이미지 Resize 설정
    cfg.data.test.test_mode = True

    # 하이퍼파라미터 설정
    cfg.data.samples_per_gpu = 3
    cfg.data.workers_per_gpu = 2
    cfg.seed = 2021
    cfg.gpu_ids = [1]  # GPU ID 설정
    cfg.model.query_head.num_classes = 10
    cfg.model.roi_head[0].bbox_head.num_classes = 10
    cfg.model.bbox_head[0].num_classes = 10
    cfg.model.train_cfg = None  # 학습 관련 설정을 제거 (inference 전용)

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
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, checkpoint_file, map_location='cpu')
    model.CLASSES = dataset.CLASSES
    model = MMDataParallel(model.cuda(), device_ids=[0])

    # Test 진행
    output = single_gpu_test(model, data_loader, show_score_thr=0.05)

    # COCO API를 사용하여 예측 결과 후처리
    prediction_strings = []
    file_names = []
    coco = COCO(cfg.data.test.ann_file)
    img_ids = coco.getImgIds()

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
    submission_file = os.path.join(cfg.work_dir, 'co_dino_5scale_lsj_swin_large_3x_coco_1.csv')
    submission.to_csv(submission_file, index=False)
    print(f"Submission file saved to {submission_file}")

if __name__ == '__main__':
    main()
