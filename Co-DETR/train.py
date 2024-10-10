import os
import logging
import torch
import heapq
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

        # # 모델 파일명 생성 (val accuracy, mAP@50, epoch, val loss 기준)
        # acc_filename = os.path.join(self.work_dir, f"checkpoint_acc_{val_acc:.4f}_mAP50_{mAP50:.4f}_epoch_{epoch}_loss_{val_loss:.4f}.pth")
        # loss_filename = os.path.join(self.work_dir, f"checkpoint_loss_{val_loss:.4f}_mAP50_{mAP50:.4f}_epoch_{epoch}_acc_{val_acc:.4f}.pth")

        # # 먼저 모델을 저장
        # torch.save(runner.model.state_dict(), acc_filename)
        # torch.save(runner.model.state_dict(), loss_filename)

        # best 모델만 저장
        acc_filename = os.path.join(self.work_dir, f"best_acc_model.pth")
        loss_filename = os.path.join(self.work_dir, f"best_loss_model.pth")

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


def main():
    """
    train_detector를 사용하여 모델을 훈련하는 메인 함수.
    """
    # Config 파일 로드 및 수정
    cfg = Config.fromfile('./projects/configs/co_dino_vit/co_dino_5scale_vit_large_coco.py')  # 모델 설정
    cfg.work_dir = './work_dirs/co_dino_5scale_vit_large_coco'                      # 로그/모델 저장 위치
    # cfg.optimizer.type = 'SGD'                                                     # optimizer 설정
    # cfg.optimizer.lr = 0.02                                                        # lr 설정
    # cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)                # gradient clipping 설정
    cfg.data.samples_per_gpu = 3                                                    # 배치 크기 설정
    # cfg.runner = dict(type='EpochBasedRunner', max_epochs=12)                      # epoch 수 설정
    cfg.seed = 2022                                                                 # 랜덤 시드 설정
    cfg.gpu_ids = [0]                                                               # 사용할 GPU 설정
    cfg.device = get_device()                                                       # 디바이스 설정 (GPU 또는 CPU)

    # 모델 경량화
    cfg.model.backbone.use_act_checkpoint = False                                   # 백본 체크포인트 사용 안함
    cfg.model.backbone.img_size = 512                                               # 백본 이미지 사이즈 설정
    cfg.data.train.pipeline[3]['policies'][0][0]['img_scale'] = [(256, 256),             # 트레인 이미지 스케일 설정 
                                                            (284, 284),
                                                            (312, 312),
                                                            (341, 341),
                                                            (369, 369),
                                                            (398, 398),
                                                            (426, 426),
                                                            (455, 455),
                                                            (483, 483),
                                                            (512, 512)]
    cfg.data.train.pipeline[3]['policies'][1][0]['img_scale'] = [(400, 400),             # 트레인 이미지 스케일 설정 
                                                            (500, 500),
                                                            (600, 600)]
    cfg.data.train.pipeline[3]['policies'][1][2]['img_scale'] = [(256, 256),             # 트레인 이미지 스케일 설정 
                                                            (284, 284),
                                                            (312, 312),
                                                            (341, 341),
                                                            (369, 369),
                                                            (398, 398),
                                                            (426, 426),
                                                            (455, 455),
                                                            (483, 483),
                                                            (512, 512)]
    cfg.data.val.pipeline[1]['img_scale'] = (512, 512)                                # 테스트 이미지 스케일 설정
    cfg.data.test.pipeline[1]['img_scale'] = (512, 512)                                # 테스트 이미지 스케일 설정

    # TensorBoard 로그 및 텍스트 로그 설정
    cfg.log_config = dict(
        interval=50,  # 로그 기록 간격 (배치 단위)
        hooks=[
            dict(type='TextLoggerHook'),  # 텍스트 기반 로그 출력 (콘솔용)
            dict(type='TensorboardLoggerHook')  # TensorBoard 로그 기록
        ]
    )

    # 커스텀 Hook 추가 (설정 파일에 등록)
    cfg.custom_hooks = [dict(type='SaveTopModelsHook', work_dir=cfg.work_dir)]

    # Dataset 설정
    classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
               "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

    root = '/data/ephemeral/home/dataset/'  # root 경로 설정
    cfg.data.train.classes = classes
    cfg.data.train.img_prefix = root
    cfg.data.train.ann_file = root + '/train_split_0.json'
    cfg.data.val.classes = classes
    cfg.data.val.img_prefix = root
    cfg.data.val.ann_file = root + '/val_split_0.json'

    # Train과 Val 데이터셋 각각 빌드
    train_dataset = build_dataset(cfg.data.train)
    val_dataset = build_dataset(cfg.data.val)

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
    main()
