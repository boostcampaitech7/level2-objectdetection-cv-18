import torch.nn as nn
import torch
import torch.optim as optim
import os
import argparse
import pandas as pd
import logging
import time
import torch.nn.functional as F 
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import StratifiedKFold
# from sklearn.ensemble import VotingClassifier
from torch.utils.tensorboard import SummaryWriter

from loss import CrossEntropyLoss
from model.model_selector import ModelSelector
from base.customize_layer import customize_layer
from curriculum.curriculum_dataloader import CustomDataset, TorchvisionTransform, AlbumentationsTransform
from curriculum.curriculum_trainer import Trainer

def set_cuda(gpu):
    torch.cuda.set_device(gpu)
    device = torch.device(f'cuda:{gpu}')
    print(f"is_available cuda : {torch.cuda.is_available()}")
    print(f"current use : cuda({torch.cuda.current_device()})\n")
    return device

def setup_directories(save_rootpath):
    # 가중치, 로그, TensorBoard 경로 설정
    weight_dir = os.path.join(save_rootpath, 'weights')
    log_dir = os.path.join(save_rootpath, 'logs')
    tensorboard_dir = os.path.join(save_rootpath, 'tensorboard')
    save_csv_dir =  os.path.join(save_rootpath, 'test_csv')

    # 디렉토리 생성 (존재하지 않으면 생성)
    os.makedirs(weight_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(save_csv_dir, exist_ok=True)

    return weight_dir, log_dir, tensorboard_dir, save_csv_dir

def inference(
    model: nn.Module,
    device: torch.device,
    test_loader: DataLoader
):
    model.to(device)
    model.eval()

    predictions = []
    with torch.no_grad(): 
        for images in tqdm(test_loader):
            images = images.to(device)

            # 모델을 통해 예측 수행
            # ensemble을 위해 스코어 벡터로 반환
            logits = model(images)
            logits = F.softmax(logits, dim=1)
            # preds = logits.argmax(dim=1)

            # 예측 스코어 벡터 저장
            # predictions.append(logits.cpu().numpy())

            # 예측 결과 저장
            predictions.extend(logits.cpu().detach().numpy())  # 결과를 CPU로 옮기고 리스트에 추가

    return predictions

# test
def test():
    device = set_cuda(args.gpu) 
    test_info = pd.read_csv(args.test_csv)

    if args.transform == "TorchvisionTransform":
        train_transform = TorchvisionTransform(is_train=True)
        val_transform = TorchvisionTransform(is_train=False)
    elif args.transform == "AlbumentationsTransform":
        train_transform = AlbumentationsTransform(is_train=True)
        val_transform = AlbumentationsTransform(is_train=False)

    test_dataset = CustomDataset(
        root_dir=args.test_dir,
        info_df=test_info,
        transform=val_transform,
        is_inference=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        drop_last=False
    )
    
    num_classes = 10

    model_selector = ModelSelector(
        model_type= args.model_type,
        num_classes = num_classes,
        model_name= args.model_name,
        pretrained= args.pretrained
    )

    model = model_selector.get_model()

    if args.pretrained == True:
        for param in model.parameters():
            param.requires_grad = False
    
        model = customize_layer(model, num_classes)
    
    model.load_state_dict(torch.load(args.weight))
    # 모델로 추론 실행
    predictions = inference(
        model=model,
        device=device,
        test_loader=test_loader
    )
    cs = np.max(predictions, axis=1)
    label_id = np.argmax(predictions, axis=1)

    class_to_new_id = {
        'General trash': 0,
        'Paper': 1,
        'Paper pack': 2,
        'Metal': 3,
        'Glass': 4,
        'Plastic': 5,
        'Styrofoam': 6,
        'Plastic bag': 7,
        'Battery': 8,
        'Clothing': 9
    }

    new_id_to_class = {v: k for k, v in class_to_new_id.items()}
    labels = [new_id_to_class[l_id] for l_id in label_id]

    df = pd.read_csv(args.test_csv)

    df['label'] = labels
    df['label_id'] = label_id
    df['cs'] = cs

    df.to_csv(args.save_path, index=False)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=1, help='cuda:(gpu)')
    
    # default 부분 수정해서 사용!
    # k_fold로 돌리기 위한 코드, 기존 코드와 달라진 부분이 있어 확인 후 사용 바람

    # method
    parser.add_argument('--model_type', type=str, default='timm', help='사용할 모델 이름 : model_selector.py 중 선택')
    parser.add_argument('--model_name', type=str, default='eva02_large_patch14_448.mim_in22k_ft_in22k_in1k', help='model/timm_model_name.txt 에서 확인, 아키텍처 확인은 "https://github.com/huggingface/pytorch-image-models/tree/main/timm/models"')
    parser.add_argument('--pretrained', type=bool, default=True, help='전이학습 or 학습된 가중치 가져오기 : True / 전체학습 : False')
    # 전이학습할 거면 꼭! (True) customize_layer.py 가서 레이어 수정, 레이어 수정 안할 거면 가서 레이어 구조 변경 부분만 주석해서 사용 (어떤 레이어 열지는 알아야함)
    # 모델 구조랑 레이어 이름 모르겠으면 위에 모델 정의 부분가서 print(model) , assert False 주석 풀어서 확인하기

    parser.add_argument('--transform', type=str, default='AlbumentationsTransform', help='transform class 선택 torchvision or albumentation / dataloader.py code 참고')
    
    # 데이터 경로
    parser.add_argument('--test_dir', type=str, default="/hdd1/lim_data/level2_dataset", help='테스트 데이터셋 루트 디렉토리 경로') # "/data/ephemeral/home/data/test"
    parser.add_argument('--test_csv', type=str, default="/home/hwang/leem/level2-objectdetection-cv-18/2-stages/classification/csv/WBF_ATSS_test.csv", help='테스트 데이터셋 csv 파일 경로') # "/data/ephemeral/home/data/test.csv"
    parser.add_argument('--weight', type=str, default="/home/hwang/leem/level2-objectdetection-cv-18/2-stages/level1-imageclassification-cv-18/main/Experiments/curriculum_garbage_2/weights/4_bestmodel_accu.pt")

    parser.add_argument('--save_path', type=str, default="/home/hwang/leem/level2-objectdetection-cv-18/2-stages/classification/csv/100_classification_WBF_ATSS.csv")
   
    # 하이퍼파라미터
    args = parser.parse_args()

    start_time = time.time()
    test()
    end_time = time.time()

    print(f" End : {(end_time - start_time)/60} min")