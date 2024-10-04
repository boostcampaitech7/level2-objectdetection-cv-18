# requirments import

import random
import numpy as np
import os
from sklearn.model_selection import train_test_split
import json
from tqdm import tqdm
import copy
import torch
import pandas as pd
from detectron2.data import detection_utils as utils
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_test_loader, build_detection_train_loader
import detectron2.data.transforms as T


# config 불러오기
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/retinanet_R_101_FPN_3x.yaml'))

# train, test 둘 다 해당하는 내용 수정
cfg.DATA_DIR = '/data/ephemeral/home/dataset/'
cfg.OUTPUT_DIR = 'output/retinanet_R_101_FPN_3x_epoch15000'
cfg.OUTPUT_EVAL_DIR = '/output_eval/retinanet_R_101_FPN_3x_epoch15000'
cfg.DATALOADER.NUM_WOREKRS = 2

# Register Dataset
# train / test를 구분할 필요는 없다. (method에서 해결)

# 1. train_set, val_set split
def split_train_and_val():

    # 전처리
    # COCO 형식의 train.json 로드
    with open(os.path.join(cfg.DATA_DIR,'train.json'), 'r') as f:
        coco_data = json.load(f)
    
    # image_id 별로 annotations를 묶기
    image_annotations = dict()
    for anno in coco_data['annotations']:
        image_id = anno['image_id']
        if image_id not in image_annotations:
            image_annotations[image_id] = []
        image_annotations[image_id].append(anno)
        
    # 이미지 리스트 추출
    image_ids = list(image_annotations.keys())

    # train, val split
    train_idx, val_idx = train_test_split(image_ids, test_size=0.25, random_state=42)
    train_image_ids = [image_ids[i] for i in train_idx]
    val_image_ids = [image_ids[i] for i in val_idx]
    
    # Train과 Val에 해당하는 annotations 필터링
    train_annotations = [anno for image_id in train_image_ids for anno in image_annotations[image_id]]
    val_annotations = [anno for image_id in val_image_ids for anno in image_annotations[image_id]]
    
    # Train과 Val에 해당하는 이미지 필터링
    train_images = [img for img in coco_data['images'] if img['id'] in train_image_ids]
    val_images = [img for img in coco_data['images'] if img['id'] in val_image_ids]

    # train fold json 생성
    train_data = coco_data.copy()
    train_data['annotations'] = train_annotations
    train_data['images'] = train_images
    with open(os.path.join(cfg.DATA_DIR,'split_train.json'), 'w') as f:
        json.dump(train_data, f)
    
    # validation fold json 생성
    val_data = coco_data.copy()
    val_data['annotations'] = val_annotations
    val_data['images'] = val_images
    with open(os.path.join(cfg.DATA_DIR, 'split_val.json'), 'w') as f:
        json.dump(val_data, f)

split_train_and_val()

try:
    # name, metadata, jsonfile, image_root
    register_coco_instances('coco_trash_train', {}, os.path.join(cfg.DATA_DIR, 'split_train.json'), cfg.DATA_DIR)
except AssertionError:
    pass

try:
    register_coco_instances('coco_trash_val', {}, os.path.join(cfg.DATA_DIR, 'split_val.json'), cfg.DATA_DIR)
except AssertionError:
    pass

try:
    # name, metadata, jsonfile, image_root
    register_coco_instances('coco_trash_test', {}, os.path.join(cfg.DATA_DIR, 'test.json'), cfg.DATA_DIR)
except AssertionError:
    pass

MetadataCatalog.get('coco_trash_train').thing_classes = ["General trash", "Paper", "Paper pack", "Metal", 
                                                         "Glass", "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"]


# train config 수정하기
cfg.DATASETS.TRAIN = ('coco_trash_train',)
cfg.DATASETS.TEST = ('coco_trash_val',)
cfg.DATALOADER.NUM_WOREKRS = 2

cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-Detection/retinanet_R_101_FPN_3x.yaml')

cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.MAX_ITER = 15000
cfg.SOLVER.STEPS = (8000,12000)
cfg.SOLVER.GAMMA = 0.005
cfg.SOLVER.CHECKPOINT_PERIOD = 3000

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.RETINANET.NUM_CLASSES = 10


# test config 수정하기
cfg.TEST.EVAL_PERIOD = 3000



# seed 고정 : 42
# Detectron2에서 특정 레이어가 새롭게 추가되거나 클래스 수가 달라질 때 랜덤으로 초기화되는 경우를 막음
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

seed_everything()


# mapper - input data를 어떤 형식으로 return할지 (따라서 augmnentation 등 데이터 전처리 포함 됨)
def trainMapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)
    image = utils.read_image(dataset_dict['file_name'], format='BGR')
    
    transform_list = [
        T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
        T.RandomBrightness(0.8, 1.8),
        T.RandomContrast(0.6, 1.3)
    ]
    
    image, transforms = T.apply_transform_gens(transform_list, image)
    
    dataset_dict['image'] = torch.as_tensor(image.transpose(2,0,1).astype('float32'))
    
    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop('annotations')
        if obj.get('iscrowd', 0) == 0
    ]
    
    instances = utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict['instances'] = utils.filter_empty_instances(instances)
    
    return dataset_dict


# mapper - input data를 어떤 형식으로 return할지
def testMapper(dataset_dict):
    
    dataset_dict = copy.deepcopy(dataset_dict)
    image = utils.read_image(dataset_dict['file_name'], format='BGR')
    
    dataset_dict['image'] = image
    
    return dataset_dict

# trainer - DefaultTrainer를 상속
class MyTrainer(DefaultTrainer):
    
    @classmethod
    def build_train_loader(cls, cfg, sampler=None):
        return build_detection_train_loader(
        cfg, mapper = trainMapper, sampler = sampler
        )
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            os.makedirs(cfg.OUTPUT_EVAL_DIR, exist_ok = True)
            output_folder = cfg.OUTPUT_EVAL_DIR
            
        return COCOEvaluator(dataset_name, cfg, False, output_folder)

# train
def train():
    os.makedirs(cfg.OUTPUT_DIR, exist_ok = True)

    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


def test():

    # config 수정하기
    cfg.DATASETS.TEST = ('coco_trash_test',)
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3

    # model
    predictor = DefaultPredictor(cfg)

    # test loader
    test_loader = build_detection_test_loader(cfg, 'coco_trash_test', testMapper)

    print(test_loader)
    # output 뽑은 후 sumbmission 양식에 맞게 후처리 
    prediction_strings = []
    file_names = []

    for data in tqdm(test_loader):
        
        prediction_string = ''
        
        data = data[0]
        
        outputs = predictor(data['image'])['instances']
        
        targets = outputs.pred_classes.cpu().tolist()
        boxes = [i.cpu().detach().numpy() for i in outputs.pred_boxes]
        scores = outputs.scores.cpu().tolist()
        
        for target, box, score in zip(targets,boxes,scores):
            prediction_string += (str(target) + ' ' + str(score) + ' ' + str(box[0]) + ' ' 
            + str(box[1]) + ' ' + str(box[2]) + ' ' + str(box[3]) + ' ')
        
        prediction_strings.append(prediction_string)
        file_names.append(data['file_name'].replace(cfg.DATA_DIR,''))

    return prediction_strings, file_names

def submission(prediction_strings, file_names, save_file_name = 'submission_det.csv'):
    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names
    submission.to_csv(os.path.join(cfg.OUTPUT_DIR, save_file_name), index=None)

if __name__ == "__main__":
    # print(cfg.MODEL)
    split_train_and_val()
    cfg.DATASETS.TRAIN = ('coco_trash_train',)
    cfg.DATASETS.TEST = ('coco_trash_val',)
    train()
    # prediction_strings, file_names = test()
    # submission(prediction_strings, file_names)