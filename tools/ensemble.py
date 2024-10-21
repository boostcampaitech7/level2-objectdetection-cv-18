# python 3.4 버전 이상에서만 작동함
import argparse
import os
from pathlib import Path

from ensemble_boxes import *
from pycocotools.coco import COCO
import numpy as np
import pandas as pd
from tqdm import tqdm


def return_image_ids(output_dir):
    output_list = os.listdir(output_dir)
    csv_data = pd.read_csv(os.path.join(output_dir, output_list[0]))
    return list(csv_data['image_id'])

# ensemble_boxes format 
def make_ensemble_format_per_image(image_id, output_dir, image_width, image_height):

    # output_dir에서 csv 파일 목록을 뽑아서 csv data를 csv_datas에 저장
    csv_datas = []
    output_list = os.listdir(output_dir)
    for output in output_list:
        csv_data = pd.read_csv(os.path.join(output_dir, output))
        csv_datas.append(csv_data)

    # csv_data에서 label, score, bbox들을 뽑고
    # 이를 labels, scores, bboxs에 저장한다.
    
    # 각 image id 별로 submission file에서 box좌표 추출
    boxes_list = []
    scores_list = []
    labels_list = []

    # 각 submission file 별로 prediction box좌표 불러오기
    for csv_data in csv_datas:
        predict_string = csv_data[csv_data['image_id'] == image_id]['PredictionString'].tolist()[0]
        predict_list = str(predict_string).split()

        # 결측값 및 하나만 있는 predict는 아예 이용하지 않는다.
        if len(predict_list)==0 or len(predict_list)==1:
            continue

        predict_list = np.reshape(predict_list, (-1, 6))
        box_list = []

        for box in predict_list[:, 2:6].tolist():
            # box의 각 좌표를 float형으로 변환한 후 image의 넓이와 높이로 각각 정규화

            box[0] = float(box[0]) / image_width
            box[1] = float(box[1]) / image_height
            box[2] = float(box[2]) / image_width
            box[3] = float(box[3]) / image_height
            box_list.append(box)

        boxes_list.append(box_list)
        scores_list.append(list(map(float, predict_list[:, 1].tolist())))
        labels_list.append(list(map(int, predict_list[:, 0].tolist())))
    return boxes_list, scores_list, labels_list

def prediction_format_per_image(boxes, scores, labels, image_width, image_height):
    output = ''
    for box, score, label in zip(boxes, scores, labels):
        
        # label이 float64로 찍히므로, 반드시 변환한다.
        label = int(label)

        # 좌표가 [0,1] 사이에 있지 않는 경우, 이로 맞춰준다.
        box[0] = min(max(box[0], 0), 1)
        box[1] = min(max(box[1], 0), 1)
        box[2] = min(max(box[2], 0), 1)
        box[3] = min(max(box[3], 0), 1)

        # normalize된 box의 좌표를 바꿔준다.
        output += f'{label} {score} {box[0]*image_width} {box[1]*image_height} {box[2]*image_width} {box[3]*image_height} '
    return output[:-1]


def set_parser():
    """
    Set basic parser
    
    Returns:
        ArgumentParser
    """
    parser = argparse.ArgumentParser(
                    prog="Ensemble",
                    description="Ensemble csv files")
    p = Path.cwd()
    parser.add_argument('-n', '--name', type=str, default='weighted_boxes_fusion', help="앙상블 할 method")
    parser.add_argument('-i', '--iou_thr', type=float, default=0.5, help="iou threshold")
    parser.add_argument('-sbt', '--skip_box_thr', type=float, default=0.0001, help="skip box threshold")
    parser.add_argument('-sig','--sigma', type=float, default=0.1, help="시그마 값")
    parser.add_argument('-t', '--target_directory', type=str, default=p.parent.joinpath('Co-DETR/work_dirs/test'), help="앙상블을 진행할 csv가 있는 directory")
    parser.add_argument('-o', '--output_directory', type=str, default=p.parent.joinpath('Co-DETR/work_dirs/ensemble'), help="앙상블한 csv가 저장될 directory")
    parser.add_argument('-w', '--width', type=int, default=1024, help="이미지 사이즈 크기")
    parser.add_argument('-l','--height', type=int, default=1024, help="이미지 사이즈 높이")
    return parser

def check_prediction(boxes):
    if len(boxes) == 0: return 0
    return np.array([boxes[0]>1, boxes[1]>1, boxes[2]>1, boxes[3]>1]).any()

def main():
    parser = set_parser()
    args = parser.parse_args()
    
    ensemble_name = args.name
    iou_thr = args.iou_thr
    skip_box_thr = args.skip_box_thr
    sigma = args.sigma
    target = args.target_directory
    output = args.output_directory
    image_width = args.width
    image_height = args.height

    # submission format 만들기
    submission = pd.DataFrame()
    image_ids = return_image_ids(target)
    
    submission['PredictionString'] = ''
    submission['image_id'] = image_ids

    for image_idx, image_id in enumerate(tqdm(image_ids)):
        
        boxes, scores, labels = make_ensemble_format_per_image(image_id, target, image_width = image_width, image_height = image_height)

        # 결측치 제거에 따라 예측 개수가 달라질 수 있다.
        # 모델에 따른 함수는 나중에 만들도록 한다.

        weights = [1] * len(labels)
        
        # 아예 box들이 예측되지 않는 경우 스킵한다.
        if len(boxes) == 0: continue

        # 앙상블 이름에 따라 분류
        if ensemble_name == 'nms':
            results = nms(boxes, scores, labels, weights=weights, iou_thr=iou_thr)
        elif ensemble_name == 'soft_nms':
            results = soft_nms(boxes, scores, labels, weights=weights, iou_thr=iou_thr, sigma=sigma, thresh=skip_box_thr)
        elif ensemble_name == 'non_maximum_weighted':
            results = non_maximum_weighted(boxes, scores, labels, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
        elif ensemble_name == 'weighted_boxes_fusion':
            results = weighted_boxes_fusion(boxes, scores, labels, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
        else: raise "no such ensemble name"
            
        predictions = prediction_format_per_image(*results, image_width = image_width, image_height = image_height)
        submission['PredictionString'][image_idx] = predictions
        
    os.makedirs(output, exist_ok=True)
    submission_file = os.path.join(output, f'{ensemble_name}_result.csv')
    submission.to_csv(submission_file, index=False)
    print(f"Submission file saved to {submission_file}")

if __name__ == '__main__': 
    main()