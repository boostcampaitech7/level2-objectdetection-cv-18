# 참고 : https://github.com/dungnb1333/global-wheat-dection-2020/blob/main/utils.py

import pandas as pd
import os
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm

def change_pascal_to_coco(test_p):
    """
    pascal_voc 제출 데이터를 coco 제출 데이터로 만든다.
    pascal_voc 제출 데이터는 다음과 같이 구성되어 있다.
    column: image_id, PrecisionString
    PredictionString = (label, score, xmin, ymin, xmax, ymax)

    Args:
        test_p (str): pascal_voc로 만들어진 csv 파일 경로

    Returns:
        coco_format (dict): coco_format 으로 만들어진 dict
    """
    # data_root = 'D:\AI-BOOSTCAMP-7TH\level2-objectdetection-cv-18\tools\ensemble\non_maximum_weighted_result.csv'
    p = Path.cwd()
    test_p = p.joinpath('ensemble/non_maximum_weighted_result.csv')
    test_data = pd.read_csv(test_p)

    # 저장할 coco_data format 만들기
    coco_data = {
        "images":[],
        "annotations":[],
        "categories":[]
    }

    for image_idx, image_id in enumerate(tqdm(list(np.unique(test_data.image_id.values)))):
        
        # predict_string 전처리
        predict_string = test_data[test_data['image_id'] == image_id]['PredictionString'].tolist()
        # 예외처리 : PredictionString이 아예 비어있을 경우 이 부분을 넘김
        if len(predict_string) == 0 : continue
        else: predict_string = predict_string[0]

        # predict_string을 " "으로 split함
        predict_list = str(predict_string).split(" ")

        # 혹시 모를 사태를 대비해 predict_list flatten
        predict_list = np.reshape(predict_list, (-1,6)) # class_id, score, xmin, ymin, xmax, ymax


        # coco_data["images"] 추가
        coco_data["images"].append({
            "id":image_idx,
            "file_name":image_id
        })

        # coco_data['categories'] 추가
        categories =  set(map(int, predict_list[:,0].tolist()))
        for category_id, category_name in enumerate(categories):
            coco_data['categories'].append({
                'id': category_id,
                'name': category_name
            })

        # coco_data['annotations'] 추가
        for anno_id, predict in enumerate(predict_list[:, :].tolist()): #xmin, ymin, xmax, ymax
            # Bounding Box (COCO는 [x, y, width(xmax-xmin), height(ymax-ymin)])변환
            label = predict[0]
            pascal_bbox = predict[2:6]
            coco_bbox = [0]*4
            coco_bbox[0] = float(pascal_bbox[0])
            coco_bbox[1] = float(pascal_bbox[1])
            coco_bbox[2] = float(pascal_bbox[2]) - float(pascal_bbox[0]) # width
            coco_bbox[3] = float(pascal_bbox[3]) - float(pascal_bbox[1]) # height

            coco_data['annotations'].append({
                'id': anno_id,
                'image_id': image_idx,
                'category_id': label,
                'bbox': coco_bbox
            })

    print("저장 중...")
    with open(test_p.parent.joinpath('test.json'),'w') as f:
        json.dump(coco_data, f, indent=4)
    return coco_data

change_pascal_to_coco('D:\AI-BOOSTCAMP-7TH\level2-objectdetection-cv-18\tools\ensemble\non_maximum_weighted_result.csv')

