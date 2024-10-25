# 참고 : https://github.com/dungnb1333/global-wheat-dection-2020/blob/main/utils.py

import pandas as pd
import os
import numpy as np
import json
from pathlib import Path

# pascal dataset을 coco dataset으로 변경
def make_pseudo_dataframe(test_p,train_p=None):

    # data_root = 'D:\AI-BOOSTCAMP-7TH\level2-objectdetection-cv-18\tools\ensemble\non_maximum_weighted_result.csv'
    p = Path.cwd()
    train_p = p.joinpath('target/train.json')
    test_p = p.joinpath('ensemble/non_maximum_weighted_result.csv')

    # 파일을 열기
    # Train data의 meta data를 불러와서 test_data에서 달라지지 않는 부분은 mapping 시킬 것이다.
    with open(train_p,'r') as f:
        train_data = json.load(f)
    test_data = pd.read_csv(test_p)

    # 저장할 coco_data format 만들기
    coco_data = {
        "images":[],
        "annotations":[],
        "categories":[]
    }

    for image_idx, image_id in enumerate(list(np.unique(test_data.image_id.values))):
        
        # predict_string 전처리
        # print(row)
        predict_string = test_data[test_data['image_id'] == image_id]['PredictionString']

        # 예외처리 : PredictionString이 아예 비어있을 경우 이 부분을 넘김
        if len(predict_string) == 0 : continue

        # predict_string을 " "으로 split함
        predict_list = str(predict_string).split(" ")

        # 혹시 모를 사태를 대비해 predict_list flatten
        predict_list = np.reshape(predict_list, (-1,6)) # class_id, score, xmin, ymin, xmax, ymax


        # coco_data["images"] 추가
        coco_data["images"].append({
            "id":image_idx,
            "file_name":image_id
        })

        coco_data['categories'] = train_data['categories']

        for anno_id, predict in enumerate(predict_list[:, :].tolist()): #xmin, ymin, xmax, ymax
            # Bounding Box (COCO는 [x, y, width(xmax-xmin), height(ymax-ymin)])변환
            label = predict[0]
            score = predict[1]
            pascal_bbox = predict[2:6]
            coco_bbox = [0]*4
            coco_bbox[0] = float(pascal_bbox[0])
            coco_bbox[1] = float(pascal_bbox[1])
            coco_bbox[2] = float(pascal_bbox[2]) - float(pascal_bbox[0]) # width
            coco_bbox[3] = float(pascal_bbox[3]) - float(pascal_bbox[1]) # height

            coco_data['annotations'].append({
                'id': anno_id,
                'image_id': image_idx,
                'category_id': coco_data['categories'][label],
                'bbox': coco_bbox
            })
    return coco_data

    # prediction String을 파싱한다.
    # Bounding Box (COCO는 [x, y, width(xmax-xmin), height(ymax-ymin)])변환
    
        # for i in range(0,len(prediction_string),6):
            # print(i)
        # label, scores, boxes=test_df.loc[test_df.image_id == image_id,'PredictionString']
    #     boxes, scores = 
    #     if boxes.shape[0] == 0:
    #         result = {
    #             'image_path': os.path.join(TEST_DIR, image_id+'.jpg'),
    #             'xmin': None,
    #             'ymin': None,
    #             'xmax': None,
    #             'ymax': None,
    #             'isbox': False
    #         }
    #         results.append(result)
    #     else:
    #         for box in boxes:
    #             result = {
    #                 'image_path': os.path.join(TEST_DIR, image_id+'.jpg'),
    #                 'xmin': box[0],
    #                 'ymin': box[1],
    #                 'xmax': box[2],
    #                 'ymax': box[3],
    #                 'isbox': True
    #             }
    #             results.append(result)
    # pseudo_df = pd.DataFrame(results, columns=['image_path', 'xmin', 'ymin', 'xmax', 'ymax', 'isbox'])
    

    # img_paths = []
    # for image_id in df.image_id.values:
    #     img_paths.append(os.path.join(TRAIN_DIR, image_id+'.jpg'))
    # df['image_path'] = np.array(img_paths)
    # valid_df = df.loc[df['fold'] == PSEUDO_FOLD]
    # train_df = df.loc[~df.index.isin(valid_df.index)]
    # valid_df = valid_df.loc[valid_df['isbox']==True]
    
    # train_df = train_df[['image_path','xmin','ymin','xmax','ymax','isbox']].reset_index(drop=True)
    # valid_df = valid_df[['image_path','xmin','ymin','xmax','ymax','isbox']].reset_index(drop=True)

    # train_df = pd.concat([train_df, pseudo_df], ignore_index=True).sample(frac=1).reset_index(drop=True)
    # train_df.to_csv('train.csv', index=False)
    # valid_df.to_csv('valid.csv', index=False)


make_pseudo_dataframe()