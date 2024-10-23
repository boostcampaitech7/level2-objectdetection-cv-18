# 출처 : https://github.com/dungnb1333/global-wheat-dection-2020/blob/main/utils.py

import pandas as pd
import os
import numpy as np
import json

# def make_pseudo_dataframe(test_df, output_dict, TEST_DIR, df, TRAIN_DIR, PSEUDO_FOLD):
def make_pseudo_dataframe(test_df, output_dict, TEST_DIR):
    results = []
    for image_id in list(np.unique(test_df.image_id.values)):
        boxes, scores = output_dict[image_id]
        if boxes.shape[0] == 0:
            result = {
                'image_path': os.path.join(TEST_DIR, image_id+'.jpg'),
                'xmin': None,
                'ymin': None,
                'xmax': None,
                'ymax': None,
                'isbox': False
            }
            results.append(result)
        else:
            for box in boxes:
                result = {
                    'image_path': os.path.join(TEST_DIR, image_id+'.jpg'),
                    'xmin': box[0],
                    'ymin': box[1],
                    'xmax': box[2],
                    'ymax': box[3],
                    'isbox': True
                }
                results.append(result)
    pseudo_df = pd.DataFrame(results, columns=['image_path', 'xmin', 'ymin', 'xmax', 'ymax', 'isbox'])
    
    print(pseudo_df)

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

with open('D:\AI-BOOSTCAMP-7TH\level2-objectdetection-cv-18\dataset\train.json', 'w') as f:
    train_df = json.load(f)

# test_df = pd.read_csv('')