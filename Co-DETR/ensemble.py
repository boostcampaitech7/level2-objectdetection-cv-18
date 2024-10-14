
from ensemble_boxes import *
import os
import pandas as pd
from tqdm import tqdm
from pycocotools.coco import COCO


def return_image_ids(output_dir):
    output_list = os.listdir(output_dir)
    csv_data = pd.read_csv(os.path.join(output_dir, output_list[0]))
    return list(csv_data['image_id'])

# ensemble_boxes format 
def make_ensemble_format_per_image(image_id, output_dir, image_size = 1024):

    # model 단위로 append
    labels_list = []
    scores_list = []
    boxes_list = []

    # output_dir에서 csv 파일 목록을 뽑아서 csv data를 csv_datas에 저장
    csv_datas = []
    output_list = os.listdir(output_dir)
    for output in output_list:
        csv_data = pd.read_csv(os.path.join(output_dir, output))
        csv_datas.append(csv_data)

    # csv_data에서 label, score, bbox들을 뽑고
    # 이를 labels, scores, bboxs에 저장한다.

    for csv_data in csv_datas:
        csv_data = csv_data.dropna(axis=0)
        try:
            prediction = csv_data['PredictionString'][image_id]
        except:
            continue

        labels = []
        scores = []
        bboxs = []
        
        # 결측치 제거
        original = prediction.split(' ')
        
        # label, score, box -> split
        for idx in range(len(original)//6):
            # 마지막 ' ' 예외 처리
            try:
                label, score, x,y,w,h = map(float, original[idx*6:idx*6+6])
                labels.append(int(label))
                scores.append(score)
                # bbox normalize
                bboxs.append([x/image_size,y/image_size,w/image_size,h/image_size])
            except: 
                print(f'unknwon_error in prediction parsing')
                continue

        boxes_list.append(bboxs)
        scores_list.append(scores)
        labels_list.append(labels)

    return boxes_list, scores_list, labels_list


def prediction_format_per_image(boxes, scores, labels, image_size = 1024):
    output = ''
    for idx in range(len(boxes)):
        label = int(labels[idx])
        score = scores[idx]
        x,y,w,h = boxes[idx]

        # normalize -> original
        output += f'{label} {score} {x*image_size} {y*image_size} {w*image_size} {h*image_size}'
    return output


def main():

    iou_thr = 0.5
    skip_box_thr = 0.0001
    sigma = 0.1
    output_dir = '/data/ephemeral/home/euna/level2-objectdetection-cv-18/Co-DETR/work_dirs/test'
    image_size = 1024

    # submission format 만들기
    submission = pd.DataFrame()
    image_ids = return_image_ids(output_dir)
    submission['image_id'] = image_ids
    submission['PredictionString'] = ''
    
    for image_idx in tqdm(range(len(image_ids))):
        boxes, scores, labels = make_ensemble_format_per_image(image_idx, output_dir, image_size=image_size)

        # 결측값 제거시 weights가 달라질 수 있음
        weights = [1] * len(labels)
        results = weighted_boxes_fusion(boxes, scores, labels, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)    
        predictions = prediction_format_per_image(*results, image_size=image_size)
        submission['PredictionString'][image_idx] = predictions

    submission_file = os.path.join(output_dir, f'result.csv')
    submission.to_csv(submission_file, index=False)
    print(f"Submission file saved to {submission_file}")

if __name__ == '__main__': 
    main()