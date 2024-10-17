
from ensemble_boxes import *
import argparse
import os
import pandas as pd
from tqdm import tqdm


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
        
        # 분할
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
        output += f'{label} {score} {x*image_size} {y*image_size} {w*image_size} {h*image_size} '
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
    
    parser.add_argument('-n', '--name', type=str, default='weighted_boxes_fusion', help="앙상블 할 method")
    parser.add_argument('-i', '--iou_thr', type=float, default=0.5, help="iou threshold")
    parser.add_argument('-sbt', '--skip_box_thr', type=float, default=0.0001, help="skip box threshold")
    parser.add_argument('-sig','--sigma', type=float, default=0.1, help="시그마 값")
    parser.add_argument('-o', '--output_directory', type=str, default='/data/ephemeral/home/euna/level2-objectdetection-cv-18/Co-DETR/work_dirs/test', help="앙상블한 csv가 저장될 장소")
    parser.add_argument('-s','--size', type=int, default=1024, help="이미지 사이즈")
    return parser

def main():
    parser = set_parser()
    args = parser.parse_args()
    
    ensemble_name = args.name
    iou_thr = args.iou_thr
    skip_box_thr = args.skip_box_thr
    sigma = args.sigma
    output_dir = args.output_directory
    image_size = args.size

    # submission format 만들기
    submission = pd.DataFrame()
    image_ids = return_image_ids(output_dir)
    
    submission['PredictionString'] = ''
    submission['image_id'] = image_ids
    
    for image_idx in tqdm(range(len(image_ids))):
        boxes, scores, labels = make_ensemble_format_per_image(image_idx, output_dir, image_size=image_size)

        # 결측값 제거시 weights가 달라질 수 있음
        weights = [1] * len(labels)
        
        if ensemble_name == 'nms':
            results = nms(boxes, scores, labels, weights=weights, iou_thr=iou_thr)
        elif ensemble_name == 'soft_nms':
            results = soft_nms(boxes, scores, labels, weights=weights, iou_thr=iou_thr, sigma=sigma, thresh=skip_box_thr)
        elif ensemble_name == 'non_maximum_weighted':
            results = non_maximum_weighted(boxes, scores, labels, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
        else:
            results = weighted_boxes_fusion(boxes, scores, labels, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)    
        predictions = prediction_format_per_image(*results, image_size=image_size)
        submission['PredictionString'][image_idx] = predictions

    submission_file = os.path.join(output_dir, f'result.csv')
    submission.to_csv(submission_file, index=False)
    print(f"Submission file saved to {submission_file}")

if __name__ == '__main__': 
    main()