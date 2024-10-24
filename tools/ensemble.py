# python 3.4 버전 이상에서만 작동함
import argparse
import os
from pathlib import Path

from ensemble_boxes import *
import numpy as np
import pandas as pd
from tqdm import tqdm


def set_parser():
    """
    Set basic parser
    백그라운드로 동시에 여러 ensemble을 진행할 수 있게 option을 만들었습니다

    기본값은 ensemble.py가 있는 폴더 안에 target/ensemble 폴더가 있고
        target : ensemble할 csv를 저장해 놓는 장소
        ensemble : ensemble한 결과값이 나오는 장소
    로 설정되어 있습니다. 밑의 -t 와 -o의 default 값을 바꾸거나, 파일이 실행될 때 옵션을 줌으로서 변경할 수 있습니다.

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
    parser.add_argument('-t', '--target_directory', type=str, default=p.joinpath('target'), help="앙상블을 진행할 csv가 있는 directory")
    parser.add_argument('-o', '--output_directory', type=str, default=p.joinpath('ensemble'), help="앙상블한 csv가 저장될 directory")
    parser.add_argument('-w', '--width', type=int, default=1024, help="이미지 사이즈 크기")
    parser.add_argument('-l','--height', type=int, default=1024, help="이미지 사이즈 높이")
    return parser

def return_image_ids(output_dir):
    output_list = os.listdir(output_dir)
    csv_data = pd.read_csv(os.path.join(output_dir, output_list[0]))
    return list(csv_data['image_id'])

def save_target_data(target_dir, output_dir, meta_file_name = 'meta_file.md'):
    p = Path(output_dir)
    with open(p.joinpath(meta_file_name), 'a') as f:
        f.write("Ensemble target: \n")
        target_list = os.listdir(target_dir)
        for target_file in target_list:
            f.write(f'- {target_file}\n')
    print("Sucess: save to target name")


def save_output_data(submission, output_dir, file_name, error_msg = None, meta_file_name = 'meta_file.md'):
    p = Path(output_dir)
    try:
        os.makedirs(output_dir, exist_ok=True)
        submission_file = os.path.join(output_dir, file_name)
        submission.to_csv(submission_file, index=False) # csv file로 변환 & 저장

        condition = 'Success'
        if error_msg:
            condition = 'Error'        
        with open(p.joinpath(meta_file_name), 'a') as f:
            f.write(f"{condition}: Submission file saved to {file_name}\n")
            if error_msg: f.write(f'{error_msg}\n')
        print("Sucess: save to status and Ensemble file")
    except Exception as e:
        print(f"Error: ")
        print(e)
    return submission_file

# def group_csv_with_normalization(predict_list, image_width, image_height):

#     predict_list = np.reshape(predict_list, (-1, 6))
#     box_list = []

#     for box_idx, box in enumerate(predict_list[:, 2:6].tolist()):
#         # box의 각 좌표를 float형으로 변환한 후 image의 넓이와 높이로 각각 정규화

#         box[0] = float(box[0]) / image_width
#         box[1] = float(box[1]) / image_height
#         box[2] = float(box[2]) / image_width
#         box[3] = float(box[3]) / image_height
#         box_list.append(box)
#     score_list = list(map(float, predict_list[:, 1].tolist()))
#     label_list = list(map(int, predict_list[:, 0].tolist()))
#     return box_list, score_list, label_list

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
    for idx, csv_data in enumerate(csv_datas):
        predict_to_list = csv_data[csv_data['image_id'] == image_id]['PredictionString'].tolist()
        if len(predict_to_list) == 0: continue # 예측 못했을 경우 예외처리
        else: predict_string = predict_to_list[0]
        predict_list = str(predict_string).split()

        # 결측값 및 하나만 있는 predict는 아예 이용하지 않는다.
        if len(predict_list)==0 or len(predict_list)==1:
            continue

        predict_list = np.reshape(predict_list, (-1, 6))
        box_list = []

        for box in predict_list[:, 2:6].tolist():
            # assert 코드가 발생하면 강제로라도 변환 불가 -> 시간 자원 아까움
            # # box의 각 좌표를 float형으로 변환한 후 image의 넓이와 높이로 각각 정규화
            # if float(box[0]) > image_width or float(box[2]) > image_width:
            #     print('out of image_width',idx,box[0],box[2])
            #     assert False
            # elif float(box[1]) > image_height or float(box[3]) > image_height:
            #     print('out of image height',idx,box[1], box[3])
            #     assert False
            box[0] = float(box[0]) / image_width
            box[1] = float(box[1]) / image_height
            box[2] = float(box[2]) / image_width
            box[3] = float(box[3]) / image_height
            box_list.append(box)


        boxes_list.append(box_list)
        scores_list.append(list(map(float, predict_list[:, 1].tolist())))
        labels_list.append(list(map(int, predict_list[:, 0].tolist())))
    return boxes_list, scores_list, labels_list

# csv 파일 별로 image_width, image_height를 주입하고 그걸 하나로 묶는 코드를 완성해야 한다.

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

def check_prediction_over(boxes):
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
    
    # 파일 이름 글로벌하게 만들기
    file_name = f'{ensemble_name}_error.csv'

    # 에러 메세지 저장
    error_msg = None

    # ensemble한 target data 이름 저장
    try:
        save_target_data(target_dir=target, output_dir=output)
    except:
        print("Error to save target name")

    try: 
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
            submission.loc[image_idx, 'PredictionString'] = predictions
        
        file_name = f'{ensemble_name}_result.csv'

    # 통상적인 Exception
    except Exception as e:
        print(image_idx, "has a problem")
        file_name = f'{ensemble_name}_error_{image_idx}.csv'
        error_msg = e

    # 강제로 Exception을 발생시킨 경우
    # else로 검출이 안되는 점을 주의
    except BaseException as e:
        print("Force!!!")
        print(f"Image {image_idx} has a problem")
        file_name = f'{ensemble_name}_error_{image_idx}.csv'
        error_msg = 'force exit'

    # 예외와 관계없이 실행
    finally:
        print(error_msg, file_name)
        submission_file = save_output_data(submission, output, file_name, error_msg = error_msg) # make submission file and save meta data
        print(f"Submission file saved to {submission_file}")

if __name__ == '__main__': 
    main()