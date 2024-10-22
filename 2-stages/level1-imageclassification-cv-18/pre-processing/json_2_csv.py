import os
import json
import cv2
import csv

# category_id에 해당하는 클래스 이름 리스트
category_names = [
    "General trash", "Paper", "Paper pack", "Metal", "Glass", 
    "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"
]

# train.json 파일 경로 설정
json_file_path = '/hdd1/lim_data/level2_dataset/train_split_4.json'

# 잘라낸 이미지를 저장할 폴더 경로 설정
output_folder = '/hdd1/lim_data/level2_dataset/crop_train_4'

# 학습 데이터를 저장할 CSV 파일 경로 설정
csv_file_path = '/hdd1/lim_data/level2_dataset/csv/train_4.csv'

# output_folder가 없으면 생성
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# json 파일 읽기
with open(json_file_path, 'r') as f:
    data = json.load(f)

# 이미지 파일 경로가 있는 상위 폴더 설정 (예: 'train/')
image_folder = '/hdd1/lim_data/level2_dataset'

# CSV 파일 생성 및 쓰기 시작
with open(csv_file_path, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    
    # 첫 번째 행에 헤더 작성
    # csvwriter.writerow(['class_name', 'image_path', 'category_id'])
    
    # 이미지 정보와 annotation 정보를 반복하면서 bbox 영역을 잘라내기
    for annotation in data['annotations']:
        # annotation에서 필요한 정보 추출
        image_id = annotation['image_id']
        id = annotation['id']
        bbox = annotation['bbox']
        category_id = annotation['category_id']
        
        # bbox 좌표 (x, y, width, height) 가져오기
        x_min, y_min, width, height = bbox

        # x_min = int(x_min - 0.15 * width)  # 좌측으로 15% 확장
        # y_min = int(y_min - 0.15 * height)  # 위로 15% 확장
        # width = int(width * 1.30)  # 너비 30% 확장
        # height = int(height * 1.30)  # 높이 30% 확장
        
        # 해당하는 이미지 파일 정보 찾기
        image_info = next(img for img in data['images'] if img['id'] == image_id)
        image_file_name = image_info['file_name']
        image_path = os.path.join(image_folder, image_file_name)
        
        # 이미지 읽기
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"이미지를 불러올 수 없습니다: {image_path}")
            continue
        
        # bbox 영역 자르기
        x_min = int(x_min)
        y_min = int(y_min)
        width = int(width)
        height = int(height)

        # 이미지 경계 내로 제한
        # x_min = max(0, x_min)
        # y_min = max(0, y_min)
        # width = min(image.shape[1] - x_min, width)
        # height = min(image.shape[0] - y_min, height)


        cropped_image = image[y_min:y_min+height, x_min:x_min+width]
        
        # 잘라낸 이미지 크기를 448x448로 리사이즈
        # resized_image = cv2.resize(cropped_image, (448, 448))
        
        # 잘라낸 이미지 저장 경로 설정 (파일명 형식: image_id_category_id_cropped.jpg)
        cropped_image_file_name = f"{image_id}_{id}_{category_id}.jpg"
        cropped_image_path = os.path.join(output_folder, cropped_image_file_name)
        relative_image_path = os.path.join('crop_train_4', cropped_image_file_name)
        
        # 이미지 저장
        if cropped_image is None or cropped_image.size == 0:
            print(f"error: {image_path}")
        else:
            cv2.imwrite(cropped_image_path, cropped_image)
        
        # class_name 가져오기
        class_name = category_names[category_id]
        
        # CSV 파일에 한 행 추가 (class_name, cropped_image_path, category_id)
        csvwriter.writerow([class_name, relative_image_path, category_id])
        
        print(f"이미지 저장 완료 및 CSV 기록: {cropped_image_path}")

print(f"CSV 파일 저장 완료: {csv_file_path}")