import os
import json

# JSON 파일 경로
json_file_path = '/data/ephemeral/home/dataset/train.json'

# 출력 디렉토리 설정
train_labels_dir = '/data/ephemeral/home/dataset/train/labels'

# 이미지 크기 설정
img_width = 1024
img_height = 1024

# JSON 파일 읽기
with open(json_file_path, 'r') as f:
    data = json.load(f)

# 이미지와 라벨 처리
image_labels = {}

# 어노테이션 수집
for annotation in data['annotations']:  # 'annotations' 키로 접근
    image_id = annotation['image_id']
    category_id = annotation['category_id']
    bbox = annotation['bbox']
    x_min, y_min, width, height = bbox

    # YOLO 형식으로 변환 (비율로 변환)
    x_center = (x_min + width / 2) / img_width
    y_center = (y_min + height / 2) / img_height
    norm_width = width / img_width
    norm_height = height / img_height

    # 이미지별 라벨 수집
    if image_id not in image_labels:
        image_labels[image_id] = []
    image_labels[image_id].append(f"{category_id} {x_center} {y_center} {norm_width} {norm_height}")

# 라벨 파일 저장
for image_id, labels in image_labels.items():
    # 4자리 숫자 형식으로 파일 이름 만들기
    image_name = f"{image_id:04d}.jpg"  # 예시: 2882 -> 2882.jpg
    label_file_path = os.path.join(train_labels_dir, f"{os.path.splitext(image_name)[0]}.txt")

    with open(label_file_path, 'w') as label_file:
        for label in labels:
            label_file.write(label + '\n')

    # 이미지 파일 복사 (여기선 이미지 파일 복사 예시가 필요해)
    # shutil.copy(f"original_images/{image_name}", train_images_dir)
