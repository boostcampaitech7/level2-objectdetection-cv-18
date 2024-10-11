import json
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches
import os

# JSON 파일 경로
json_path = '/data/ephemeral/home/jiwan/dataset/train.json'

# 이미지 파일이 저장된 기본 경로
base_image_path = '/data/ephemeral/home/dataset'

# 저장할 폴더 경로 (저장할 폴더가 없으면 생성)
output_dir = '/data/ephemeral/home/jiwan/level2-objectdetection-cv-18/YOLOv5/runs/train_img'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# JSON 파일 읽기
with open(json_path, 'r') as f:
    data = json.load(f)

# JSON 데이터에서 이미지와 주석 정보 추출
images = data['images']
annotations = data['annotations']
categories = data['categories']

# 카테고리 ID와 이름을 매칭하는 딕셔너리 생성
category_dict = {cat['id']: cat['name'] for cat in categories}

# 이미지에 바운딩 박스 그리기 함수 (저장 및 라벨 추가)
def save_bounding_box_image(image_id, annotations, images):
    # 이미지 정보 찾기
    image_data = next((item for item in images if item["id"] == image_id), None)
    if image_data is None:
        print(f"No image data found for image_id: {image_id}")
        return

    # 이미지 파일 경로 (기본 경로와 file_name을 합침)
    image_path = os.path.join(base_image_path, image_data['file_name'])
    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
        return

    # 이미지 열기
    image = Image.open(image_path)

    # 이미지를 그리기 위한 figure 생성
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # 주어진 이미지에 해당하는 annotation 찾기
    for ann in annotations:
        if ann['image_id'] == image_id:
            bbox = ann['bbox']
            category_id = ann['category_id']
            label = category_dict.get(category_id, 'Unknown')

            # 바운딩 박스 그리기 (x, y, width, height) - 두께 조정 (linewidth=3)
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

            # 라벨 텍스트 추가 (배경색 없음)
            plt.text(bbox[0], bbox[1] - 10, label, color='white', fontsize=12)

    # 저장할 파일 경로 설정
    save_path = os.path.join(output_dir, f"{os.path.basename(image_path)}")

    # 이미지 저장
    plt.savefig(save_path)
    plt.close()  # 화면에 표시하지 않고 figure 닫기

# 모든 이미지에 대해 바운딩 박스 그리기 및 저장
for image in images:
    save_bounding_box_image(image['id'], annotations, images)

print(f"Images with bounding boxes and labels saved to {output_dir}")
