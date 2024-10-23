import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches

# 데이터를 Pandas로 읽어오기
data = './norm_predictions.csv'

# DataFrame 생성
df = pd.read_csv(data)

# 클래스와 색상 정의
classes = (
    "General trash", "Paper", "Paper pack", "Metal", "Glass", 
    "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"
)

# 클래스에 숫자와 색상 매핑
class_mapping = {name: idx for idx, name in enumerate(classes)}
color_map = {
    "General trash": 'red',
    "Paper": 'blue',
    "Paper pack": 'green',
    "Metal": 'yellow',
    "Glass": 'purple',
    "Plastic": 'orange',
    "Styrofoam": 'cyan',
    "Plastic bag": 'magenta',
    "Battery": 'brown',
    "Clothing": 'pink',
}

# Bounding box 시각화 함수
def visualize_and_save_bbox(row):
    img_path = os.path.join('/data/ephemeral/home/dataset', row['image_id'])
    img = Image.open(img_path)
    print(img.size)
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    splitted = row['PredictionString'].split()
    
    # Bounding box 좌표와 레이블
    class_ids = splitted[::6]
    x_min, y_min, x_max, y_max = splitted[2::6], splitted[3::6], splitted[4::6], splitted[5::6]
    
    # Bounding box 생성 및 추가
    for i in range(len(x_min)):
        # 클래스 이름으로 색상 및 ID 가져오기
        class_name = classes[int(class_ids[i])]  # 숫자로 클래스 이름 가져오기
        color = color_map.get(class_name, 'gray')  # 색상 매핑
        rect = patches.Rectangle((float(x_min[i]), float(y_min[i])), 
                                 float(x_max[i]) - float(x_min[i]), 
                                 float(y_max[i]) - float(y_min[i]), 
                                 linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        
        # 레이블 추가
        ax.text(float(x_min[i]), float(y_min[i]) - 5, class_name, 
                color=color, fontsize=12, weight='bold', 
                bbox=dict(facecolor='white', alpha=0.5, edgecolor=color, boxstyle='round,pad=0.3'))

    # 이미지 저장 (저장 경로는 원하는 경로로 수정)
    output_path = os.path.join('output_fixed', os.path.basename(img_path))
    plt.axis('off')  # 축 제거
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# 이미지가 저장될 output 디렉토리 생성
os.makedirs('output_fixed', exist_ok=True)

# 모든 행에 대해 시각화 및 저장
for _, row in df.iloc[::-1].iterrows():
    visualize_and_save_bbox(row)