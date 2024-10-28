import json
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

data_path = 'jeonga/level2-objectdetection-cv-18/detectron2/output/metrics.json'
log_dir = 'jeonga/level2-objectdetection-cv-18/detectron2/log'

# 로그를 저장할 디렉토리 지정
writer = SummaryWriter(log_dir)

# # 로그 기록할 데이터의 예시 (반복해서 기록하는 형태)
with open(data_path, 'r') as f:
    logging = f.readlines()
    for step, data in tqdm(enumerate(logging)):
        logging_dict = json.loads(data)
        for key in logging_dict.keys():

            # bbox 부분엔 의미 없는 데이터가 들어가기 때문에, 삭제
            # 만약 bbox 부분이 필요하다면 if문을 지우면 된다.
            # 설령 bbox 부분이 없더라도 정상 작동한다.
            if key[:4] != 'bbox':
                writer.add_scalar(key, logging_dict[key], step)