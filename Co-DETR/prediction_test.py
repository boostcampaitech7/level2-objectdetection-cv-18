import pandas as pd

# 파일 불러오기
csv_file = pd.read_csv('D:\\AI-BOOSTCAMP-7TH\\level2-objectdetection-cv-18\\Co-DETR\\work_dirs\\test\\co_dino_5scale_lsj_swin_large_3x_coco_4.csv')

prediction_strings = csv_file['PredictionString']
for id_, predictions in enumerate(prediction_strings):
    prediction = predictions.split()
    for i in range(0,len(prediction),6):
        p_id, score, bbox = prediction[i], prediction[i+1], prediction[i+2:i+6]
        for x in bbox:
            x = float(x) / 1280
            if float(x) < 0 or float(x) > 1: print(id_,x)