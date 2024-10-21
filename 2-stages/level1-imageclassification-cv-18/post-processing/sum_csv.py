import pandas as pd

# 여러 CSV 파일을 하나로 합치는 함수
def merge_prediction_strings(file_paths):
    # 모든 파일을 읽어와서 리스트에 저장
    dfs = [pd.read_csv(file_path) for file_path in file_paths]
    
    # 첫 번째 파일을 기준으로 병합
    merged_df = dfs[0]

    # 나머지 파일들을 반복하여 병합
    for df in dfs[1:]:
        # image_id 기준으로 병합하고, PredictionString을 문자열로 결합
        merged_df = pd.merge(merged_df, df, on='image_id', suffixes=('', '_other'))
        
        # 요소를 띄어쓰기 없이 합친 후 마지막에만 공백 추가
        merged_df['PredictionString'] = merged_df[['PredictionString', 'PredictionString_other']].apply(
            lambda x: ''.join(x) + ' ', axis=1)  # 요소들을 띄어쓰기 없이 합친 후 마지막에만 공백 추가
        merged_df.drop(columns=['PredictionString_other'], inplace=True)  # 임시 열 제거

    return merged_df

# 결과 CSV 저장 함수
def save_merged_csv(merged_df, output_path):
    merged_df.to_csv(output_path, index=False)

# 실행 예시
file_paths = [
    '/home/hwang/leem/level2-objectdetection-cv-18/2-stages/level1-imageclassification-cv-18/post-processing/co_dino_5sca...o_12e.csv',
    '/home/hwang/leem/level2-objectdetection-cv-18/2-stages/level1-imageclassification-cv-18/post-processing/WBF_ATSS.csv'
]  # 여러 파일 경로
output_path = 'concat_output2.csv'
merged_df = merge_prediction_strings(file_paths)
save_merged_csv(merged_df, output_path)