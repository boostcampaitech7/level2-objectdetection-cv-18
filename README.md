<div align="right">
  <a href="https://hits.seeyoufarm.com"><img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https://github.com/boostcampaitech7/level2-objectdetection-cv-18&count_bg=%23C6D2FF&title_bg=%23555555&icon=&icon_color=%23FFFFFF&title=hits&edge_flat=false"/></a>
</div>

| 커밋 유형 | 의미 |
| :-: | -|
|feat|	새로운 기능 추가|
|fix|	버그 수정|
|docs	|문서 수정|
|style|	코드 formatting, 세미콜론 누락, 코드 자체의 변경이 없는 경우|
|refactor	|코드 리팩토링|
|test|	테스트 코드, 리팩토링 테스트 코드 추가|
|chore|	패키지 매니저 수정, 그 외 기타 수정 ex) .gitignore|
|design|	CSS 등 사용자 UI 디자인 변경|
|comment	|필요한 주석 추가 및 변경|
|rename|	파일 또는 폴더 명을 수정하거나 옮기는 작업만인 경우|
|remove|	파일을 삭제하는 작업만 수행한 경우|
|!BREAKING |CHANGE	커다란 API 변경의 경우|
|!HOTFIX	|급하게 치명적인 버그를 고쳐야 하는 경우|


# 딥하조
![image](https://github.com/user-attachments/assets/1d61152d-0f72-499f-b70f-88ccdf21870d)
- 2024.10.02 ~ 2024.10.24
- 재활용 품목 분류를 위한 Object Detection
- Naver Connect & Upstage 주관 대회
- [CSV](https://docs.google.com/spreadsheets/d/1UjokS8UYo729eNL_m7iWYXZjsI_MPKUfhAe2mp3oBJQ/edit?usp=sharing)
## 💡 팀원 소개

| [![](https://avatars.githubusercontent.com/chan-note)](https://github.com/chan-note) | [![](https://avatars.githubusercontent.com/Donghwan127)](https://github.com/Donghwan127) | [![](https://avatars.githubusercontent.com/batwan01)](https://github.com/batwan01) | [![](https://avatars.githubusercontent.com/taehan79-kim)](https://github.com/taehan79-kim) | [![](https://avatars.githubusercontent.com/nOctaveLay)](https://github.com/nOctaveLay)  | [![](https://avatars.githubusercontent.com/Two-Silver)](https://github.com/Two-Silver)  |
| ---------------------------------------------------- | ------------------------------------------------------ | --------------------------------------------------- | ------------------------------------------------------- | ----------------------------------------------------- | ----------------------------------------------------- |
| [임찬혁](https://github.com/chan-note)                  | [서동환](https://github.com/Donghwan127)                  | 🦇[박지완](https://github.com/batwan01)          | [김태한](https://github.com/taehan79-kim)                  | 🐈[임정아](https://github.com/nOctaveLay)                  | 🐡[이은아](https://github.com/Two-Silver)                  |

## 대회 소개

![image](https://github.com/user-attachments/assets/c3f7a3e7-dffc-427e-ac34-57b2c4659b21)

분리수거는 이러한 환경 부담을 줄일 수 있는 방법 중 하나입니다. 잘 분리배출 된 쓰레기는 자원으로서 가치를 인정받아 재활용되지만, 잘못 분리배출 되면 그대로 폐기물로 분류되어 매립 또는 소각되기 때문입니다.

따라서 우리는 사진에서 쓰레기를 Detection 하는 모델을 만들어 이러한 문제점을 해결해보고자 합니다. 문제 해결을 위한 데이터셋으로는 일반 쓰레기, 플라스틱, 종이, 유리 등 10 종류의 쓰레기가 찍힌 사진 데이터셋이 제공됩니다.

여러분에 의해 만들어진 우수한 성능의 모델은 쓰레기장에 설치되어 정확한 분리수거를 돕거나, 어린아이들의 분리수거 교육 등에 사용될 수 있을 것입니다. 부디 지구를 위기로부터 구해주세요! 🌎

## 사용된 데이터셋 정보

- **데이터셋 이름**: Coco Data
- **출처**: [coco format 다운로드 링크](https://aistages-api-public-prod.s3.amazonaws.com/app/Competitions/000325/data/data.tar.gz)

![image](https://github.com/user-attachments/assets/a01e996d-39e2-45f7-9bee-4e7e837ae6b6)


![image](https://github.com/user-attachments/assets/2790b64a-26c1-4cc4-be7a-269b6121c567)

### 데이터셋 설명

**전체 이미지 개수 : 9754장**
- **학습 데이터**:  4883장
- **Private & Public 평가 데이터**: 4871장

```bash
├── dataset
    ├── train.json
    ├── test.json
    ├── train
    └── test
```

## Leaderboard

![image](https://github.com/user-attachments/assets/a6e460ca-b192-4db5-b9e8-39f0f685b84c)

![image](https://github.com/user-attachments/assets/8d6f47e8-58b6-4a0a-82fd-8d5fe777493c)


## Tools
- Github ( Issue, Projects )
- Notion
- Slack
- Google Sheets, Presentation

## Project Timeline


## Models



## Augmentations


## Voting


