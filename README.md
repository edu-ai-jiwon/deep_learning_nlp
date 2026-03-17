## Huggingface_pipeline 
- transformers 라이브러리 활용

------------

### 감성 분석 (1)
- 모델: `sangrimlee/bert-base-multilingual-cased-nsmc`
- task: Text Classification (sentiment-analysis)
- 특징: 네이버 영화 리뷰(NSMC) 데이터로 학습된 한국어 전용 모델
- 분류: positive / negative

**1차 구현**
<img width="1725" height="671" alt="image" src="https://github.com/user-attachments/assets/4f5a2b6d-ab7e-4411-9cb8-e8d0ef1d914e" />

**2차 구현(수정)**
<img width="1760" height="768" alt="image" src="https://github.com/user-attachments/assets/1e86877c-ce80-49f3-a0e4-3778d03d9479" />


------------
### 감성 분석 (2)
- 모델: `Seonghaa/korean-emotion-classifier-roberta`
- task: Text Classification
- 특징: klue/roberta-base 기반 파인튜닝, 6가지 감정 분류
- 분류: 기쁨 / 슬픔 / 분노 / 불안 / 당황 / 평온

**1차 구현**
<img width="1748" height="618" alt="image" src="https://github.com/user-attachments/assets/1f0f9c32-0f34-409f-8cf0-7f5288fe0de4" />
**2차 구현(수정)**

<img width="1468" height="310" alt="image" src="https://github.com/user-attachments/assets/5e952806-0e7f-4a35-a3b5-03988b8f547e" />

------------
### 구현
![huggingface+구현](https://github.com/user-attachments/assets/a3c0a0b1-a52e-4dec-8f0f-e7dced5f9f5d)



