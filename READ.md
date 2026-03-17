## Huggingface_pipeline 
- transformers 라이브러리 활용

------------

### 감성 분석 (1)
- 모델: `sangrimlee/bert-base-multilingual-cased-nsmc`
- task: Text Classification (sentiment-analysis)
- 특징: 네이버 영화 리뷰(NSMC) 데이터로 학습된 한국어 전용 모델
- 분류: positive / negative

------------
### 감성 분석 (2)
- 모델: `Seonghaa/korean-emotion-classifier-roberta`
- task: Text Classification
- 특징: klue/roberta-base 기반 파인튜닝, 6가지 감정 분류
- 분류: 기쁨 / 슬픔 / 분노 / 불안 / 당황 / 평온
