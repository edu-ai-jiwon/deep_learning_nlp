#pip install streamlit
#streamlit run movie.py

import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

st.set_page_config(page_title="영화 감상 리뷰 감정 분석 ChatBOt", page_icon="🤗", layout="wide")
st.title("🤗 영화 감상 리뷰 감정 분석")
st.write("➡️ 라이브러리 활용: Transformers 라이브러리")
st.write("➡️ 총 2가지의 모델 사용")



# 영화 리뷰 34개
reviews = [
    "오랜만에 극장에서 볼 만한 영화 나왔다... 진짜 꼭 보세요!!! 올해 가장 잘한 일? 왕사남 개봉하자마자 극장 가서 본 일... 믿고 보는 유해진 역시는 역시...ㅠㅠ 웃기다가 울리다가 다하는 거 ㄹㅇ 반칙임;;;; 역사책 찢고 나온 열연 미쳤고 박지훈도 그냥 단종 그 자체... 나 과몰입해서 광릉에 별점 테러하러 간다. 아무도 막지마. 연휴에 부모님 모시고 가서 또 봐야지!!!",
    "아 ㅋㅋ 진짜 미치겠다 세조 싸움 잘함?",
    "유해진 조선사람 그자체라는 짤 보고갔는데도 너무 리얼해서 깜짝놀람 ㅋㅋㅋㅋ단종역할 배우도 처음봤는데 대배우 되겠더라 넘 재밌고 슬펐음 ㅜㅜ",
    "홍위도 울고 흥도도 울고 나도 울었다",
    "이동진 평론가가 극찬했다는 이유를 알 것 같아요 2026년 올해의 영화... 갑니다~!",
    "와 오랜만에 극장에서 볼만한 영화 나온듯배우들 연기 차력쇼 미쳤고, 보면서 눈물 콧물 다 흘림 진짜로.. 설 연휴에 부모님이랑 같이 보기 좋은 듯!!",
    "가족들과 한번 더 볼거에요.... 유해진 박지훈 연기 미쳤고, 웃고 우느라 속절없이 당해버림. 단종은 박지훈.",
    "오랜만에 사극 영화 보고 웃다가 울다가 과몰입한 영화!! 연기까지 완벽!! 설연휴 영화로 강추",
    "박지훈 퍼스널컬러는 단종이였다",
    "단종을 나약하게만 그린 것이 아니라 왕이었다는 것을 느끼게해준 영화. 유해진 박지훈의 조합이 이렇게나 감동을 주다니 여운이 깊게 남습니다.",
    "박지훈은 영화계 최고의 보석이다. 눈에 보석박음.. 눈만봐도 눈물..",
    "569년이 지났지만 단종은 모두가 응원하고 잊지 않을 존재로 기억되고 있네요. 세조도 무덤에서 일어나서 이 영화 보시길",
    "마지막에 오열해버렸습니다",
    "단종은 하고 싶어서 하는 역할이 아니라 주어진 자가 할 수 있는 역할이라는데 그 주어진 자가 박지훈인게 확실하다. 향후 10년동안은 단종하면 박지훈을 떠올릴듯",
    "이런 사극 오랜만인데 너무 좋았습니다 ㅠㅠ 박지훈 대배우될 거 같음",
    "오늘 이후로 김은희 남편이 아닌 <왕과 사는 남자>의 감독으로 불릴것! 결과를 우리 모두 다 알고 있기에 영화에서의 결말만큼이라도 성공하기를 바랬다 ㅠㅠ",
    "그냥.. 보세요.. 진짜 이런 감정을 느끼게 해준 영화가 얼마만인지... 감히 올해의 영화라 할 수 있다고 봅니다",
    "유해진 드디어 남우주연상에 오를듯,,,,마지막 클라이막스는 유해진이 아니면 살릴수없는 연기",
    "영월 광천리(광천골)가 시댁인 사람입니다.남편과 저는 단종에 대한 예의로다가 영화를 꼭 보자고 했었고 관람일기준으로 출산한지 60일째에 오전, 오후로 예매를 하고 한명씩 보고 왔습니다. 아기를 낳고 쭉 쪽잠만 자왔던 터라 보다가 잠들면 어쩌나 걱정하면서 보기시작했는데 처음부터 끝까지 한순간도 놓치지 않고 지루할 틈없이 큰재미와 감동받으며 영화를 보았습니다! 배우들 연기도 너무 좋았고 특히 유지태, 박지훈배우가 너무 좋았습니다.무엇보다 단종에 대해 재조명해 영화화해서 단종의 당시 상황을 입체적으로 표현한 부분이 좋았습니다.올해본 영화중에 1등입니다. 올해 첫 천만영화되길 바래요",
    "연기구멍은 호랑이 뿐, 배우들 연기가 미쳤습니다!! 그냥 엄흥도랑 단종 데려다놓은줄",
    "재미도 있지만, 엄마랑 둘이 광광 울었네요..박지훈님 그 눈망울이 계속 머릿속에 맴돌아요",
    "호감 배우들의 연기파티에 실화의 묵직함까지, 항주니 감독님의 최고의 영화가 될듯..",
    "유해진 박지훈 배우의 연기합이 매우 좋았습니다. 재밌게 잘봤어요",
    "불쌍한 왕이라고만 생각했는데 영화를 보고나니 어쩐지 미안한 마음이 들었다. 박지훈의 재발견",
    "박지훈 단종대왕 대상 줘라",
    "결혼이후 처음 울 남편 눈물을 보았다!!!!영월 꼭 가봐야지.배우 캐스팅 완벽!",
    "와.... 배우들 연기뭐에요? 정말 오랜만에 볼만한 전통사극 한편 나온듯요",
    "항준이형 드디어 해냈구나",
    "빈약한 서사 허술한 각본 엉성한 연출 감동적인 연기",
    "별점이 이렇게 다들 10점줄만큼 좋은영화인가요? 진심궁금합니다",
    "좋은 소재와 좋은 배우들의 연기력에 비해 낮은수준의 연출력이 아쉬움. 장이수와 참바다는 중복캐스팅, 과다한 애드립을 통제못하는 감독, 잦은 스토리의 끊김과 개연성 없는장면들.. 단종역은 별10개",
    "유해진이 작두타고 유지태가 호령하고 박지훈의 눈빛으로 완성되는 거장 등극 직전 감독 장항준의 역작.",
    "아쉬운 연출을 채워주는 박지훈의 눈빛, 유해진의 김치찌개. 홍위의 텅 빈 눈이 백성들의 사랑과 지지 속에 범의 눈이 되는 걸 지켜보면서, 이변없이 집권했다면 이런 성군이 되었겠구나, 비극으로 얼룩진 그의 삶에 색채가 덧입혀지는 느낌이다. 수염도 안 난 어린 것이 왔다며 한탄하던 흥도가 점점 홍위를 제 임금으로 모시게 되고, 아들처럼 품게 되고, 끝에는 나까지 흥도로 만든다. 저 어린 것이...",
    "배우들의 연기는 최고지만 연출과 편집이 뭔가 싶다.",
]


# 모델 로딩

@st.cache_resource
def sentiment_clf():
    movie_clf = pipeline("sentiment-analysis", model="sangrimlee/bert-base-multilingual-cased-nsmc")
    return movie_clf

@st.cache_resource
def emotion_clf():
    model_id = "Seonghaa/korean-emotion-classifier-roberta" #허깅 페이스에서 가져온 영화 리뷰 기반 모델
    device = 0 if torch.cuda.is_available() else -1
    clf = pipeline(
        "text-classification",
        model=model_id,
        tokenizer=model_id,
        device=device,
    )
    return clf


st.divider()


# 섹션 1: 감성분석 (1) 긍정 / 부정

st.header("😊 1. 긍정 / 부정 감성분석")
st.write("모델: `sangrimlee/bert-base-multilingual-cased-nsmc`")

input_mode1 = st.radio(
    "입력 방식 선택",
    ["직접 입력", "영화 리뷰 34개 분석"],
    key="input_mode1",
    horizontal=True,    # 선택지를 가로로 나열(X 세로로 나열됨)
)

if input_mode1 == "직접 입력":
    user_text1 = st.text_area("텍스트 입력", height=100, key="user_text1")
    texts_analyze1 = [user_text1] if user_text1.strip() else []
else:
    texts_analyze1 = reviews
    st.info(f"📋 영화 리뷰 {len(reviews)}개를 분석합니다.")

if st.button("분석하기", key="sentiment_btn"):
    if not texts_analyze1:
        st.warning("텍스트를 입력해주세요.")
    else:
        with st.spinner("분석 중..."):
            movie_clf = sentiment_clf()
            output = movie_clf(texts_analyze1)

        
        positive_count = 0
        negative_count = 0

        for i, (review, result) in enumerate(zip(texts_analyze1, output)):
            label = result["label"]
            score = result["score"]
            #특정 슬랭어 
            slang_positive=["미쳤다","미쳤고", "미침", "실화냐", "레전드", "개잘생김","역대급", "슬프다", "슬퍼", "미모 미쳤다.", "대상 줘라", "미쳤습니다"]
            if any(word in review for word in slang_positive):
                label = "positive"

            slang_angry=["개구림", "별로임", "별로", "어이없음", "노잼", "개별로", "돈 버림", "질 떨어짐", "정 떨어짐"]
            if any(word in review for word in slang_angry):
                label="negative"

            if label=="positive":
                label="positive"
                st.success(f"[{i}] ⭕ 긍정 ({score:.0%}) | {review[:30]}...")
                positive_count += 1
            else:
                st.error(f"[{i}] ❌ 부정 ({score:.0%}) | {review[:30]}...")
                negative_count += 1

        st.divider()
        col1, col2 = st.columns(2)
        col1.metric("⭕ 긍정", f"{positive_count}개")
        col2.metric("❌ 부정", f"{negative_count}개")

st.divider()


# 섹션 2: 감성분석 (2) 감정 분류

st.header("🎭 2. 감정 분류")
st.write("모델: `Seonghaa/korean-emotion-classifier-roberta`")

input_mode_2 = st.radio(
    "입력 방식 선택",
    ["직접 입력", "영화 리뷰 34개 분석"],
    key="input_mode_2",
    horizontal=True,
)

if input_mode_2 == "직접 입력":
    user_text2 = st.text_area("텍스트 입력", height=100, key="user_text2")
    texts_analyze2 = [user_text2] if user_text2.strip() else []
else:
    texts_analyze2 = reviews
    st.info(f"📋 영화 리뷰 {len(reviews)}개를 분석합니다.")

EMOTION_EMOJI = {
    "기쁨": "😄",
    "슬픔": "😢",
    "분노": "😠",
    "불안": "😰",
    "평온": "😌",
}

if st.button("분석하기", key="emotion_btn"):
    if not texts_analyze2:
        st.warning("텍스트를 입력해주세요.")
    else:
        with st.spinner("분석 중..."):
            clf = emotion_clf()
            results = [(clf(i, truncation=True, max_length=256)[0], i) for i in texts_analyze2]

        joy_count = 0
        sadness_count = 0
        anger_count=0
        anxiety_count=0
        tranquility_count=0

        for pred, i in results:
            label = pred["label"]
            score = pred["score"]
            emoji = EMOTION_EMOJI.get(label, "🔍")

            # 특정 슬랭어
            # 뉘앙스 구분
            slang_joy=["미쳤다", "미쳤고","미침","실화냐","레전드", "개잘생김","역대급"]
            if any(word in i for word in slang_joy):
                label="기쁨"
                joy_count +=1

            
            slang_angry=["개구림", "별로임", "별로", "어이없음", "뭐임", "노잼", "개별로", "돈 버림", "질 떨어짐"]
            if any(word in i for word in slang_angry):
                label = "분노"
                angry_count +=1

            slang_anxiety=["혼란스럽다", "홀란", "이게 뭔데", "알 수 없음"]
            if any(word in i for word in slang_anxiety):
                label="당황"
                anxiety_count +=1

            slang_tranquility=["느끼다", "느끼게 해주다", "느낀다", "느끼게"]
            if any(word in i for word in slang_anxiety):
                label="평온"
                tranquility_count += 1

            #EMOTION_EMOJI에서 label이 기쁨일 경우 
            # -> 😄 이모지 가져오게 함/ dic 없는 감정이면 🔍(기본값)
            emoji=EMOTION_EMOJI.get(label,"🔍")
            st.write(f"{emoji} **{label}** ({score:.0%}) | {i[:40]}...")
            #st.write~
            # :이모지+감정(감정 명칭)+신뢰도+리뷰 앞 40글자 화면에 출력하게 함
        
        st.divider()
        col1, col2 = st.columns(2)
        col1.metric("😄 기쁨", f"{joy_count}개")
        col2.metric("😢 슬픔", f"{sadness_count}개")
        col2.metric("😠 분노", f"{anger_count}개")
        col2.metric("😰 불안", f"{anxiety_count}개")
        col2.metric("😌 평온", f"{tranquility_count}개")
