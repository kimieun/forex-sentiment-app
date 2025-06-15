
import streamlit as st
import pandas as pd
from prophet import Prophet
from datetime import datetime, timedelta
import numpy as np
import requests

st.set_page_config(page_title="환율 + 뉴스 감성 예측 AI", layout="wide")
st.title("💱 환율 + 감성 기반 예측 AI")

# ===== STEP 1: 한국은행 API에서 환율 데이터 불러오기 =====
@st.cache_data
def fetch_exchange_rate():
    API_KEY = "99BO6UEVOS1ZHTSHK79J"
    start_date = "20250601"
    end_date = datetime.today().strftime("%Y%m%d")
    url = f"http://ecos.bok.or.kr/api/StatisticSearch/{API_KEY}/json/kr/1/1000/036Y001/DD/{start_date}/{end_date}/0002"

    try:
        response = requests.get(url)

        # 디버깅 출력
        st.write("📡 응답 상태 코드:", response.status_code)
        try:
            st.json(response.json())
        except Exception as e:
            st.error(f"JSON 파싱 실패: {e}")

        data = response.json()
        rows = data['StatisticSearch']['row']
        df = pd.DataFrame(rows)
        df = df[['TIME', 'DATA_VALUE']]
        df.columns = ['ds', 'y']
        df['ds'] = pd.to_datetime(df['ds'])
        df['y'] = df['y'].astype(float)
        return df
    except Exception as e:
        st.error(f"환율 API 로딩 실패: {e}")
        return None

# ===== STEP 2: 감성 점수 데이터 불러오기 =====
@st.cache_data
def load_sentiment():
    try:
        s_df = pd.read_csv("data/sentiment.csv")
        s_df['ds'] = pd.to_datetime(s_df['ds'])
        return s_df
    except:
        return None

# ===== STEP 3: 데이터 병합 =====
rate_df = fetch_exchange_rate()
sentiment_df = load_sentiment()

if rate_df is None or sentiment_df is None:
    st.warning("환율 또는 감성 점수 데이터를 불러올 수 없습니다.")
    st.stop()

merged_df = pd.merge(rate_df, sentiment_df, on="ds", how="inner")

# ===== 사용자 입력 =====
st.sidebar.header("예측 설정")
days = st.sidebar.slider("예측할 일 수", min_value=3, max_value=30, value=7)

# ===== Prophet 예측 =====
try:
    model = Prophet()
    model.add_regressor('sentiment_score')
    model.fit(merged_df)

    future = model.make_future_dataframe(periods=days)
    future = future.merge(sentiment_df, on="ds", how="left")
    future['sentiment_score'].fillna(method='ffill', inplace=True)

    forecast = model.predict(future)
    result = forecast[['ds', 'yhat']].tail(days)
    result.columns = ['날짜', '예측 환율 (KRW/USD)']

    st.subheader("📈 예측 결과")
    st.line_chart(result.set_index("날짜"))
    st.dataframe(result)

except Exception as e:
    st.error(f"모델 예측 실패: {e}")
