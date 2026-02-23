# ==============================================================================
# BIPV 통합 관제 시스템 (Streamlit Cloud 최적화 버전)
# ==============================================================================
__version__ = "5.30"

import os
import sys
import pandas as pd
import numpy as np
import requests
import urllib.parse
import streamlit as st
import plotly.graph_objects as go
import pvlib
from pvlib.location import Location
from datetime import datetime, timedelta
import pytz
import joblib
import xgboost as xgb
import gdown

# ==============================================================================
# 1. 통합 환경 설정
# ==============================================================================
# ★ 여기에 구글 드라이브 파일 ID를 입력하세요 ★
GDRIVE_FILE_ID = "여기에_파일_ID_입력"

KMA_SERVICE_KEY = "c6ffb5b520437f3e6983a55234e73701fce509cbb3153c9473ebbe5756a1da00"
LAT, LON, TZ = 37.5665, 126.9780, "Asia/Seoul"
NX, NY = 60, 127
KST = pytz.timezone(TZ)

DEFAULT_CAPACITY = 300
DEFAULT_EFFICIENCY = 18.7
DEFAULT_LOSS = 0.85
DEFAULT_KEPCO = 210
DEFAULT_UNIT_COUNT = 1

DEFAULT_LOUVER_COUNT = 10
DEFAULT_WIDTH_MM = 1000.0   
DEFAULT_HEIGHT_MM = 160.0   
ANGLE_CAP_DEG_DEFAULT = 90.0
XGB_FEATURE_NAMES = ["hour", "month", "zenith", "azimuth", "ghi", "dni", "dhi", "cloud_cover"]
XGB_MODEL_FILENAME = "bipv_xgboost_model.pkl"

# ==============================================================================
# 2. 핵심 로직: 모델 로드 및 기상청 API
# ==============================================================================
@st.cache_resource(show_spinner="AI 모델을 설정 중입니다...")
def load_xgb_model(file_id):
    """구글 드라이브에서 모델 다운로드 및 메모리 적재 (캐싱)"""
    if not file_id or file_id == "여기에_파일_ID_입력":
        return None
    
    # 파일이 없으면 구글 드라이브에서 다운로드
    if not os.path.isfile(XGB_MODEL_FILENAME):
        try:
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, XGB_MODEL_FILENAME, quiet=False)
        except Exception as e:
            st.error(f"구글 드라이브 다운로드 에러: {e}")
            return None

    # 다운로드된 파일 로드
    if os.path.isfile(XGB_MODEL_FILENAME):
        try:
            return joblib.load(XGB_MODEL_FILENAME)
        except Exception as e:
            st.error(f"모델 파일 손상: {e}")
            return None
    return None

@st.cache_data(ttl=3600, show_spinner="기상청 데이터를 불러오는 중입니다...")
def get_kma_forecast():
    """기상청 단기예보 호출 (1시간 동안 데이터 캐싱 유지)"""
    decoded_key = urllib.parse.unquote(KMA_SERVICE_KEY)
    now_kst = datetime.now(KST)
    base_date = now_kst.strftime("%Y%m%d")
    now_hour = now_kst.hour
    
    available_hours = [2, 5, 8, 11, 14, 17, 20, 23]
    base_time_int = max([h for h in available_hours if h <= now_hour] or [23])
    if base_time_int == 23 and now_hour < 2:
        base_date = (now_kst - timedelta(days=1)).strftime("%Y%m%d")
    base_time = f"{base_time_int:02d}00"

    url = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"
    params = {"serviceKey": decoded_key, "numOfRows": "1000", "dataType": "JSON",
              "base_date": base_date, "base_time": base_time, "nx": NX, "ny": NY}

    try:
        res = requests.get(url, params=params, timeout=10).json()
        items = res["response"]["body"]["items"]["item"]
        df = pd.DataFrame(items)
        tomorrow = (now_kst + timedelta(days=1)).strftime("%Y%m%d")
        df_tom = df[df["fcstDate"] == tomorrow]
        df_tom = df_tom.drop_duplicates(subset=['fcstDate', 'fcstTime', 'category'])
        return df_tom.pivot(index="fcstTime", columns="category", values="fcstValue"), tomorrow
    except Exception:
        tomorrow = (now_kst + timedelta(days=1)).strftime("%Y%m%d")
        return None, tomorrow

def predict_angles_xgb(model, times, zenith_arr, azimuth_arr, ghi_real, dni_arr, dhi_arr, cloud_series, angle_cap_deg):
    month = times.month.values
    hour = times.hour.values
    X = pd.DataFrame({
        "hour": hour, "month": month, "zenith": zenith_arr, "azimuth": azimuth_arr,
        "ghi": ghi_real, "dni": dni_arr, "dhi": dhi_arr, "cloud_cover": cloud_series,
    }, columns=XGB_FEATURE_NAMES)
    
    if hasattr(model, "feature_names_in_"):
        cols = [c for c in model.feature_names_in_ if c in X.columns]
        if cols: X = X[cols]
    try:
        pred = model.predict(X)
    except Exception:
        return None
    pred = np.asarray(pred).ravel()
    pred = np.clip(pred, 0, min(90, angle_cap_deg))
    pred[ghi_real < 10] = 0
    return pred.astype(float)

def _poa_with_iam_app(surface_tilt, surface_azimuth, solar_zenith, solar_azimuth, dni, ghi, dhi, a_r=0.16):
    irrad = pvlib.irradiance.get_total_irradiance(
        surface_tilt=surface_tilt, surface_azimuth=surface_azimuth,
        dni=dni, ghi=ghi, dhi=dhi, solar_zenith=solar_zenith, solar_azimuth=solar_azimuth
    )
    poa_direct = np.nan_to_num(irrad["poa_direct"], nan=0.0)
    poa_diffuse = np.nan_to_num(irrad["poa_diffuse"], nan=0.0)
    aoi = pvlib.irradiance.aoi(surface_tilt, surface_azimuth, solar_zenith, solar_azimuth)
    aoi = np.clip(np.asarray(aoi, dtype=float), 0, 90)
    iam = pvlib.iam.martin_ruiz(aoi, a_r=a_r)
    return poa_direct * iam + poa_diffuse

# ==============================================================================
# 3. Streamlit 앱 렌더링
# ==============================================================================
st.set_page_config(page_title="BIPV Dashboard", layout="wide")

# 기상청 데이터 로드
kma, tomorrow_str = get_kma_forecast()
if kma is None:
    st.warning("기상청 데이터를 불러오지 못해 '클리어스카이(맑음)' 기준으로 시뮬레이션합니다.")

# 모델 로드
xgb_model = load_xgb_model(GDRIVE_FILE_ID)

# 사이드바 UI
st.sidebar.title("■ 통합 환경 설정")

if st.sidebar.button("🔄 AI 모델 재접속 (오류 시 클릭)"):
    st.cache_resource.clear()
    st.rerun()

st.sidebar.subheader("1. 시간 및 날짜")
sim_date = st.sidebar.date_input("시뮬레이션 날짜", datetime.strptime(tomorrow_str, "%Y%m%d"))
sunshine_hours = 10

st.sidebar.subheader("2. 설치 면적 (루버 1개 가로×세로, 개수)")
width_mm = st.sidebar.number_input("루버 1개 가로 (mm)", min_value=100.0, value=float(DEFAULT_WIDTH_MM), step=100.0)
height_mm = st.sidebar.number_input("루버 1개 세로 (mm)", min_value=100.0, value=float(DEFAULT_HEIGHT_MM), step=100.0)
louver_count = st.sidebar.number_input("루버 개수 (개)", min_value=1, value=DEFAULT_LOUVER_COUNT, step=1)

ref_area = DEFAULT_WIDTH_MM * DEFAULT_HEIGHT_MM * DEFAULT_LOUVER_COUNT if DEFAULT_LOUVER_COUNT > 0 else 1.0
user_area = width_mm * height_mm * louver_count if louver_count > 0 else ref_area
area_scale = user_area / ref_area if ref_area > 0 else 1.0

st.sidebar.subheader("3. 패널 스펙")
unit_count = st.sidebar.number_input("설치 유닛 수 (개)", min_value=1, value=DEFAULT_UNIT_COUNT)
capacity_w = st.sidebar.number_input("기준 패널 용량 (W)", value=DEFAULT_CAPACITY)
target_eff = st.sidebar.number_input("패널 효율 (%)", value=DEFAULT_EFFICIENCY, step=0.1)
kepco_rate = st.sidebar.number_input("전기 요금 (원/kWh)", value=DEFAULT_KEPCO)

# 연산 변수 준비
_sim_d = sim_date.strftime("%Y-%m-%d")
site = Location(LAT, LON, tz=TZ)
times = pd.date_range(start=f"{_sim_d} 00:00", periods=24, freq="h", tz=TZ)

solpos = site.get_solarposition(times)
clearsky = site.get_clearsky(times)
zenith_arr = np.asarray(solpos["apparent_zenith"].values, dtype=float)
azimuth_arr = np.asarray(solpos["azimuth"].values, dtype=float)

# 기상 적용
if _sim_d.replace("-", "") == tomorrow_str and kma is not None:
    kma_reindex = kma.reindex(times.strftime("%H00"))
    cloud_series = kma_reindex["SKY"].apply(lambda x: 0.0 if x == "1" else (0.5 if x == "3" else 1.0)).fillna(0.0).astype(float).values
else:
    cloud_series = np.zeros(24)

ghi_real = np.asarray(clearsky["ghi"].values, dtype=float) * (1.0 - (cloud_series * 0.65))
_dni = pvlib.irradiance.dirint(ghi_real, solpos["apparent_zenith"], times).fillna(0)
dni_arr = np.asarray(_dni.values, dtype=float).ravel()
dhi_arr = (ghi_real - dni_arr * np.cos(np.radians(zenith_arr))).clip(0).astype(float).ravel()

# 운영 시간 설정
solar_noon_idx = int(np.argmin(zenith_arr))
half = int(sunshine_hours) // 2
op_start = max(0, min(solar_noon_idx - half, 7))
op_end = min(23, solar_noon_idx + half + (int(sunshine_hours) % 2))
op_hours = (op_start, op_end)

# 각도 예측 (AI or Rule-based)
angle_mode = "규칙 기반 (수학 연산)"
current_ai_angles = None

if xgb_model is not None:
    xgb_angles = predict_angles_xgb(xgb_model, times, zenith_arr, azimuth_arr, ghi_real, dni_arr, dhi_arr, cloud_series, ANGLE_CAP_DEG_DEFAULT)
    if xgb_angles is not None:
        current_ai_angles = xgb_angles
        angle_mode = "XGBoost (AI 추론)"

if current_ai_angles is None:
    current_ai_angles = np.where(ghi_real < 10, 0, np.clip(90 - zenith_arr, 0, ANGLE_CAP_DEG_DEFAULT).astype(float))

def calc_power(angles_list):
    tilt = 90 - np.array(angles_list, dtype=float)
    poa_eff = _poa_with_iam_app(tilt, np.full_like(tilt, 180), zenith_arr, azimuth_arr, dni_arr, ghi_real, dhi_arr)
    mask = (times.hour >= op_hours[0]) & (times.hour <= op_hours[1])
    eff_factor = float(target_eff) / DEFAULT_EFFICIENCY
    base_wh = (poa_eff[mask] / 1000 * capacity_w * unit_count * eff_factor * DEFAULT_LOSS).sum()
    return base_wh * area_scale

pow_ai = calc_power(current_ai_angles)
pow_fix_0 = calc_power([0] * 24)
rev_ai = (pow_ai / 1000) * kepco_rate

weather_status = "맑음" if np.mean(cloud_series) < 0.3 else ("구름많음" if np.mean(cloud_series) < 0.8 else "흐림")

# 화면 출력
st.title("■ BIPV 통합 관제 대시보드 v5.30")
st.markdown(f"**날짜:** {_sim_d} | **날씨:** {weather_status} | **제어 엔진:** **{angle_mode}**")

c1, c2 = st.columns(2)
c1.metric("AI 제어", f"{int(rev_ai):,} 원  {pow_ai/1000:.2f} kWh", "당일")
c2.metric("고정 0°", f"{int((pow_fix_0/1000)*kepco_rate):,} 원  {pow_fix_0/1000:.2f} kWh", "당일")

st.markdown("---")
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("■ 제어 스케줄")
    mask_plot = (times.hour >= 6) & (times.hour <= 19)
    x_plot = times[mask_plot].strftime("%H:%M")
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x_plot, y=ghi_real[mask_plot], name="일사량 (W/m²)", marker_color="orange", opacity=0.3, yaxis="y1"))
    fig.add_trace(go.Scatter(x=x_plot, y=current_ai_angles[mask_plot], name="제어 각도", line=dict(color="blue", width=4), yaxis="y2"))
    fig.update_layout(
        yaxis=dict(title="일사량 (W/m²)", showgrid=False),
        yaxis2=dict(title="각도 (°)", overlaying="y", side="right", range=[0, 95], showgrid=True),
        legend=dict(orientation="h", y=1.1)
    )
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("■ 발전량 비교")
    comp_data = [{"Mode": "AI 제어", "Val": pow_ai, "Color": "#1a73e8"}, {"Mode": "고정(0°)", "Val": pow_fix_0, "Color": "gray"}]
    df_comp = pd.DataFrame(comp_data)
    fig_bar = go.Figure(data=[go.Bar(x=df_comp["Mode"], y=df_comp["Val"], marker_color=[d["Color"] for d in comp_data], text=[f"{v:.0f}Wh" for v in df_comp["Val"]], textposition="auto")])
    fig_bar.update_layout(height=300, margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(fig_bar, use_container_width=True)
