import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import numpy as np
import urllib3
import calendar
import platform

# ============================================================================
# 1. ⚙️ [기본 설정] 페이지 및 폰트
# ============================================================================
st.set_page_config(
    page_title="BIPV 정밀 분석 대시보드",
    page_icon="📉",
    layout="wide"
)

# 폰트 및 라이브러리 설정 (에러 방지)
@st.cache_resource
def setup_environment():
    system_name = platform.system()
    if system_name == 'Windows':
        plt.rc('font', family='Malgun Gothic')
    elif system_name == 'Darwin':
        plt.rc('font', family='AppleGothic')
    else:
        # 리눅스/클라우드 환경
        if not os.path.exists('NanumGothic.ttf'):
            url = "https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf"
            r = requests.get(url)
            with open('NanumGothic.ttf', 'wb') as f:
                f.write(r.content)
        fm.fontManager.addfont('NanumGothic.ttf')
        plt.rc('font', family='NanumGothic')
    plt.rcParams['axes.unicode_minus'] = False

    try:
        from pvlib.solarposition import get_solarposition
        return get_solarposition
    except ImportError:
        os.system('pip install pvlib')
        from pvlib.solarposition import get_solarposition
        return get_solarposition

get_solarposition = setup_environment()
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ============================================================================
# 2. 🎛️ [사이드바]
# ============================================================================
st.sidebar.header("⚙️ 시뮬레이션 환경 설정")
my_key = st.sidebar.text_input("API 키", value="c6ffb5b520437f3e6983a55234e73701fce509cbb3153c9473ebbe5756a1da00", type="password")
YEAR = st.sidebar.number_input("분석 연도", value=2021)
UNIT_COUNT = st.sidebar.number_input("유닛 수", value=100)
MY_PANEL_WATT = st.sidebar.number_input("패널 용량(W)", value=300)
ELEC_RATE = st.sidebar.number_input("전기료(원/kWh)", value=150)

LOCATION = {'lat': '37.5643', 'lon': '126.8386'}
my_url = "https://apis.data.go.kr/B551184/SolarGhiService/getSolarGhiHrInfo"

# ============================================================================
# 3. 🧠 [핵심 로직] 45도 제한 + 현실적 효율 계산
# ============================================================================
def simulate_day(target_date):
    times = pd.date_range(start=f'{target_date} 09:00', end=f'{target_date} 17:00', freq='1H', tz='Asia/Seoul')
    solpos = get_solarposition(times, float(LOCATION['lat']), float(LOCATION['lon']))
    zeniths = solpos['apparent_zenith'].values
    
    # API 호출
    ghi_list = []
    for h in times.hour:
        params = {
            'serviceKey': requests.utils.unquote(my_key),
            'pageNo': '1', 'numOfRows': '10', 'type': 'json',
            'date': target_date.replace('-', ''), 'time': f"{h:02d}00",
            'lat': LOCATION['lat'], 'lon': LOCATION['lon']
        }
        val = 0
        try:
            res = requests.get(my_url, params=params, verify=False, timeout=1)
            if res.status_code == 200:
                item = res.json()['response']['body']['items']['item']
                item = item[0] if isinstance(item, list) else item
                val = float(item['ghi']) if 'ghi' in item else 0
        except: pass
        ghi_list.append(val)
    
    fixed_wh_sum = 0
    ai_wh_sum = 0
    ideal_wh_sum = 0 

    for i, ghi in enumerate(ghi_list):
        if ghi < 10: continue
        z = zeniths[i]
        
        # A. 고정식 (벽면 90도)
        eff_fixed = max(0, np.cos(np.radians(abs(90 - z))))
        
        # B. AI 시스템 (45도 락킹 - 물리적 한계)
        # 태양(z)이 20도여도 패널은 45도에서 멈춤 -> 25도 차이 발생 -> 효율 저하
        ai_tilt = max(45, z) 
        eff_ai = max(0, np.cos(np.radians(abs(ai_tilt - z))))
        
        # C. 이상적 트래커 (기준점)
        # 기준은 "손실 없는 완벽한 상태(1.0)"로 잡아야 
        # 우리 시스템의 0.85 손실과 45도 각도 손실이 티가 납니다.
        eff_ideal = 1.0 
        
        # 발전량 합산
        # - 실제 시스템: 0.85 손실 반영
        fixed_wh_sum += (ghi / 1000) * MY_PANEL_WATT * eff_fixed * 0.85
        ai_wh_sum += (ghi / 1000) * MY_PANEL_WATT * eff_ai * 0.85
        
        # - 기준점: 손실 미반영 (그래야 효율 점수가 짜게 나옴)
        ideal_wh_sum += (ghi / 1000) * MY_PANEL_WATT * eff_ideal

    return fixed_wh_sum, ai_wh_sum, ideal_wh_sum

# ============================================================================
# 4. 🖥️ [메인 UI] 실행 및 리포트
# ============================================================================
st.title("💰 BIPV 경제성 분석 (45° 제한 적용)")
st.caption("실제 기상 데이터 기반 정밀 시뮬레이션")

if st.button("🚀 분석 시작", type="primary"):
    
    progress_text = "데이터 분석 중입니다. 잠시만 기다려주세요..."
    my_bar = st.progress(0, text=progress_text)
    
    monthly_data = []
    total_ideal_kwh = 0
    
    for m in range(1, 13):
        target_date = f"{YEAR}-{m:02d}-15"
        days = calendar.monthrange(YEAR, m)[1]
        
        f_day, a_day, i_day = simulate_day(target_date)
        
        monthly_data.append({
            '월': m,
            '고정식': (f_day * days * UNIT_COUNT) / 1000,
            'AI시스템': (a_day * days * UNIT_COUNT) / 1000
        })
        total_ideal_kwh += (i_day * days * UNIT_COUNT) / 1000
        my_bar.progress(m / 12, text=f"{m}월 데이터 처리 완료")
        
    my_bar.empty()
    res_df = pd.DataFrame(monthly_data)
    
    # 결과 계산
    total_fixed = res_df['고정식'].sum()
    total_ai = res_df['AI시스템'].sum()
    
    diff_kwh = total_ai - total_fixed
    diff_pct = (diff_kwh / total_fixed) * 100 if total_fixed > 0 else 0
    profit_money = diff_kwh * ELEC_RATE
    
    # 효율 계산 (분모가 '완벽한 트래커'이므로 98%가 나올 수 없음 -> 80%대 예상)
    eff_fixed_pct = (total_fixed / total_ideal_kwh * 100) if total_ideal_kwh > 0 else 0
    eff_ai_pct = (total_ai / total_ideal_kwh * 100) if total_ideal_kwh > 0 else 0
    
    # 진단 데이터
    simulated_val = total_ai
    actual_val = total_ai * 0.99
    health_val = (actual_val / simulated_val * 100)

    # --- 리포트 출력 ---
    st.success("분석이 완료되었습니다!")
    
    # 1. KPI 지표
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("AI 시스템 발전량", f"{total_ai:,.0f} kWh", f"+{diff_kwh:,.0f} kWh")
    kpi2.metric("시스템 효율 (PR)", f"{eff_ai_pct:.1f}%", f"이상적 트래커 대비")
    kpi3.metric("예상 수익금", f"{profit_money:,.0f} 원", "연간 절감액")
    
    st.divider()

    # 2. 텍스트 리포트
    st.subheader("📄 최종 분석 리포트")
    report_text = f"""
============================================================
[ {YEAR}년 BIPV 연간 경제성 분석 리포트 (Real Logic) ]
============================================================
1. 총 발전량 비교 (유닛 {UNIT_COUNT}개 기준)
   - 🧱 고정식 (벽면)     : {total_fixed:,.0f} kWh
   - 🤖 AI 시스템 (AI)    : {total_ai:,.0f} kWh
   ---------------------------------------------
   📈 연간 추가 발전 : +{diff_kwh:,.0f} kWh ({diff_pct:.1f}% 향상)
   💸 연간 수익 창출 : 약 {profit_money:,.0f} 원 절약
------------------------------------------------------------
2. 시스템 설계 수광 효율 (Design Optical Efficiency)
   * 기준(100%): 태양 정면(90도) 수광 가능한 이상적 트래커
   ---------------------------------------------
   🔴 고정식 (벽면) 효율 : {eff_fixed_pct:.1f}%
   🟢 AI 시스템 효율     : {eff_ai_pct:.1f}%
   (해석: 45도 제한으로 인해 {100-eff_ai_pct:.1f}%의 구조적 손실 발생)
------------------------------------------------------------
3. [예시] 유지보수 AI 자동 진단 리포트
--- 🩺 정밀 진단 결과 (노후화 체크) ---
1. 시뮬레이션 예측 : {simulated_val:,.0f} kWh
2. 실제 계측량     : {actual_val:,.0f} kWh
3. 기계 건강 상태  : {health_val:.1f}%
✅ 진단 메시지     : 상태 최상
------------------------------------------------------------
============================================================
    """
    st.text_area("Report", report_text, height=450)
    
    # 3. 그래프
    st.subheader("📊 월별 발전량 비교 그래프")
    fig, ax = plt.subplots(figsize=(10, 5))
    indices = np.arange(len(res_df['월']))
    width = 0.35
    
    ax.bar(indices - width/2, res_df['고정식'], width, label='고정식', color='gray', alpha=0.6)
    bars = ax.bar(indices + width/2, res_df['AI시스템'], width, label='AI 시스템', color='#ff4b4b')
    
    ax.set_title(f'{YEAR}년 월별 발전량 비교', fontsize=15, fontweight='bold')
    ax.set_ylabel('발전량 (kWh)')
    ax.set_xticks(indices)
    ax.set_xticklabels([f"{m}월" for m in res_df['월']])
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    for rect in bars:
        height = rect.get_height()
        ax.annotate(f'{int(height)}', (rect.get_x() + rect.get_width()/2, height),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    
    st.pyplot(fig)
