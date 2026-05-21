"""
BID-AI Streamlit 대시보드
- 메인 대시보드: 통계 요약 + 공고 목록
- 낙찰률 예측: 공고 정보 입력 → 투찰율 예측
- 제안서 생성: 공고 선택 → AI 프롬프트 생성
- 분석 리포트: Feature Importance, 분포 차트
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pickle
import os

matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# 데이터 & 모델 로드
# ============================================================
@st.cache_data
def load_data():
    scored_path = os.path.join(BASE_DIR, '05output', 'scored_all.csv')
    if os.path.exists(scored_path):
        df = pd.read_csv(scored_path, encoding='utf-8-sig')
    else:
        # Google Drive 직접 다운로드 링크 사용
        url = "https://drive.google.com/uc?id=1Mfqh0eautBticb3iUp0d2W8hMWEMmncb"
        df = pd.read_csv(url, encoding='utf-8-sig')
    df['공고게시일자'] = pd.to_datetime(df['공고게시일자'], errors='coerce')
    return df

@st.cache_data
def load_experiment_results():
    # 06output 우선 (논문급 다중 모델 비교)
    path06 = os.path.join(BASE_DIR, '06output', 'model_comparison.csv')
    if os.path.exists(path06):
        return pd.read_csv(path06, encoding='utf-8-sig')
    path = os.path.join(BASE_DIR, '04output', 'all_experiment_results.csv')
    if os.path.exists(path):
        return pd.read_csv(path, encoding='utf-8-sig')
    return None

@st.cache_data
def load_feature_importance():
    # 06output 우선 (최종 모델 FI)
    path06 = os.path.join(BASE_DIR, '06output', 'feature_importance_final.csv')
    if os.path.exists(path06):
        return pd.read_csv(path06, encoding='utf-8-sig')
    path = os.path.join(BASE_DIR, '04output', 'feature_importance_v2.csv')
    if os.path.exists(path):
        return pd.read_csv(path, encoding='utf-8-sig')
    return None

df = load_data()
exp_results = load_experiment_results()
feat_imp = load_feature_importance()

# SW/IT 필터
sw_keywords = ['소프트웨어', '정보시스템', '전산', '데이터', '클라우드', '보안', '네트워크']
sw_mask = df['공공조달분류'].fillna('').str.contains('|'.join(sw_keywords))

# ============================================================
# 사이드바
# ============================================================
st.sidebar.title("BID-AI")
st.sidebar.markdown("공공입찰 낙찰률 예측 &\nAI 제안서 생성 시스템")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "메뉴",
    ["Dashboard", "Prediction", "Proposal", "Analysis"],
    index=0
)

st.sidebar.markdown("---")
sw_only = st.sidebar.checkbox("SW/IT 공고만 보기", value=False)
if sw_only:
    df_view = df[sw_mask].copy()
else:
    df_view = df.copy()

st.sidebar.metric("전체 공고", f"{len(df):,}건")
st.sidebar.metric("SW/IT 공고", f"{sw_mask.sum():,}건")


# ============================================================
# PAGE 1: Dashboard
# ============================================================
if page == "Dashboard":
    st.title("Dashboard")
    st.markdown("공공입찰 데이터 현황 및 적합도 요약")

    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("총 공고 수", f"{len(df_view):,}건")
    with col2:
        st.metric("평균 투찰율", f"{df_view['투찰율'].mean():.1f}%")
    with col3:
        st.metric("평균 배정예산", f"{df_view['배정예산금액'].mean()/1e8:.1f}억")
    with col4:
        if 'fit_score' in df_view.columns:
            st.metric("평균 적합도", f"{df_view['fit_score'].mean():.1f}점")
        else:
            st.metric("평균 참여자수", f"{df_view['입찰참여자수'].mean():.1f}")

    st.markdown("---")

    # 차트 Row 1
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("투찰율 분포")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(df_view['투찰율'].dropna(), bins=40, color='#38BDF8', edgecolor='white', alpha=0.8)
        ax.axvline(df_view['투찰율'].mean(), color='red', linestyle='--', label=f"평균: {df_view['투찰율'].mean():.1f}%")
        ax.set_xlabel('투찰율 (%)')
        ax.set_ylabel('건수')
        ax.legend()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("공공조달분류 TOP 10")
        top10 = df_view['공공조달분류'].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(range(len(top10)), top10.values, color='#818CF8', alpha=0.8)
        ax.set_yticks(range(len(top10)))
        ax.set_yticklabels(top10.index, fontsize=9)
        ax.set_xlabel('건수')
        ax.invert_yaxis()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # 적합도 기반 추천 (스코어링 결과 있을 때)
    if 'fit_score' in df_view.columns:
        st.subheader("적합도 TOP 공고")
        top_bids = df_view.nlargest(20, 'fit_score')[
            ['입찰공고명', '공공조달분류', '배정예산금액', '투찰율', '입찰참여자수', 'fit_score', 'grade']
        ].copy()
        top_bids['배정예산금액'] = top_bids['배정예산금액'].apply(
            lambda x: f"{x/1e8:.1f}억" if x >= 1e8 else f"{x/1e4:.0f}만"
        )
        top_bids.columns = ['공고명', '분류', '예산', '투찰율(%)', '참여자수', '적합도', '등급']
        st.dataframe(top_bids, use_container_width=True, hide_index=True)
    else:
        st.subheader("최근 공고 목록")
        recent = df_view.nlargest(20, '공고게시일자')[
            ['입찰공고명', '공공조달분류', '배정예산금액', '투찰율', '입찰참여자수']
        ].copy()
        recent['배정예산금액'] = recent['배정예산금액'].apply(
            lambda x: f"{x/1e8:.1f}억" if x >= 1e8 else f"{x/1e4:.0f}만"
        )
        st.dataframe(recent, use_container_width=True, hide_index=True)


# ============================================================
# PAGE 2: Prediction
# ============================================================
elif page == "Prediction":
    st.title("낙찰률 예측")
    st.markdown("공고 정보를 입력하면 예상 투찰율과 적합도를 산출합니다.")

    col1, col2 = st.columns(2)

    with col1:
        title = st.text_input("공고명", "정보시스템 유지관리 용역")
        budget = st.number_input("배정예산금액 (원)", value=200000000, step=10000000, format="%d")
        category = st.selectbox("공공조달분류",
                                df['공공조달분류'].value_counts().head(30).index.tolist())

    with col2:
        participants = st.number_input("예상 참여자수", value=5, min_value=1, max_value=500)
        has_industry = st.checkbox("SW 업종 제한 있음", value=True)
        has_region = st.checkbox("지역 제한 있음", value=False)

    if st.button("예측하기", type="primary"):
        # 간이 예측 (통계 기반)
        # 해당 분류의 평균 투찰율
        cat_data = df[df['공공조달분류'] == category]
        if len(cat_data) > 0:
            base_rate = cat_data['투찰율'].mean()
        else:
            base_rate = df['투찰율'].mean()

        # 참여자수 보정
        part_adj = 0
        if participants <= 2:
            part_adj = 3.0
        elif participants <= 5:
            part_adj = 1.0
        elif participants <= 10:
            part_adj = -0.5
        else:
            part_adj = -2.0

        # 예산 규모 보정
        budget_adj = 0
        if budget >= 1e9:
            budget_adj = 1.5
        elif budget >= 5e8:
            budget_adj = 0.5

        predicted_rate = min(100, max(60, base_rate + part_adj + budget_adj))

        # 적합도 점수
        row = {
            '투찰율': predicted_rate,
            '배정예산금액': budget,
            '면허업종제한목록': '(1468)' if has_industry else '',
            '입찰참여자수': participants,
        }

        # 간이 스코어링 (업종 매칭은 사전 필터 → 점수 항목에서 제외)
        rate_score = min(100, max(0, (predicted_rate - 60) / 40 * 100))
        budget_score = 100 if 3e7 <= budget <= 5e9 else 50
        if participants <= 2: comp_score = 100
        elif participants <= 5: comp_score = 80
        elif participants <= 10: comp_score = 60
        else: comp_score = 30

        fit_score = rate_score * 0.487 + budget_score * 0.268 + comp_score * 0.246

        if fit_score >= 80: grade = "A (강력 추천)"
        elif fit_score >= 65: grade = "B (추천)"
        elif fit_score >= 50: grade = "C (검토)"
        else: grade = "D (비추천)"

        st.markdown("---")
        st.subheader("예측 결과")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("예상 투찰율", f"{predicted_rate:.1f}%")
        with col2:
            st.metric("적합도 점수", f"{fit_score:.1f}")
        with col3:
            st.metric("등급", grade)

        # 세부 점수
        st.markdown("**세부 점수** (업종 매칭은 사전 필터로 적용됨)")
        score_df = pd.DataFrame({
            '항목': ['낙찰률 점수', '예산 적합도', '경쟁 강도'],
            '점수': [rate_score, budget_score, comp_score],
            '가중치': ['48.7%', '26.8%', '24.6%'],
        })
        st.dataframe(score_df, use_container_width=True, hide_index=True)

        # 유사 공고
        st.markdown("**유사 공고 참고**")
        similar = cat_data.nlargest(5, '투찰율')[
            ['입찰공고명', '배정예산금액', '투찰율', '입찰참여자수']
        ].copy()
        similar['배정예산금액'] = similar['배정예산금액'].apply(
            lambda x: f"{x/1e8:.1f}억" if x >= 1e8 else f"{x/1e4:.0f}만"
        )
        st.dataframe(similar, use_container_width=True, hide_index=True)


# ============================================================
# PAGE 3: Proposal
# ============================================================
elif page == "Proposal":
    st.title("AI 제안서 생성")
    st.markdown("공고 정보를 입력하면 Claude API용 프롬프트를 생성합니다.")

    col1, col2 = st.columns(2)
    with col1:
        title = st.text_input("공고명", "한국보육진흥원 e러닝 시스템 클라우드 전환사업")
        org = st.text_input("수요기관", "한국보육진흥원")
        budget = st.number_input("배정예산 (원)", value=326000000, step=10000000, format="%d")
    with col2:
        category = st.text_input("공공조달분류", "정보시스템개발서비스")
        industry = st.text_input("업종 요건", "소프트웨어사업자(컴퓨터관련서비스사업)")
        requirements = st.text_area("핵심 요구사항 (선택)",
                                     "클라우드 전환, 기존 시스템 마이그레이션, 보안 강화")

    if st.button("프롬프트 생성", type="primary"):
        budget_str = f"{budget/1e8:.1f}억원" if budget >= 1e8 else f"{budget/1e4:.0f}만원"

        prompt = f"""당신은 공공입찰 기술제안서 전문 작성자입니다.
아래 공고 정보를 바탕으로 기술제안서 초안을 작성해주세요.

## 공고 정보
- 공고명: {title}
- 수요기관: {org}
- 배정예산: {budget_str}
- 분류: {category}
- 업종 요건: {industry}
- 핵심 요구사항: {requirements}

## 작성 요청 사항
다음 구조로 제안서를 작성해주세요:

### 1. 사업 이해
- 본 사업의 배경과 목적을 분석하고, 수요기관의 핵심 니즈를 정리

### 2. 추진 전략
- 사업 수행을 위한 핵심 전략 3~4가지
- 각 전략별 구체적 실행 방안

### 3. 기술 제안
- 적용 기술 및 아키텍처
- 핵심 기능 설명
- 기존 시스템과의 연계 방안

### 4. 수행 계획
- 단계별 추진 일정 (WBS)
- 투입 인력 구성
- 품질 관리 방안

### 5. 차별화 전략
- 경쟁사 대비 당사의 강점
- 유사 사업 수행 실적 (가상)
- 리스크 관리 방안

전문적이고 구체적으로 작성하되, 배정예산 규모에 맞는 현실적인 제안을 해주세요."""

        st.markdown("---")
        st.subheader("생성된 프롬프트")
        st.code(prompt, language="markdown")

        st.info("위 프롬프트를 Claude 웹(claude.ai)에 붙여넣기하면 제안서 초안이 생성됩니다.")

        # 유사 공고 참고
        st.markdown("---")
        st.subheader("유사 공고 참고 데이터")
        cat_data = df[df['공공조달분류'].str.contains(category[:4], na=False)]
        if len(cat_data) > 0:
            similar = cat_data.nlargest(5, '투찰율')[
                ['입찰공고명', '대표업체', '배정예산금액', '투찰율', '입찰참여자수']
            ].copy()
            similar['배정예산금액'] = similar['배정예산금액'].apply(
                lambda x: f"{x/1e8:.1f}억" if x >= 1e8 else f"{x/1e4:.0f}만"
            )
            st.dataframe(similar, use_container_width=True, hide_index=True)
        else:
            st.write("유사 공고 데이터가 없습니다.")


# ============================================================
# PAGE 4: Analysis
# ============================================================
elif page == "Analysis":
    st.title("분석 리포트")
    st.markdown("모델 성능 및 데이터 분석 결과")

    # 실험 결과
    if exp_results is not None:
        st.subheader("모델 실험 결과")
        st.dataframe(exp_results, use_container_width=True, hide_index=True)

        # MAE 비교 차트
        fig, ax = plt.subplots(figsize=(10, 4))
        models = exp_results.iloc[:, 0].tolist()
        maes = exp_results['MAE'].tolist()
        colors = ['#94A3B8'] * (len(models) - 1) + ['#34D399']
        ax.barh(range(len(models)), maes, color=colors)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models, fontsize=9)
        ax.set_xlabel('MAE')
        ax.set_title('모델별 MAE 비교', fontweight='bold')
        ax.invert_yaxis()
        for i, v in enumerate(maes):
            ax.text(v + 0.02, i, f'{v:.3f}', va='center', fontweight='bold')
        st.pyplot(fig)
        plt.close()

    # Feature Importance
    if feat_imp is not None:
        st.subheader("Feature Importance TOP 20")
        top20 = feat_imp.head(20)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(top20)), top20['importance'].values, color='#38BDF8', alpha=0.8)
        ax.set_yticks(range(len(top20)))
        ax.set_yticklabels(top20['feature'].values, fontsize=9)
        ax.set_xlabel('Importance')
        ax.set_title('Feature Importance TOP 20 (LightGBM)', fontweight='bold')
        ax.invert_yaxis()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # 투찰율 vs 주요 변수
    st.subheader("투찰율과 주요 변수 관계")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(7, 4))
        bins_part = pd.cut(df_view['입찰참여자수'], bins=[0, 1, 3, 5, 10, 50, 10000],
                           labels=['1', '2~3', '4~5', '6~10', '11~50', '50+'])
        grouped = df_view.groupby(bins_part, observed=True)['투찰율'].mean()
        ax.bar(range(len(grouped)), grouped.values, color='#34D399', alpha=0.8)
        ax.set_xticks(range(len(grouped)))
        ax.set_xticklabels(grouped.index)
        ax.set_title('참여자수 구간별 평균 투찰율', fontweight='bold')
        ax.set_ylabel('평균 투찰율 (%)')
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(7, 4))
        budget_bins = pd.cut(df_view['배정예산금액'],
                             bins=[0, 5e7, 1e8, 5e8, 1e9, 1e10, float('inf')],
                             labels=['~5천만', '~1억', '~5억', '~10억', '~100억', '100억+'])
        grouped_b = df_view.groupby(budget_bins, observed=True)['투찰율'].mean()
        ax.bar(range(len(grouped_b)), grouped_b.values, color='#818CF8', alpha=0.8)
        ax.set_xticks(range(len(grouped_b)))
        ax.set_xticklabels(grouped_b.index, rotation=30)
        ax.set_title('예산 규모별 평균 투찰율', fontweight='bold')
        ax.set_ylabel('평균 투찰율 (%)')
        st.pyplot(fig)
        plt.close()

    # EDA 이미지 표시
    st.markdown("---")
    st.subheader("EDA 시각화")
    for folder in ['01output', '02output', '03output', '04output']:
        for img in ['eda_overview.png', 'eda_target_relations.png', 'model_results.png',
                     'model_improved_results.png', 'residual_analysis.png', 'scoring_analysis.png']:
            img_path = os.path.join(BASE_DIR, folder, img)
            if os.path.exists(img_path):
                st.image(img_path, caption=f"{folder}/{img}")
