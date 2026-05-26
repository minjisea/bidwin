"""
BID-AI Step 2: Feature Engineering
- 텍스트 피처 (TF-IDF, 키워드)
- 수치 피처 (로그 변환, 구간화)
- 범주형 피처 (인코딩, 그룹핑)
- 파생 피처 (기관별/업종별 통계)
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import re
import os
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# ============================================================
# 1. 전처리된 데이터 로드
# ============================================================
print("=" * 60)
print("1. 데이터 로드")
print("=" * 60)

df = pd.read_csv(os.path.join(OUTPUT_DIR, 'preprocessed.csv'), encoding='utf-8-sig')
print(f"로드 완료: {len(df):,}건, {len(df.columns)}개 컬럼")

# 날짜 컬럼 복원
date_cols = ['개찰예정일자', '공고게시일자', '투찰일자', '입찰서접수시작일자', '입찰서접수마감일자']
for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')

print()

# ============================================================
# 2. 텍스트 피처
# ============================================================
print("=" * 60)
print("2. 텍스트 피처 생성")
print("=" * 60)

# (1) 공고명 키워드 피처
keywords = {
    'kw_si': ['시스템', 'SI', '구축', '개발', '플랫폼'],
    'kw_maintain': ['유지보수', '유지관리', '운영', '운용'],
    'kw_consult': ['컨설팅', 'ISP', 'ISMP', '감리', '진단'],
    'kw_infra': ['인프라', '서버', '네트워크', '클라우드', 'IDC'],
    'kw_security': ['보안', '정보보호', '개인정보', '취약점'],
    'kw_data': ['데이터', '빅데이터', 'AI', '인공지능', '분석'],
    'kw_sw': ['소프트웨어', 'SW', '라이선스', '패키지'],
    'kw_education': ['교육', '훈련', '연수'],
    'kw_research': ['연구', '조사', '용역'],
    'kw_construction': ['공사', '시공', '건설', '철거', '설치'],
    'kw_cleaning': ['청소', '미화', '소독', '방역'],
    'kw_transport': ['운송', '운행', '수학여행', '현장체험'],
    'kw_waste': ['폐기물', '처리', '수거', '재활용'],
}

for feat_name, kw_list in keywords.items():
    pattern = '|'.join(kw_list)
    df[feat_name] = df['입찰공고명'].str.contains(pattern, case=False, na=False).astype(int)

print(f"  키워드 피처 {len(keywords)}개 생성")

# (2) 공고명 길이
df['title_length'] = df['입찰공고명'].str.len()

# (3) TF-IDF (상위 300차원)
print("  TF-IDF 벡터화 중...")
tfidf = TfidfVectorizer(max_features=300, analyzer='char_wb', ngram_range=(2, 4))
tfidf_matrix = tfidf.fit_transform(df['입찰공고명'].fillna(''))
tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])],
    index=df.index
)
print(f"  TF-IDF {tfidf_matrix.shape[1]}차원 생성 완료")
print()

# ============================================================
# 3. 수치 피처
# ============================================================
print("=" * 60)
print("3. 수치 피처 생성")
print("=" * 60)

# (1) 배정예산 로그 변환
df['budget_log'] = np.log1p(df['배정예산금액'])

# (2) 예산 규모 구간
df['budget_tier'] = pd.cut(
    df['배정예산금액'],
    bins=[0, 3e7, 5e7, 1e8, 3e8, 5e8, 1e9, 5e9, float('inf')],
    labels=['~3천만', '~5천만', '~1억', '~3억', '~5억', '~10억', '~50억', '50억+']
)

# (3) 입찰참여자수 로그 변환
df['participants_log'] = np.log1p(df['입찰참여자수'])

# (4) 참여자수 구간
df['participants_tier'] = pd.cut(
    df['입찰참여자수'],
    bins=[0, 1, 3, 5, 10, 30, float('inf')],
    labels=['수의(1)', '2~3', '4~5', '6~10', '11~30', '30+']
)

# (5) 낙찰하한율 존재 여부
df['has_lower_limit'] = df['낙찰하한율'].notna().astype(int)

# (6) 기초금액 존재 여부 및 비율
df['has_base_price'] = df['기초금액'].notna().astype(int)
df['base_to_budget_ratio'] = (df['기초금액'] / df['배정예산금액']).clip(-10, 10)
df['base_to_budget_ratio'] = df['base_to_budget_ratio'].fillna(0)

# (7) 예정가격/배정예산 비율
df['estimate_to_budget'] = (df['예정가격'] / df['배정예산금액']).clip(-10, 10)
df['estimate_to_budget'] = df['estimate_to_budget'].fillna(0)

print("  budget_log, budget_tier, participants_log, participants_tier 등 생성")
print()

# ============================================================
# 4. 범주형 피처
# ============================================================
print("=" * 60)
print("4. 범주형 피처 생성")
print("=" * 60)

# (1) 공공조달분류 → Label Encoding
le_category = LabelEncoder()
df['category_encoded'] = le_category.fit_transform(df['공공조달분류'].fillna('기타'))
print(f"  공공조달분류: {len(le_category.classes_)}개 카테고리 인코딩")

# (2) 공공조달분류 빈도 기반 피처 (출현 빈도가 적은 것은 '기타'로 묶기)
category_counts = df['공공조달분류'].value_counts()
top_categories = category_counts[category_counts >= 100].index.tolist()
df['category_grouped'] = df['공공조달분류'].apply(lambda x: x if x in top_categories else '기타')
le_cat_group = LabelEncoder()
df['category_grouped_encoded'] = le_cat_group.fit_transform(df['category_grouped'])
print(f"  공공조달분류 그룹핑: {len(top_categories)}개 주요 + 기타")

# (3) 면허업종제한 파싱
def extract_industry_codes(text):
    if pd.isna(text):
        return []
    codes = re.findall(r'\((\d{4})\)', str(text))
    return codes

df['industry_code_count'] = df['면허업종제한목록'].apply(lambda x: len(extract_industry_codes(x)))
df['has_industry_limit'] = df['면허업종제한목록'].notna().astype(int)

# SW 관련 업종코드 체크
sw_codes = ['1468', '1469', '1470', '1471']  # 소프트웨어사업자 관련
df['is_sw_industry'] = df['면허업종제한목록'].fillna('').apply(
    lambda x: int(any(code in str(x) for code in sw_codes))
)
print(f"  업종제한: 코드 수, SW업종 여부 등 생성")

# (4) 지역제한 여부
df['has_region_limit'] = df['제한지역코드목록'].notna().astype(int)

print()

# ============================================================
# 5. 시간 피처
# ============================================================
print("=" * 60)
print("5. 시간 피처 생성")
print("=" * 60)

# (1) 공고 월, 요일, 분기
df['month'] = df['공고게시일자'].dt.month
df['dayofweek'] = df['공고게시일자'].dt.dayofweek
df['quarter'] = df['공고게시일자'].dt.quarter

# (2) 접수 기간 (일수)
df['submission_days'] = (df['입찰서접수마감일자'] - df['입찰서접수시작일자']).dt.days
df['submission_days'] = df['submission_days'].fillna(0).clip(0, 365)

# (3) 공고~개찰 기간
df['announce_to_open_days'] = (df['개찰예정일자'] - df['공고게시일자']).dt.days
df['announce_to_open_days'] = df['announce_to_open_days'].fillna(0).clip(0, 365)

print("  month, quarter, dayofweek, submission_days, announce_to_open_days 생성")
print()

# ============================================================
# 6. 파생 통계 피처 (기관별/업종별 평균 투찰율)
# ============================================================
print("=" * 60)
print("6. 파생 통계 피처 (Target Encoding 방식)")
print("=" * 60)

# (1) 공공조달분류별 평균 투찰율
cat_mean = df.groupby('공공조달분류')['투찰율'].mean().to_dict()
df['category_mean_rate'] = df['공공조달분류'].map(cat_mean)

# (2) 예산 구간별 평균 투찰율
tier_mean = df.groupby('budget_tier', observed=True)['투찰율'].mean().to_dict()
df['budget_tier_mean_rate'] = df['budget_tier'].map(tier_mean)

# (3) 참여자 구간별 평균 투찰율
part_mean = df.groupby('participants_tier', observed=True)['투찰율'].mean().to_dict()
df['participants_tier_mean_rate'] = df['participants_tier'].map(part_mean)

# (4) 월별 평균 투찰율
month_mean = df.groupby('month')['투찰율'].mean().to_dict()
df['month_mean_rate'] = df['month'].map(month_mean)

print("  category_mean_rate, budget_tier_mean_rate, participants_tier_mean_rate, month_mean_rate 생성")
print()

# ============================================================
# 7. 최종 피처 셋 구성
# ============================================================
print("=" * 60)
print("7. 최종 피처 셋 구성")
print("=" * 60)

# 수치형 피처 목록
numeric_features = [
    # 원본 수치
    'budget_log', 'participants_log',
    # 비율/관계
    'base_to_budget_ratio', 'estimate_to_budget',
    # 이진
    'has_lower_limit', 'has_base_price', 'has_industry_limit',
    'has_region_limit', 'is_sw_industry',
    # 키워드
    'kw_si', 'kw_maintain', 'kw_consult', 'kw_infra', 'kw_security',
    'kw_data', 'kw_sw', 'kw_education', 'kw_research',
    'kw_construction', 'kw_cleaning', 'kw_transport', 'kw_waste',
    # 텍스트
    'title_length',
    # 범주 인코딩
    'category_grouped_encoded', 'industry_code_count',
    # 시간
    'month', 'dayofweek', 'quarter',
    'submission_days', 'announce_to_open_days',
    # 파생 통계
    'category_mean_rate', 'budget_tier_mean_rate',
    'participants_tier_mean_rate', 'month_mean_rate',
]

target = '투찰율'

# TF-IDF 합치기
df_features = pd.concat([df[numeric_features + [target]].reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)

# 결측치 처리 (수치형만)
for col in df_features.columns:
    if df_features[col].dtype in ['float64', 'int64', 'float32', 'int32']:
        df_features[col] = df_features[col].fillna(0)

print(f"  수치/범주/이진 피처: {len(numeric_features)}개")
print(f"  TF-IDF 피처: {tfidf_df.shape[1]}개")
print(f"  총 피처 수: {df_features.shape[1] - 1}개 (Target 제외)")
print(f"  데이터 수: {len(df_features):,}건")
print()

# ============================================================
# 8. 저장
# ============================================================
print("=" * 60)
print("8. 저장")
print("=" * 60)

# 피처 데이터 저장
feature_path = os.path.join(OUTPUT_DIR, 'features.csv')
df_features.to_csv(feature_path, index=False, encoding='utf-8-sig')
print(f"  → features.csv ({len(df_features):,}건, {df_features.shape[1]}개 컬럼)")

# 피처 목록 저장
feature_list_path = os.path.join(OUTPUT_DIR, 'feature_list.txt')
all_features = [c for c in df_features.columns if c != target]
with open(feature_list_path, 'w', encoding='utf-8') as f:
    f.write(f"총 피처 수: {len(all_features)}\n\n")
    f.write("=== 수치/범주/이진 피처 ===\n")
    for feat in numeric_features:
        f.write(f"  {feat}\n")
    f.write(f"\n=== TF-IDF 피처 ({tfidf_df.shape[1]}개) ===\n")
    f.write(f"  tfidf_0 ~ tfidf_{tfidf_df.shape[1]-1}\n")
print(f"  → feature_list.txt")

# TF-IDF 모델 저장
import pickle
tfidf_path = os.path.join(OUTPUT_DIR, 'tfidf_model.pkl')
with open(tfidf_path, 'wb') as f:
    pickle.dump(tfidf, f)
print(f"  → tfidf_model.pkl")

# 원본 + 파생피처 전체 저장 (나중에 제안서 생성 등에 사용)
full_path = os.path.join(OUTPUT_DIR, 'full_featured.csv')
df.to_csv(full_path, index=False, encoding='utf-8-sig')
print(f"  → full_featured.csv ({len(df):,}건)")

print()
print("=" * 60)
print("Feature Engineering 완료 요약")
print("=" * 60)
print(f"  총 피처: {len(all_features)}개")
print(f"    - 수치/범주/이진: {len(numeric_features)}개")
print(f"    - TF-IDF (char ngram): {tfidf_df.shape[1]}개")
print(f"  데이터: {len(df_features):,}건")
print(f"  Target: 투찰율 (평균 {df_features[target].mean():.2f}%)")
