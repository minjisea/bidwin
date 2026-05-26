"""
BID-AI Step 1: 데이터 전처리 + EDA
- 조달청 입찰공고 결과내역 전처리
- 탐색적 데이터 분석 (EDA)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 1. 데이터 로드
# ============================================================
print("=" * 60)
print("1. 데이터 로드")
print("=" * 60)

data_dir = BASE_DIR
csv_file = [f for f in os.listdir(data_dir) if f.endswith('.csv')][0]
df_raw = pd.read_csv(
    os.path.join(data_dir, csv_file),
    encoding='utf-16', sep='\t', skiprows=70, quotechar='"', low_memory=False
)

print(f"원본 데이터: {df_raw.shape[0]:,}건, {df_raw.shape[1]}개 컬럼")
print()

# ============================================================
# 2. 컬럼 정리 및 타입 변환
# ============================================================
print("=" * 60)
print("2. 컬럼 정리 및 타입 변환")
print("=" * 60)

df = df_raw.copy()

# 금액 컬럼: 쉼표 제거 후 숫자 변환
money_cols = ['기초금액', '합계배정금액', '배정예산금액', '예정가격', '투찰금액', '입찰추정가격']
for col in money_cols:
    df[col] = df[col].astype(str).str.replace(',', '').str.strip()
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 수치 컬럼 변환
df['입찰참여자수'] = pd.to_numeric(df['입찰참여자수'].astype(str).str.replace(',', ''), errors='coerce')
df['투찰율'] = pd.to_numeric(df['투찰율'], errors='coerce')
df['낙찰하한율'] = pd.to_numeric(df['낙찰하한율'], errors='coerce')

# 날짜 컬럼 변환
date_cols = ['개찰예정일자', '공고게시일자', '투찰일자', '입찰서접수시작일자', '입찰서접수마감일자']
for col in date_cols:
    df[col] = pd.to_datetime(df[col], format='%Y%m%d', errors='coerce')

print("타입 변환 완료")
print()

# ============================================================
# 3. 결측치 및 이상치 처리
# ============================================================
print("=" * 60)
print("3. 결측치 및 이상치 처리")
print("=" * 60)

print(f"변환 후 전체: {len(df):,}건")
print()

# 핵심 컬럼 결측치 확인
core_cols = ['입찰공고명', '배정예산금액', '투찰금액', '투찰율', '공고게시일자', '공공조달분류']
print("[핵심 컬럼 결측치]")
for col in core_cols:
    null_cnt = df[col].isnull().sum()
    print(f"  {col}: {null_cnt:,}건 ({null_cnt/len(df)*100:.1f}%)")
print()

# 핵심 컬럼 결측 제거
before = len(df)
df = df.dropna(subset=['입찰공고명', '배정예산금액', '투찰금액', '투찰율', '공고게시일자'])
print(f"핵심 컬럼 결측 제거: {before:,} → {len(df):,}건 ({before - len(df):,}건 제거)")

# 이상치 제거
# 배정예산금액 0 이하 제거
before = len(df)
df = df[df['배정예산금액'] > 0]
print(f"배정예산 0 이하 제거: {before:,} → {len(df):,}건")

# 투찰율 범위 확인 및 필터
before = len(df)
df = df[(df['투찰율'] > 0) & (df['투찰율'] <= 100)]
print(f"투찰율 0~100% 범위 필터: {before:,} → {len(df):,}건")

# 투찰금액 0 이하 제거
before = len(df)
df = df[df['투찰금액'] > 0]
print(f"투찰금액 0 이하 제거: {before:,} → {len(df):,}건")

print(f"\n최종 데이터: {len(df):,}건")
print()

# ============================================================
# 4. 기초 통계
# ============================================================
print("=" * 60)
print("4. 기초 통계")
print("=" * 60)

print("\n[투찰율 (Target) 통계]")
print(df['투찰율'].describe().to_string())
print()

print("[배정예산금액 통계]")
print(df['배정예산금액'].describe().apply(lambda x: f"{x:,.0f}").to_string())
print()

print("[입찰참여자수 통계]")
print(df['입찰참여자수'].describe().to_string())
print()

print("[공공조달분류 TOP 15]")
print(df['공공조달분류'].value_counts().head(15).to_string())
print()

# ============================================================
# 5. EDA 시각화
# ============================================================
print("=" * 60)
print("5. EDA 시각화 생성")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('BID-AI EDA: 입찰공고 결과내역 탐색적 분석', fontsize=16, fontweight='bold')

# (1) 투찰율 분포
ax = axes[0, 0]
ax.hist(df['투찰율'], bins=50, color='#38BDF8', edgecolor='white', alpha=0.8)
ax.axvline(df['투찰율'].mean(), color='red', linestyle='--', label=f"평균: {df['투찰율'].mean():.1f}%")
ax.axvline(df['투찰율'].median(), color='orange', linestyle='--', label=f"중앙값: {df['투찰율'].median():.1f}%")
ax.set_title('투찰율 (Target) 분포', fontweight='bold')
ax.set_xlabel('투찰율 (%)')
ax.set_ylabel('건수')
ax.legend()

# (2) 배정예산금액 분포 (로그 스케일)
ax = axes[0, 1]
budget_log = np.log10(df['배정예산금액'].clip(lower=1))
ax.hist(budget_log, bins=50, color='#818CF8', edgecolor='white', alpha=0.8)
ax.set_title('배정예산금액 분포 (log10)', fontweight='bold')
ax.set_xlabel('log10(배정예산금액)')
ax.set_ylabel('건수')

# (3) 입찰참여자수 분포
ax = axes[0, 2]
participants = df['입찰참여자수'].clip(upper=df['입찰참여자수'].quantile(0.95))
ax.hist(participants, bins=50, color='#34D399', edgecolor='white', alpha=0.8)
ax.set_title('입찰참여자수 분포 (상위 5% 클리핑)', fontweight='bold')
ax.set_xlabel('참여자수')
ax.set_ylabel('건수')

# (4) 공공조달분류 TOP 10
ax = axes[1, 0]
top10 = df['공공조달분류'].value_counts().head(10)
bars = ax.barh(range(len(top10)), top10.values, color='#38BDF8', alpha=0.8)
ax.set_yticks(range(len(top10)))
ax.set_yticklabels(top10.index, fontsize=9)
ax.set_title('공공조달분류 TOP 10', fontweight='bold')
ax.set_xlabel('건수')
ax.invert_yaxis()

# (5) 배정예산 vs 투찰율 산점도
ax = axes[1, 1]
sample = df.sample(min(3000, len(df)), random_state=42)
ax.scatter(np.log10(sample['배정예산금액'].clip(lower=1)), sample['투찰율'],
           alpha=0.15, s=10, color='#818CF8')
ax.set_title('배정예산(log) vs 투찰율', fontweight='bold')
ax.set_xlabel('log10(배정예산금액)')
ax.set_ylabel('투찰율 (%)')

# (6) 월별 공고 건수 추이
ax = axes[1, 2]
monthly = df.groupby(df['공고게시일자'].dt.to_period('M')).size()
ax.plot(range(len(monthly)), monthly.values, marker='o', color='#FB923C', markersize=4)
ax.set_title('월별 공고 건수 추이', fontweight='bold')
ax.set_xlabel('월')
ax.set_ylabel('건수')
# x축 라벨 간소화
tick_positions = range(0, len(monthly), max(1, len(monthly) // 6))
ax.set_xticks(tick_positions)
ax.set_xticklabels([str(monthly.index[i]) for i in tick_positions], rotation=45, fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'eda_overview.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  → eda_overview.png 저장 완료")

# ── 추가 차트: 투찰율 vs 주요 변수 상관 ──
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('투찰율(Target)과 주요 변수 관계', fontsize=14, fontweight='bold')

# (1) 입찰참여자수 vs 투찰율
ax = axes[0]
bins_part = pd.cut(df['입찰참여자수'], bins=[0, 1, 3, 5, 10, 50, 1000],
                   labels=['1', '2~3', '4~5', '6~10', '11~50', '50+'])
grouped = df.groupby(bins_part, observed=True)['투찰율'].mean()
ax.bar(range(len(grouped)), grouped.values, color='#34D399', alpha=0.8)
ax.set_xticks(range(len(grouped)))
ax.set_xticklabels(grouped.index)
ax.set_title('참여자수 구간별 평균 투찰율', fontweight='bold')
ax.set_xlabel('참여자수 구간')
ax.set_ylabel('평균 투찰율 (%)')

# (2) 예산 규모별 평균 투찰율
ax = axes[1]
budget_bins = pd.cut(df['배정예산금액'],
                     bins=[0, 5e7, 1e8, 5e8, 1e9, 1e10, float('inf')],
                     labels=['~5천만', '~1억', '~5억', '~10억', '~100억', '100억+'])
grouped_b = df.groupby(budget_bins, observed=True)['투찰율'].mean()
ax.bar(range(len(grouped_b)), grouped_b.values, color='#818CF8', alpha=0.8)
ax.set_xticks(range(len(grouped_b)))
ax.set_xticklabels(grouped_b.index, rotation=30)
ax.set_title('예산 규모별 평균 투찰율', fontweight='bold')
ax.set_xlabel('예산 규모')
ax.set_ylabel('평균 투찰율 (%)')

# (3) 공공조달분류 TOP10별 평균 투찰율
ax = axes[2]
top10_cats = df['공공조달분류'].value_counts().head(10).index
df_top10 = df[df['공공조달분류'].isin(top10_cats)]
cat_mean = df_top10.groupby('공공조달분류')['투찰율'].mean().sort_values(ascending=True)
ax.barh(range(len(cat_mean)), cat_mean.values, color='#FB923C', alpha=0.8)
ax.set_yticks(range(len(cat_mean)))
ax.set_yticklabels(cat_mean.index, fontsize=9)
ax.set_title('조달분류별 평균 투찰율 (TOP10)', fontweight='bold')
ax.set_xlabel('평균 투찰율 (%)')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'eda_target_relations.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  → eda_target_relations.png 저장 완료")

# ============================================================
# 6. 수치형 컬럼 상관관계
# ============================================================
print()
print("=" * 60)
print("6. 수치형 컬럼 상관관계 (투찰율 기준)")
print("=" * 60)

numeric_cols = ['투찰율', '배정예산금액', '투찰금액', '입찰참여자수', '기초금액', '예정가격', '낙찰하한율', '입찰추정가격']
corr = df[numeric_cols].corr()['투찰율'].drop('투찰율').sort_values(ascending=False)
print(corr.to_string())
print()

# ============================================================
# 7. 전처리된 데이터 저장
# ============================================================
print("=" * 60)
print("7. 전처리된 데이터 저장")
print("=" * 60)

save_path = os.path.join(OUTPUT_DIR, 'preprocessed.csv')
df.to_csv(save_path, index=False, encoding='utf-8-sig')
print(f"  → {save_path}")
print(f"  → {len(df):,}건 저장 완료")
print()

# ============================================================
# 요약
# ============================================================
print("=" * 60)
print("전처리 + EDA 요약")
print("=" * 60)
print(f"  원본 데이터:     {df_raw.shape[0]:,}건")
print(f"  전처리 후:       {len(df):,}건")
print(f"  제거된 데이터:   {df_raw.shape[0] - len(df):,}건")
print(f"  Target (투찰율): 평균 {df['투찰율'].mean():.2f}%, 중앙값 {df['투찰율'].median():.2f}%")
print(f"  배정예산 범위:   {df['배정예산금액'].min():,.0f} ~ {df['배정예산금액'].max():,.0f}원")
print(f"  공공조달분류:    {df['공공조달분류'].nunique()}개 카테고리")
print(f"  시각화 저장:     output/eda_overview.png, output/eda_target_relations.png")
print(f"  전처리 데이터:   output/preprocessed.csv")
