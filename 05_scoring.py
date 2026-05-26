"""
BID-AI Step 5: 입찰 적합도 스코어링
- 3가지 스코어링 방식 비교 실험
  (1) 휴리스틱 (수동 구간 + 수동 가중치)
  (2) 백분위 기반 (데이터 분포 기반 점수 + 수동 가중치)
  (3) 데이터 기반 (백분위 점수 + Feature Importance 가중치)
- 평가: Spearman 상관계수, 등급별 투찰율 분리도
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
INPUT_DIR = os.path.join(BASE_DIR, '04output')
OUTPUT_DIR = os.path.join(BASE_DIR, '05output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 1. 데이터 & 모델 로드
# ============================================================
print("=" * 60)
print("1. 데이터 & Feature Importance 로드")
print("=" * 60)

# 06output에 새 모델 예측값이 있으면 사용, 없으면 기존 사용
pred_path = os.path.join(BASE_DIR, '06output', 'full_with_predictions.csv')
if os.path.exists(pred_path):
    df_full = pd.read_csv(pred_path, encoding='utf-8-sig')
    print(f"데이터: {len(df_full):,}건 (06 최종 모델 예측값 포함)")
    if 'predicted_rate' in df_full.columns:
        print(f"  예측값 있는 건: {df_full['predicted_rate'].notna().sum():,}건")
else:
    df_full = pd.read_csv(os.path.join(os.path.join(BASE_DIR, '03output'), 'full_featured.csv'), encoding='utf-8-sig')
    print(f"데이터: {len(df_full):,}건 (기존 데이터)")

df_exp = pd.read_csv(os.path.join(INPUT_DIR, 'all_experiment_results.csv'), encoding='utf-8-sig')
print("\n실험 결과:")
print(df_exp.to_string(index=False))
print()

for col in ['개찰예정일자', '공고게시일자', '투찰일자']:
    df_full[col] = pd.to_datetime(df_full[col], errors='coerce')

# Feature Importance 로드 (06output 우선)
fi_path = os.path.join(BASE_DIR, '06output', 'feature_importance_final.csv')
if not os.path.exists(fi_path):
    fi_path = os.path.join(INPUT_DIR, 'feature_importance_v2.csv')
fi = pd.read_csv(fi_path, encoding='utf-8-sig')
print(f"Feature Importance: {len(fi)}개 피처")
print()

# ============================================================
# 2. 업종 매칭 사전 필터
# ============================================================
print("=" * 60)
print("2. 업종 매칭 사전 필터 (공통)")
print("=" * 60)

def check_industry_match(row, target_industries=None):
    if target_industries is None:
        target_industries = ['1468', '1469', '1470', '1471']
    industry_text = str(row.get('면허업종제한목록', ''))
    if any(code in industry_text for code in target_industries):
        return True
    if industry_text == 'nan' or industry_text == '':
        return True
    return False

df_full['industry_match'] = df_full.apply(check_industry_match, axis=1)
print(f"업종 매칭 통과: {df_full['industry_match'].sum():,}건 / {len(df_full):,}건")
print("(업종 매칭은 3가지 방식 모두 동일한 사전 필터로 적용)")
print()

# ============================================================
# 3. 방식 A: 휴리스틱 스코어링
# ============================================================
print("=" * 60)
print("3. 방식 A: 휴리스틱 (수동 구간 + 수동 가중치)")
print("=" * 60)

def score_heuristic(df):
    scores = pd.DataFrame(index=df.index)

    # 낙찰률 점수: 수동 선형 변환
    scores['rate'] = ((df['투찰율'] - 60) / 40 * 100).clip(0, 100)

    # 예산 적합도: 수동 구간
    def budget_manual(b):
        if 3e7 <= b <= 5e9: return 100
        elif b < 3e7: return max(0, b / 3e7 * 80)
        else: return max(0, 100 - (b - 5e9) / 5e9 * 50)
    scores['budget'] = df['배정예산금액'].apply(budget_manual)

    # 경쟁 강도: 수동 구간
    def comp_manual(p):
        if p <= 2: return 100
        elif p <= 5: return 80
        elif p <= 10: return 60
        elif p <= 30: return 40
        else: return 20
    scores['competition'] = df['입찰참여자수'].apply(comp_manual)

    # 수동 가중치
    scores['fit_score'] = (scores['rate'] * 0.45 +
                           scores['budget'] * 0.30 +
                           scores['competition'] * 0.25)
    return scores

heuristic = score_heuristic(df_full)
df_full['fit_A'] = heuristic['fit_score'].round(1)
print(f"  가중치: 낙찰률 0.45 / 예산 0.30 / 경쟁 0.25 (수동 설정)")
print(f"  점수 방식: 수동 구간 (예: 참여자 2명이하=100, 5명이하=80 ...)")
print(f"  평균: {df_full['fit_A'].mean():.1f}")
print()

# ============================================================
# 4. 방식 B: 백분위 기반 스코어링
# ============================================================
print("=" * 60)
print("4. 방식 B: 백분위 기반 (데이터 분포 점수 + 수동 가중치)")
print("=" * 60)

def score_percentile(df):
    scores = pd.DataFrame(index=df.index)

    # 낙찰률: 백분위 (높을수록 좋음)
    scores['rate'] = df['투찰율'].rank(pct=True) * 100

    # 예산 적합도: 적정 범위(3천만~50억)와의 거리 기반 백분위
    def budget_distance(b):
        if 3e7 <= b <= 5e9: return 0  # 범위 내 = 최적
        elif b < 3e7: return (3e7 - b) / 3e7
        else: return (b - 5e9) / 5e9
    dist = df['배정예산금액'].apply(budget_distance)
    # 거리가 작을수록 좋음 → 역순 백분위
    scores['budget'] = (1 - dist.rank(pct=True)) * 100

    # 경쟁 강도: 참여자 적을수록 좋음 → 역순 백분위
    scores['competition'] = (1 - df['입찰참여자수'].rank(pct=True)) * 100

    # 수동 가중치 (방식 A와 동일)
    scores['fit_score'] = (scores['rate'] * 0.45 +
                           scores['budget'] * 0.30 +
                           scores['competition'] * 0.25)
    return scores

percentile = score_percentile(df_full)
df_full['fit_B'] = percentile['fit_score'].round(1)
print(f"  가중치: 낙찰률 0.45 / 예산 0.30 / 경쟁 0.25 (수동, A와 동일)")
print(f"  점수 방식: 백분위 순위 (데이터 분포 기반)")
print(f"  평균: {df_full['fit_B'].mean():.1f}")
print()

# ============================================================
# 5. 방식 C: 데이터 기반 스코어링
# ============================================================
print("=" * 60)
print("5. 방식 C: 데이터 기반 (백분위 점수 + FI 가중치)")
print("=" * 60)

# Feature Importance에서 관련 피처 중요도 추출
fi_dict = dict(zip(fi['feature'], fi['importance']))

# 각 항목에 해당하는 피처들의 importance 합산
rate_features = ['estimate_to_budget', 'budget_cat_mean_rate', 'category_mean_rate',
                 'cat_median', 'cat_std', 'month_mean_rate']
budget_features = ['budget_log', 'budget_per_participant_log']
comp_features = ['competition_intensity', 'participants_log', 'cat_count']

rate_imp = sum(fi_dict.get(f, 0) for f in rate_features)
budget_imp = sum(fi_dict.get(f, 0) for f in budget_features)
comp_imp = sum(fi_dict.get(f, 0) for f in comp_features)
total_imp = rate_imp + budget_imp + comp_imp

w_rate = round(rate_imp / total_imp, 3)
w_budget = round(budget_imp / total_imp, 3)
w_comp = round(comp_imp / total_imp, 3)

print(f"  Feature Importance 기반 가중치 산출:")
print(f"    낙찰률 관련 피처: {rate_features}")
print(f"      → importance 합: {rate_imp:,} → 가중치: {w_rate:.3f}")
print(f"    예산 관련 피처: {budget_features}")
print(f"      → importance 합: {budget_imp:,} → 가중치: {w_budget:.3f}")
print(f"    경쟁 관련 피처: {comp_features}")
print(f"      → importance 합: {comp_imp:,} → 가중치: {w_comp:.3f}")
print()

def score_data_driven(df, w_r, w_b, w_c):
    scores = pd.DataFrame(index=df.index)

    # 백분위 점수 (방식 B와 동일)
    scores['rate'] = df['투찰율'].rank(pct=True) * 100

    def budget_distance(b):
        if 3e7 <= b <= 5e9: return 0
        elif b < 3e7: return (3e7 - b) / 3e7
        else: return (b - 5e9) / 5e9
    dist = df['배정예산금액'].apply(budget_distance)
    scores['budget'] = (1 - dist.rank(pct=True)) * 100

    scores['competition'] = (1 - df['입찰참여자수'].rank(pct=True)) * 100

    # Feature Importance 기반 가중치
    scores['fit_score'] = (scores['rate'] * w_r +
                           scores['budget'] * w_b +
                           scores['competition'] * w_c)
    return scores

data_driven = score_data_driven(df_full, w_rate, w_budget, w_comp)
df_full['fit_C'] = data_driven['fit_score'].round(1)
print(f"  점수 방식: 백분위 순위 (방식 B와 동일)")
print(f"  가중치: 낙찰률 {w_rate:.3f} / 예산 {w_budget:.3f} / 경쟁 {w_comp:.3f} (FI 기반)")
print(f"  평균: {df_full['fit_C'].mean():.1f}")
print()

# ============================================================
# 6. 3가지 방식 비교 평가
# ============================================================
print("=" * 60)
print("6. 스코어링 방식 비교 평가")
print("=" * 60)

def grade_fn(score):
    if score >= 80: return 'A (강력 추천)'
    elif score >= 65: return 'B (추천)'
    elif score >= 50: return 'C (검토)'
    else: return 'D (비추천)'

methods = {
    'A: 휴리스틱': 'fit_A',
    'B: 백분위+수동가중치': 'fit_B',
    'C: 백분위+FI가중치': 'fit_C',
}

eval_results = []

for name, col in methods.items():
    # (1) Spearman 상관계수: Fit Score와 실제 투찰율
    corr, pval = stats.spearmanr(df_full[col], df_full['투찰율'])

    # (2) 등급별 평균 투찰율
    df_full[f'grade_{col}'] = df_full[col].apply(grade_fn)
    grade_means = df_full.groupby(f'grade_{col}')['투찰율'].mean()

    a_mean = grade_means.get('A (강력 추천)', 0)
    b_mean = grade_means.get('B (추천)', 0)
    c_mean = grade_means.get('C (검토)', 0)
    d_mean = grade_means.get('D (비추천)', 0)

    # (3) A-D 등급 투찰율 차이 (분리도)
    separation = a_mean - d_mean

    # (4) 단조성: A > B > C > D 순서인지
    available = [v for v in [a_mean, b_mean, c_mean, d_mean] if v > 0]
    monotonic = all(available[i] >= available[i+1] for i in range(len(available)-1))

    eval_results.append({
        'method': name,
        'spearman_corr': round(corr, 4),
        'p_value': pval,
        'A등급_투찰율': round(a_mean, 2),
        'B등급_투찰율': round(b_mean, 2),
        'C등급_투찰율': round(c_mean, 2),
        'D등급_투찰율': round(d_mean, 2),
        'A-D_분리도': round(separation, 2),
        '단조성': '통과' if monotonic else '실패',
    })

    print(f"\n[{name}]")
    print(f"  Spearman 상관계수: {corr:.4f} (p={pval:.2e})")
    print(f"  등급별 평균 투찰율: A={a_mean:.2f}% / B={b_mean:.2f}% / C={c_mean:.2f}% / D={d_mean:.2f}%")
    print(f"  A-D 분리도: {separation:.2f}%p")
    print(f"  단조성 (A>B>C>D): {'통과' if monotonic else '실패'}")

    grade_dist = df_full[f'grade_{col}'].value_counts().sort_index()
    print(f"  등급 분포: {dict(grade_dist)}")

# 비교 테이블
print()
print("=" * 60)
print("7. 비교 요약")
print("=" * 60)

df_eval = pd.DataFrame(eval_results)
print(df_eval[['method', 'spearman_corr', 'A-D_분리도', '단조성']].to_string(index=False))
print()

# 최적 방식 선택
best_idx = df_eval['spearman_corr'].idxmax()
best = df_eval.loc[best_idx]
print(f"★ 최적 방식: {best['method']}")
print(f"  Spearman: {best['spearman_corr']:.4f}")
print(f"  A-D 분리도: {best['A-D_분리도']:.2f}%p")
print()

# 최적 방식의 컬럼을 fit_score로 사용
best_col = list(methods.values())[best_idx]
df_full['fit_score'] = df_full[best_col]
df_full['grade'] = df_full[f'grade_{best_col}']

# ============================================================
# 8. SW/IT 필터링 (최적 방식 기준)
# ============================================================
print("=" * 60)
print(f"8. SW/IT 필터링 (최적 방식: {best['method']})")
print("=" * 60)

sw_keywords = ['소프트웨어', '정보시스템', '전산', '데이터', '클라우드', '보안', '네트워크']
sw_mask = df_full['공공조달분류'].fillna('').str.contains('|'.join(sw_keywords))
industry_mask = df_full['industry_match'] == True
df_sw = df_full[sw_mask & industry_mask].copy()

print(f"SW/IT 관련 공고: {len(df_sw):,}건")
print(f"Fit Score 평균: {df_sw['fit_score'].mean():.1f}")
print(f"A등급: {(df_sw['grade']=='A (강력 추천)').sum():,}건")
print()

# TOP 10
top10 = df_sw.nlargest(10, 'fit_score')[
    ['입찰공고명', '배정예산금액', '투찰율', '입찰참여자수', 'fit_score', 'grade']
]
print("[SW/IT TOP 10 추천 공고]")
for i, (_, row) in enumerate(top10.iterrows(), 1):
    budget_str = f"{row['배정예산금액']/1e8:.1f}억" if row['배정예산금액'] >= 1e8 else f"{row['배정예산금액']/1e4:.0f}만"
    print(f"  {i}. [{row['grade']}] {row['fit_score']:.1f}점 | {row['입찰공고명'][:40]}")
    print(f"     예산: {budget_str} | 투찰율: {row['투찰율']:.1f}% | 참여: {row['입찰참여자수']}개사")
print()

# ============================================================
# 9. 시각화
# ============================================================
print("=" * 60)
print("9. 시각화")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('BID-AI 스코어링 방식 비교 실험', fontsize=16, fontweight='bold')

colors_method = ['#FB923C', '#38BDF8', '#34D399']
method_labels = ['A: 휴리스틱', 'B: 백분위+수동', 'C: 백분위+FI']
fit_cols = ['fit_A', 'fit_B', 'fit_C']

# 상단: 각 방식별 Fit Score 분포
for i, (col, label, c) in enumerate(zip(fit_cols, method_labels, colors_method)):
    ax = axes[0, i]
    ax.hist(df_full[col], bins=40, color=c, edgecolor='white', alpha=0.8)
    ax.axvline(df_full[col].mean(), color='red', linestyle='--',
               label=f"평균: {df_full[col].mean():.1f}")
    ax.set_title(label, fontweight='bold')
    ax.set_xlabel('Fit Score')
    ax.set_ylabel('건수')
    ax.legend()

# 하단 왼쪽: 등급별 평균 투찰율 비교
ax = axes[1, 0]
grade_order = ['A (강력 추천)', 'B (추천)', 'C (검토)', 'D (비추천)']
x = np.arange(len(grade_order))
w = 0.25
for i, (col, label, c) in enumerate(zip(fit_cols, method_labels, colors_method)):
    g_col = f'grade_{col}'
    means = []
    for g in grade_order:
        subset = df_full[df_full[g_col] == g]['투찰율']
        means.append(subset.mean() if len(subset) > 0 else 0)
    ax.bar(x + (i - 1) * w, means, w, label=label, color=c, alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(['A', 'B', 'C', 'D'])
ax.set_title('등급별 평균 투찰율', fontweight='bold')
ax.set_ylabel('평균 투찰율(%)')
ax.legend(fontsize=9)

# 하단 중앙: Spearman 상관계수 비교
ax = axes[1, 1]
corrs = [df_eval.loc[i, 'spearman_corr'] for i in range(3)]
bars = ax.bar(method_labels, corrs, color=colors_method, alpha=0.8)
for bar, val in zip(bars, corrs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{val:.4f}', ha='center', fontweight='bold')
ax.set_title('Spearman 상관계수 (투찰율)', fontweight='bold')
ax.set_ylabel('상관계수')
ax.tick_params(axis='x', labelsize=9)

# 하단 오른쪽: A-D 분리도 비교
ax = axes[1, 2]
seps = [df_eval.loc[i, 'A-D_분리도'] for i in range(3)]
bars = ax.bar(method_labels, seps, color=colors_method, alpha=0.8)
for bar, val in zip(bars, seps):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f'{val:.2f}%p', ha='center', fontweight='bold')
ax.set_title('A-D 등급 분리도', fontweight='bold')
ax.set_ylabel('투찰율 차이(%p)')
ax.tick_params(axis='x', labelsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'scoring_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  → scoring_comparison.png 저장")

# 기존 형식 차트도 유지
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f'BID-AI 입찰 적합도 스코어링 (최적: {best["method"]})', fontsize=16, fontweight='bold')

ax = axes[0, 0]
ax.hist(df_full['fit_score'], bins=40, color='#38BDF8', edgecolor='white', alpha=0.8)
ax.axvline(df_full['fit_score'].mean(), color='red', linestyle='--',
           label=f"평균: {df_full['fit_score'].mean():.1f}")
ax.set_title('Fit Score 분포 (최적 방식)', fontweight='bold')
ax.set_xlabel('Fit Score')
ax.set_ylabel('건수')
ax.legend()

ax = axes[0, 1]
grade_counts = df_full['grade'].value_counts().sort_index()
colors_grade = ['#34D399', '#38BDF8', '#FB923C', '#F87171']
cg = colors_grade[:len(grade_counts)]
ax.bar(range(len(grade_counts)), grade_counts.values, color=cg)
ax.set_xticks(range(len(grade_counts)))
ax.set_xticklabels(grade_counts.index, fontsize=10)
ax.set_title('등급별 분포', fontweight='bold')
ax.set_ylabel('건수')
for i, v in enumerate(grade_counts.values):
    ax.text(i, v + 100, f'{v:,}', ha='center', fontweight='bold')

ax = axes[1, 0]
sample_sw = df_sw.sample(min(2000, len(df_sw)), random_state=42)
scatter = ax.scatter(
    np.log10(sample_sw['배정예산금액'].clip(lower=1)),
    sample_sw['fit_score'],
    c=sample_sw['투찰율'], cmap='RdYlGn', alpha=0.5, s=15
)
plt.colorbar(scatter, ax=ax, label='투찰율(%)')
ax.set_title('SW/IT: 예산(log) vs Fit Score', fontweight='bold')
ax.set_xlabel('log10(배정예산금액)')
ax.set_ylabel('Fit Score')

ax = axes[1, 1]
sub_labels = ['낙찰률 점수', '예산 적합도', '경쟁강도 점수']
sub_cols = ['rate', 'budget', 'competition']
best_scores = score_data_driven(df_full, w_rate, w_budget, w_comp) if best_col == 'fit_C' else \
              score_percentile(df_full) if best_col == 'fit_B' else score_heuristic(df_full)
sw_idx = df_sw.index
sw_means = [best_scores.loc[sw_idx, s].mean() for s in sub_cols]
all_means = [best_scores[s].mean() for s in sub_cols]
x = np.arange(len(sub_labels))
w = 0.35
ax.bar(x - w/2, all_means, w, label='전체', color='#94A3B8', alpha=0.8)
ax.bar(x + w/2, sw_means, w, label='SW/IT', color='#38BDF8', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(sub_labels)
ax.set_title('세부 점수 비교 (전체 vs SW/IT)', fontweight='bold')
ax.set_ylabel('평균 점수')
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'scoring_analysis.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  → scoring_analysis.png 저장")

# ============================================================
# 10. 저장
# ============================================================
print()
print("=" * 60)
print("10. 저장")
print("=" * 60)

# 전체 스코어링 결과 (3방식 + 최적 방식)
save_cols = [c for c in df_full.columns if not c.startswith('grade_')]
df_full[save_cols].to_csv(os.path.join(OUTPUT_DIR, 'scored_all.csv'), index=False, encoding='utf-8-sig')
print(f"  → scored_all.csv ({len(df_full):,}건)")

df_sw.to_csv(os.path.join(OUTPUT_DIR, 'scored_sw_it.csv'), index=False, encoding='utf-8-sig')
print(f"  → scored_sw_it.csv ({len(df_sw):,}건)")

# 비교 결과 저장
df_eval.to_csv(os.path.join(OUTPUT_DIR, 'scoring_method_comparison.csv'), index=False, encoding='utf-8-sig')
print(f"  → scoring_method_comparison.csv")

# FI 가중치 정보 저장
fi_weights = {'rate': w_rate, 'budget': w_budget, 'competition': w_comp,
              'rate_features': rate_features, 'budget_features': budget_features,
              'comp_features': comp_features, 'best_method': best['method']}
with open(os.path.join(OUTPUT_DIR, 'scoring_weights.pkl'), 'wb') as f:
    pickle.dump(fi_weights, f)
print(f"  → scoring_weights.pkl")

print()
print("=" * 60)
print("스코어링 비교 실험 완료")
print("=" * 60)
print(f"  ★ 최적 방식: {best['method']}")
print(f"  Spearman: {best['spearman_corr']:.4f}")
print(f"  A-D 분리도: {best['A-D_분리도']:.2f}%p")
print(f"  단조성: {best['단조성']}")
print(f"  SW/IT: {len(df_sw):,}건 / A등급: {(df_sw['grade']=='A (강력 추천)').sum():,}건")
