"""
BID-AI Step 6: 논문급 실험 보강
- 다중 모델 비교 (LightGBM, XGBoost, CatBoost, RandomForest, Ridge)
- 반복 실험 (3×10-Fold CV) + 표준편차
- 통계 유의성 검증 (Paired t-test)
- Ablation Study (피처 그룹별 기여도)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import pickle
import os
import time
import warnings
warnings.filterwarnings('ignore')

matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
INPUT_DIR = os.path.join(BASE_DIR, '04output')
OUTPUT_DIR = os.path.join(BASE_DIR, '06output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 1. 데이터 로드 (04 단계 전처리 재현)
# ============================================================
print("=" * 60)
print("1. 데이터 로드 & 피처 준비")
print("=" * 60)

df_full = pd.read_csv(os.path.join(os.path.join(BASE_DIR, '03output'), 'full_featured.csv'), encoding='utf-8-sig')
for col in ['개찰예정일자', '공고게시일자', '투찰일자', '입찰서접수시작일자', '입찰서접수마감일자']:
    if col in df_full.columns:
        df_full[col] = pd.to_datetime(df_full[col], errors='coerce')

# 이상치 제거
df = df_full[df_full['투찰율'] >= 60].copy()
print(f"데이터: {len(df):,}건 (투찰율 60% 미만 제거)")

# 추가 피처 (04 단계와 동일)
df['is_private'] = (df['입찰참여자수'] == 1).astype(int)
df['competition_intensity'] = np.log1p(df['입찰참여자수']) * np.log1p(df['배정예산금액'])
df['budget_per_participant'] = df['배정예산금액'] / df['입찰참여자수'].clip(lower=1)
df['budget_per_participant_log'] = np.log1p(df['budget_per_participant'])
df['has_year_in_title'] = df['입찰공고명'].str.contains(r'20\d{2}', na=False).astype(int)
df['has_number_in_title'] = df['입찰공고명'].str.contains(r'제?\d+차|제?\d+호', na=False).astype(int)
df['title_word_count'] = df['입찰공고명'].str.split().str.len()
df['lower_limit_gap'] = df['투찰율'] - df['낙찰하한율']
df['lower_limit_gap'] = df['lower_limit_gap'].fillna(0)

cat_stats = df.groupby('공공조달분류')['투찰율'].agg(['mean', 'std', 'median', 'count'])
cat_stats.columns = ['cat_mean', 'cat_std', 'cat_median', 'cat_count']
df = df.merge(cat_stats, left_on='공공조달분류', right_index=True, how='left')

part_stats = df.groupby('입찰참여자수')['투찰율'].agg(['mean', 'count'])
part_stats.columns = ['part_exact_mean', 'part_exact_count']
df = df.merge(part_stats, left_on='입찰참여자수', right_index=True, how='left')

df['budget_cat_interaction'] = (
    pd.qcut(df['배정예산금액'], q=10, labels=False, duplicates='drop').astype(str) + '_' +
    df['category_grouped'].astype(str)
)
interact_mean = df.groupby('budget_cat_interaction')['투찰율'].mean()
df['budget_cat_mean_rate'] = df['budget_cat_interaction'].map(interact_mean)

# TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_char = TfidfVectorizer(max_features=200, analyzer='char_wb', ngram_range=(2, 4))
tfidf_char_mat = tfidf_char.fit_transform(df['입찰공고명'].fillna(''))
tfidf_word = TfidfVectorizer(max_features=100, analyzer='word', ngram_range=(1, 2), min_df=5)
tfidf_word_mat = tfidf_word.fit_transform(df['입찰공고명'].fillna(''))

tfidf_char_df = pd.DataFrame(
    tfidf_char_mat.toarray(),
    columns=[f'tfidf_c{i}' for i in range(tfidf_char_mat.shape[1])],
    index=df.index
)
tfidf_word_df = pd.DataFrame(
    tfidf_word_mat.toarray(),
    columns=[f'tfidf_w{i}' for i in range(tfidf_word_mat.shape[1])],
    index=df.index
)

# 피처 그룹 정의 (Ablation Study용)
text_features = [f'tfidf_c{i}' for i in range(200)] + [f'tfidf_w{i}' for i in range(100)]
keyword_features = ['kw_si', 'kw_maintain', 'kw_consult', 'kw_infra', 'kw_security',
                    'kw_data', 'kw_sw', 'kw_education', 'kw_research',
                    'kw_construction', 'kw_cleaning', 'kw_transport', 'kw_waste',
                    'title_length', 'title_word_count', 'has_year_in_title', 'has_number_in_title']
numeric_base = ['budget_log', 'participants_log', 'base_to_budget_ratio', 'estimate_to_budget',
                'has_lower_limit', 'has_base_price', 'has_industry_limit', 'has_region_limit',
                'is_sw_industry']
derived_features = ['is_private', 'competition_intensity', 'budget_per_participant_log',
                    'lower_limit_gap']
stat_features = ['category_mean_rate', 'budget_tier_mean_rate', 'participants_tier_mean_rate',
                 'month_mean_rate', 'cat_mean', 'cat_std', 'cat_median', 'cat_count',
                 'part_exact_mean', 'part_exact_count', 'budget_cat_mean_rate']
category_features = ['category_grouped_encoded', 'industry_code_count']
time_features = ['month', 'dayofweek', 'quarter', 'submission_days', 'announce_to_open_days']

all_numeric = (numeric_base + keyword_features + derived_features +
               stat_features + category_features + time_features)
all_numeric = [f for f in all_numeric if f in df.columns]

target = '투찰율'

df_feat = pd.concat([
    df[all_numeric + [target]].reset_index(drop=True),
    tfidf_char_df.reset_index(drop=True),
    tfidf_word_df.reset_index(drop=True),
], axis=1)

for col in df_feat.columns:
    if df_feat[col].dtype in ['float64', 'int64', 'float32', 'int32']:
        df_feat[col] = df_feat[col].fillna(0)

feature_cols = [c for c in df_feat.columns if c != target]
X = df_feat[feature_cols].values
y = df_feat[target].values

print(f"피처: {len(feature_cols)}개, 데이터: {len(X):,}건")
print()

# ============================================================
# 2. 모델 정의
# ============================================================
print("=" * 60)
print("2. 비교 모델 정의 (5개)")
print("=" * 60)

# LightGBM 최적 파라미터 (04 단계에서 찾은 것)
models = {
    'Ridge': Ridge(alpha=1.0),
    'XGBoost': xgb.XGBRegressor(
        n_estimators=500, learning_rate=0.03, max_depth=8,
        colsample_bytree=0.7, subsample=0.8,
        reg_alpha=0.05, reg_lambda=0.5,
        early_stopping_rounds=30,
        random_state=42, n_jobs=-1, verbosity=0
    ),
    'CatBoost': CatBoostRegressor(
        iterations=500, learning_rate=0.05, depth=8,
        l2_leaf_reg=3, random_seed=42, verbose=0
    ),
    'LightGBM': lgb.LGBMRegressor(
        n_estimators=500, learning_rate=0.03, max_depth=8,
        num_leaves=127, colsample_bytree=0.7,
        min_child_samples=20, subsample=0.8,
        reg_alpha=0.05, reg_lambda=0.5,
        random_state=42, verbose=-1, n_jobs=-1
    ),
}

for name in models:
    print(f"  - {name}")
print()

# ============================================================
# 3. 3×10-Fold 반복 교차 검증
# ============================================================
print("=" * 60)
print("3. 3×10-Fold 반복 교차 검증")
print("=" * 60)

n_repeats = 3
n_splits = 5
rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

cv_results = {name: {'MAE': [], 'RMSE': [], 'R2': []} for name in models}

total_folds = n_repeats * n_splits
for name, model_template in models.items():
    print(f"\n[{name}] 실험 중...")
    start = time.time()

    for fold_i, (train_idx, val_idx) in enumerate(rkf.split(X)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # 매 fold마다 새 모델 인스턴스 생성
        from sklearn.base import clone
        try:
            model = clone(model_template)
        except:
            # CatBoost는 clone 안될 수 있음
            if name == 'CatBoost':
                model = CatBoostRegressor(
                    iterations=500, learning_rate=0.05, depth=8,
                    l2_leaf_reg=3, random_seed=42, verbose=0)
            else:
                model = clone(model_template)

        # Early stopping for tree models
        if name == 'LightGBM':
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        elif name == 'XGBoost':
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                      verbose=False)
        elif name == 'CatBoost':
            model.fit(X_tr, y_tr, eval_set=(X_val, y_val),
                      early_stopping_rounds=50, verbose=False)
        else:
            model.fit(X_tr, y_tr)

        y_pred = model.predict(X_val)
        cv_results[name]['MAE'].append(mean_absolute_error(y_val, y_pred))
        cv_results[name]['RMSE'].append(np.sqrt(mean_squared_error(y_val, y_pred)))
        cv_results[name]['R2'].append(r2_score(y_val, y_pred))

        if (fold_i + 1) % 10 == 0:
            elapsed = time.time() - start
            print(f"  {fold_i+1}/{total_folds} folds 완료 ({elapsed:.0f}초)")

    elapsed = time.time() - start
    mae_arr = np.array(cv_results[name]['MAE'])
    print(f"  완료: MAE = {mae_arr.mean():.4f} ± {mae_arr.std():.4f} ({elapsed:.0f}초)")

# ============================================================
# 4. 결과 정리
# ============================================================
print()
print("=" * 60)
print("4. 모델 비교 결과 (3×10-Fold CV)")
print("=" * 60)

summary = []
for name in models:
    mae_arr = np.array(cv_results[name]['MAE'])
    rmse_arr = np.array(cv_results[name]['RMSE'])
    r2_arr = np.array(cv_results[name]['R2'])
    summary.append({
        'Model': name,
        'MAE_mean': round(mae_arr.mean(), 4),
        'MAE_std': round(mae_arr.std(), 4),
        'RMSE_mean': round(rmse_arr.mean(), 4),
        'RMSE_std': round(rmse_arr.std(), 4),
        'R2_mean': round(r2_arr.mean(), 4),
        'R2_std': round(r2_arr.std(), 4),
    })

df_summary = pd.DataFrame(summary).sort_values('MAE_mean')
print(df_summary.to_string(index=False))
print()

best_model = df_summary.iloc[0]['Model']
print(f"★ 최적 모델: {best_model}")
print(f"  MAE: {df_summary.iloc[0]['MAE_mean']:.4f} ± {df_summary.iloc[0]['MAE_std']:.4f}")
print(f"  R²:  {df_summary.iloc[0]['R2_mean']:.4f} ± {df_summary.iloc[0]['R2_std']:.4f}")
print()

# ============================================================
# 5. 통계 유의성 검증 (Paired t-test)
# ============================================================
print("=" * 60)
print("5. 통계 유의성 검증 (Paired t-test)")
print("=" * 60)

best_mae_arr = np.array(cv_results[best_model]['MAE'])
ttest_results = []

for name in models:
    if name == best_model:
        continue
    other_mae = np.array(cv_results[name]['MAE'])
    t_stat, p_val = stats.ttest_rel(best_mae_arr, other_mae)
    sig = "유의 (p<0.05)" if p_val < 0.05 else "유의하지 않음"
    ttest_results.append({
        'Comparison': f'{best_model} vs {name}',
        't_statistic': round(t_stat, 4),
        'p_value': round(p_val, 6),
        'significant': sig,
    })
    print(f"  {best_model} vs {name}: t={t_stat:.4f}, p={p_val:.6f} → {sig}")

df_ttest = pd.DataFrame(ttest_results)
print()

# ============================================================
# 6. Ablation Study (피처 그룹별 기여도)
# ============================================================
print("=" * 60)
print("6. Ablation Study (피처 그룹별 기여도)")
print("=" * 60)

# 피처 그룹 인덱스 매핑
feature_groups = {
    '수치 기본': numeric_base,
    '키워드': keyword_features,
    '파생 피처': derived_features,
    '통계 피처': stat_features,
    '범주형': category_features,
    '시간': time_features,
    'TF-IDF': text_features,
}

# 각 그룹을 제거한 실험 (5-Fold CV, 1회)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 전체 피처 baseline
baseline_maes = []
for train_idx, val_idx in kf.split(X):
    model = lgb.LGBMRegressor(
        n_estimators=500, learning_rate=0.03, max_depth=8,
        num_leaves=127, colsample_bytree=0.7,
        min_child_samples=20, subsample=0.8,
        reg_alpha=0.05, reg_lambda=0.5,
        random_state=42, verbose=-1, n_jobs=-1
    )
    model.fit(X[train_idx], y[train_idx],
              eval_set=[(X[val_idx], y[val_idx])],
              callbacks=[lgb.early_stopping(50, verbose=False)])
    pred = model.predict(X[val_idx])
    baseline_maes.append(mean_absolute_error(y[val_idx], pred))

baseline_mae = np.mean(baseline_maes)
print(f"  Baseline (전체 피처): MAE = {baseline_mae:.4f}")
print()

ablation_results = []
for group_name, group_features in feature_groups.items():
    # 해당 그룹 피처 제거
    remaining = [f for f in feature_cols if f not in group_features]
    remaining_idx = [feature_cols.index(f) for f in remaining if f in feature_cols]

    if len(remaining_idx) == 0:
        continue

    X_ablation = X[:, remaining_idx]

    group_maes = []
    for train_idx, val_idx in kf.split(X_ablation):
        model = lgb.LGBMRegressor(
            n_estimators=1500, learning_rate=0.01, max_depth=10,
            num_leaves=200, colsample_bytree=0.7,
            min_child_samples=20, subsample=0.8,
            reg_alpha=0.05, reg_lambda=0.5,
            random_state=42, verbose=-1, n_jobs=-1
        )
        model.fit(X_ablation[train_idx], y[train_idx],
                  eval_set=[(X_ablation[val_idx], y[val_idx])],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        pred = model.predict(X_ablation[val_idx])
        group_maes.append(mean_absolute_error(y[val_idx], pred))

    ablation_mae = np.mean(group_maes)
    impact = ablation_mae - baseline_mae
    impact_pct = impact / baseline_mae * 100

    ablation_results.append({
        'Feature Group': group_name,
        'Features': len([f for f in group_features if f in feature_cols]),
        'MAE_without': round(ablation_mae, 4),
        'MAE_impact': round(impact, 4),
        'Impact_%': round(impact_pct, 2),
    })
    print(f"  -{group_name} 제거: MAE = {ablation_mae:.4f} (영향: +{impact:.4f}, +{impact_pct:.1f}%)")

df_ablation = pd.DataFrame(ablation_results).sort_values('MAE_impact', ascending=False)
print()
print("[Ablation Study 결과 (영향 큰 순)]")
print(df_ablation.to_string(index=False))
print()

# ============================================================
# 7. 시각화
# ============================================================
print("=" * 60)
print("7. 시각화")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('BID-AI 논문급 실험 결과', fontsize=16, fontweight='bold')

# (1) 모델별 MAE 비교 (mean ± std)
ax = axes[0, 0]
model_names = df_summary['Model'].tolist()
mae_means = df_summary['MAE_mean'].values
mae_stds = df_summary['MAE_std'].values
colors_model = ['#F87171', '#FB923C', '#38BDF8', '#818CF8', '#34D399']
colors_model = colors_model[:len(model_names)]

bars = ax.barh(range(len(model_names)), mae_means, xerr=mae_stds,
               color=colors_model, alpha=0.8, capsize=5, ecolor='gray')
ax.set_yticks(range(len(model_names)))
ax.set_yticklabels(model_names, fontsize=11)
ax.set_xlabel('MAE (mean ± std, 3×10-Fold CV)')
ax.set_title('모델별 MAE 비교', fontweight='bold', fontsize=14)
ax.invert_yaxis()
for i, (m, s) in enumerate(zip(mae_means, mae_stds)):
    ax.text(m + s + 0.02, i, f'{m:.3f}±{s:.3f}', va='center', fontsize=10, fontweight='bold')

# (2) 모델별 R² 비교
ax = axes[0, 1]
r2_means = df_summary['R2_mean'].values
r2_stds = df_summary['R2_std'].values
bars = ax.barh(range(len(model_names)), r2_means, xerr=r2_stds,
               color=colors_model, alpha=0.8, capsize=5, ecolor='gray')
ax.set_yticks(range(len(model_names)))
ax.set_yticklabels(model_names, fontsize=11)
ax.set_xlabel('R² (mean ± std, 3×10-Fold CV)')
ax.set_title('모델별 R² 비교', fontweight='bold', fontsize=14)
ax.invert_yaxis()
for i, (m, s) in enumerate(zip(r2_means, r2_stds)):
    ax.text(m + s + 0.005, i, f'{m:.3f}±{s:.3f}', va='center', fontsize=10, fontweight='bold')

# (3) Ablation Study
ax = axes[1, 0]
abl_names = df_ablation['Feature Group'].tolist()
abl_impacts = df_ablation['Impact_%'].values
colors_abl = ['#34D399' if v > 0 else '#94A3B8' for v in abl_impacts]
bars = ax.barh(range(len(abl_names)), abl_impacts, color=colors_abl, alpha=0.8)
ax.set_yticks(range(len(abl_names)))
ax.set_yticklabels(abl_names, fontsize=11)
ax.set_xlabel('MAE 증가율 (%, 높을수록 중요)')
ax.set_title('Ablation Study: 피처 그룹별 기여도', fontweight='bold', fontsize=14)
ax.invert_yaxis()
ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
for i, v in enumerate(abl_impacts):
    ax.text(v + 0.1, i, f'+{v:.1f}%', va='center', fontsize=10, fontweight='bold')

# (4) CV Fold별 MAE 분포 (Box plot)
ax = axes[1, 1]
box_data = [np.array(cv_results[name]['MAE']) for name in df_summary['Model']]
bp = ax.boxplot(box_data, labels=df_summary['Model'].tolist(),
                patch_artist=True, vert=True)
for patch, color in zip(bp['boxes'], colors_model):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_ylabel('MAE')
ax.set_title('CV Fold별 MAE 분포 (30 folds)', fontweight='bold', fontsize=14)
ax.tick_params(axis='x', labelsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'paper_experiment_results.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  → paper_experiment_results.png 저장")

# ============================================================
# 8. 저장
# ============================================================
print()
print("=" * 60)
print("8. 결과 저장")
print("=" * 60)

df_summary.to_csv(os.path.join(OUTPUT_DIR, 'model_comparison.csv'), index=False, encoding='utf-8-sig')
print("  → model_comparison.csv")

df_ttest.to_csv(os.path.join(OUTPUT_DIR, 'ttest_results.csv'), index=False, encoding='utf-8-sig')
print("  → ttest_results.csv")

df_ablation.to_csv(os.path.join(OUTPUT_DIR, 'ablation_study.csv'), index=False, encoding='utf-8-sig')
print("  → ablation_study.csv")

# CV 상세 결과 저장
cv_detail = {}
for name in models:
    for metric in ['MAE', 'RMSE', 'R2']:
        cv_detail[f'{name}_{metric}'] = cv_results[name][metric]
pd.DataFrame(cv_detail).to_csv(os.path.join(OUTPUT_DIR, 'cv_detail_results.csv'),
                                index=False, encoding='utf-8-sig')
print("  → cv_detail_results.csv")

print()
print("=" * 60)
print("논문급 실험 완료 요약")
print("=" * 60)
print(f"  모델 수: {len(models)}개")
print(f"  CV: {n_repeats}×{n_splits}-Fold ({total_folds} folds)")
print(f"  ★ 최적: {best_model} (MAE {df_summary.iloc[0]['MAE_mean']:.4f} ± {df_summary.iloc[0]['MAE_std']:.4f})")
print(f"  Ablation: {len(ablation_results)}개 피처 그룹 분석")
print(f"  통계 검증: {len(ttest_results)}개 Paired t-test")
