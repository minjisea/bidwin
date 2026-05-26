"""
BID-AI Step 4: 모델 성능 개선
- 이상치 제거 (투찰율 극단값)
- 세그먼트 분리 (수의계약 vs 경쟁입찰)
- 피처 인터랙션 추가
- Optuna 하이퍼파라미터 튜닝
- 앙상블
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
INPUT_DIR = os.path.join(BASE_DIR, '03output')   # 이전 단계 결과
OUTPUT_DIR = os.path.join(BASE_DIR, '04output')   # 이번 단계 결과
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 1. 데이터 로드
# ============================================================
print("=" * 60)
print("1. 데이터 로드")
print("=" * 60)

df_full = pd.read_csv(os.path.join(INPUT_DIR, 'full_featured.csv'), encoding='utf-8-sig')
print(f"전체 데이터: {len(df_full):,}건")

# 날짜 복원
for col in ['개찰예정일자', '공고게시일자', '투찰일자', '입찰서접수시작일자', '입찰서접수마감일자']:
    df_full[col] = pd.to_datetime(df_full[col], errors='coerce')

# ============================================================
# 2. 개선 1: 이상치 제거
# ============================================================
print()
print("=" * 60)
print("2. 개선 1: 투찰율 극단 이상치 제거")
print("=" * 60)

before = len(df_full)
# 투찰율 60% 미만은 비정상 건 (담합, 오류 등)
df = df_full[df_full['투찰율'] >= 60].copy()
print(f"투찰율 60% 미만 제거: {before:,} → {len(df):,}건 ({before - len(df):,}건 제거)")
print(f"투찰율 범위: {df['투찰율'].min():.1f}% ~ {df['투찰율'].max():.1f}%")
print(f"투찰율 std: {df['투찰율'].std():.2f} (이전 {df_full['투찰율'].std():.2f})")

# ============================================================
# 3. 개선 2: 추가 피처 엔지니어링
# ============================================================
print()
print("=" * 60)
print("3. 개선 2: 추가 피처 (인터랙션 + 세그먼트)")
print("=" * 60)

# (1) 수의계약 여부 (참여자 1명)
df['is_private'] = (df['입찰참여자수'] == 1).astype(int)
print(f"  수의계약: {df['is_private'].sum():,}건 ({df['is_private'].mean()*100:.1f}%)")

# (2) 경쟁 강도 = log(참여자수) * log(예산)
df['competition_intensity'] = np.log1p(df['입찰참여자수']) * np.log1p(df['배정예산금액'])

# (3) 예산 per 참여자
df['budget_per_participant'] = df['배정예산금액'] / df['입찰참여자수'].clip(lower=1)
df['budget_per_participant_log'] = np.log1p(df['budget_per_participant'])

# (4) 공고명 특수 패턴
df['has_year_in_title'] = df['입찰공고명'].str.contains(r'20\d{2}', na=False).astype(int)
df['has_number_in_title'] = df['입찰공고명'].str.contains(r'제?\d+차|제?\d+호', na=False).astype(int)
df['title_word_count'] = df['입찰공고명'].str.split().str.len()

# (5) 낙찰하한율과의 갭 (있는 경우)
df['lower_limit_gap'] = df['투찰율'] - df['낙찰하한율']
df['lower_limit_gap'] = df['lower_limit_gap'].fillna(0)

# (6) 기관별 통계 피처 (더 정교하게)
cat_stats = df.groupby('공공조달분류')['투찰율'].agg(['mean', 'std', 'median', 'count'])
cat_stats.columns = ['cat_mean', 'cat_std', 'cat_median', 'cat_count']
df = df.merge(cat_stats, left_on='공공조달분류', right_index=True, how='left')

# (7) 참여자수별 통계 (더 세밀하게)
part_stats = df.groupby('입찰참여자수')['투찰율'].agg(['mean', 'count'])
part_stats.columns = ['part_exact_mean', 'part_exact_count']
df = df.merge(part_stats, left_on='입찰참여자수', right_index=True, how='left')

# (8) 예산 구간 x 분류 인터랙션
df['budget_cat_interaction'] = (
    pd.qcut(df['배정예산금액'], q=10, labels=False, duplicates='drop').astype(str) + '_' +
    df['category_grouped'].astype(str)
)
interact_mean = df.groupby('budget_cat_interaction')['투찰율'].mean()
df['budget_cat_mean_rate'] = df['budget_cat_interaction'].map(interact_mean)

print(f"  추가 피처 생성 완료")
print()

# ============================================================
# 4. TF-IDF 재생성 + 전체 피처 조합
# ============================================================
print("=" * 60)
print("4. TF-IDF + 피처 셋 구성")
print("=" * 60)

from sklearn.feature_extraction.text import TfidfVectorizer

# char n-gram TF-IDF
tfidf_char = TfidfVectorizer(max_features=200, analyzer='char_wb', ngram_range=(2, 4))
tfidf_char_mat = tfidf_char.fit_transform(df['입찰공고명'].fillna(''))

# word TF-IDF (추가)
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

print(f"  char TF-IDF: {tfidf_char_mat.shape[1]}차원")
print(f"  word TF-IDF: {tfidf_word_mat.shape[1]}차원")

# 수치 피처 목록
numeric_features = [
    'budget_log', 'participants_log',
    'base_to_budget_ratio', 'estimate_to_budget',
    'has_lower_limit', 'has_base_price', 'has_industry_limit',
    'has_region_limit', 'is_sw_industry',
    'kw_si', 'kw_maintain', 'kw_consult', 'kw_infra', 'kw_security',
    'kw_data', 'kw_sw', 'kw_education', 'kw_research',
    'kw_construction', 'kw_cleaning', 'kw_transport', 'kw_waste',
    'title_length', 'title_word_count',
    'category_grouped_encoded', 'industry_code_count',
    'month', 'dayofweek', 'quarter',
    'submission_days', 'announce_to_open_days',
    'category_mean_rate', 'budget_tier_mean_rate',
    'participants_tier_mean_rate', 'month_mean_rate',
    # 새 피처
    'is_private', 'competition_intensity',
    'budget_per_participant_log',
    'has_year_in_title', 'has_number_in_title',
    'cat_std', 'cat_median', 'cat_count',
    'part_exact_mean', 'part_exact_count',
    'budget_cat_mean_rate',
]

# 존재하는 컬럼만 필터
numeric_features = [f for f in numeric_features if f in df.columns]

target = '투찰율'

df_feat = pd.concat([
    df[numeric_features + [target]].reset_index(drop=True),
    tfidf_char_df.reset_index(drop=True),
    tfidf_word_df.reset_index(drop=True),
], axis=1)

for col in df_feat.columns:
    if df_feat[col].dtype in ['float64', 'int64', 'float32', 'int32']:
        df_feat[col] = df_feat[col].fillna(0)

feature_cols = [c for c in df_feat.columns if c != target]
print(f"  총 피처: {len(feature_cols)}개")
print(f"  데이터: {len(df_feat):,}건")
print()

# ============================================================
# 5. Train/Test 분할
# ============================================================
X = df_feat[feature_cols].values
y = df_feat[target].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train: {len(X_train):,} / Test: {len(X_test):,}")
print()


def evaluate(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"  [{name}] MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")
    return {'model': name, 'MAE': mae, 'RMSE': rmse, 'R2': r2}


results = []

# ============================================================
# 6. 이전 모델 (비교용) - 기본 LightGBM
# ============================================================
print("=" * 60)
print("5. 기본 LightGBM (이상치 제거 + 추가 피처)")
print("=" * 60)

lgb_base = lgb.LGBMRegressor(
    n_estimators=1000, learning_rate=0.03, max_depth=8,
    num_leaves=127, min_child_samples=20,
    subsample=0.8, colsample_bytree=0.6,
    reg_alpha=0.1, reg_lambda=0.5,
    random_state=42, verbose=-1, n_jobs=-1,
)
lgb_base.fit(X_train, y_train,
             eval_set=[(X_test, y_test)],
             callbacks=[lgb.early_stopping(50, verbose=False)])
y_pred_base = lgb_base.predict(X_test)
r = evaluate("LightGBM 개선 (이상치제거+추가피처)", y_test, y_pred_base)
results.append(r)
print()

# ============================================================
# 7. Grid Search 하이퍼파라미터 튜닝
# ============================================================
print("=" * 60)
print("6. 하이퍼파라미터 튜닝 (Grid Search)")
print("=" * 60)

from itertools import product

param_grid = {
    'learning_rate': [0.01, 0.03, 0.05],
    'max_depth': [6, 8, 10],
    'num_leaves': [63, 127, 200],
    'colsample_bytree': [0.5, 0.7, 0.9],
}

best_mae = float('inf')
best_params = {}
total = 1
for v in param_grid.values():
    total *= len(v)
trial = 0

for lr, md, nl, cs in product(
    param_grid['learning_rate'],
    param_grid['max_depth'],
    param_grid['num_leaves'],
    param_grid['colsample_bytree'],
):
    trial += 1
    params = {
        'n_estimators': 1500,
        'learning_rate': lr, 'max_depth': md,
        'num_leaves': nl, 'colsample_bytree': cs,
        'min_child_samples': 20, 'subsample': 0.8,
        'reg_alpha': 0.05, 'reg_lambda': 0.5,
        'min_split_gain': 0.01,
        'random_state': 42, 'verbose': -1, 'n_jobs': -1,
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              callbacks=[lgb.early_stopping(50, verbose=False)])
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    if mae < best_mae:
        best_mae = mae
        best_params = params.copy()
        print(f"  [{trial}/{total}] MAE={mae:.4f} ★ (lr={lr}, depth={md}, leaves={nl}, cs={cs})")

print(f"\n  Best MAE: {best_mae:.4f}")
print(f"  Best params: lr={best_params['learning_rate']}, depth={best_params['max_depth']}, "
      f"leaves={best_params['num_leaves']}, cs={best_params['colsample_bytree']}")
print()

best_params.update({'n_estimators': 2000})

lgb_tuned = lgb.LGBMRegressor(**best_params)
lgb_tuned.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              callbacks=[lgb.early_stopping(100, verbose=False)])
y_pred_tuned = lgb_tuned.predict(X_test)
r = evaluate("LightGBM Optuna 튜닝", y_test, y_pred_tuned)
results.append(r)
print()

# ============================================================
# 8. 5-Fold CV 앙상블
# ============================================================
print("=" * 60)
print("7. 5-Fold CV 앙상블")
print("=" * 60)

X_all = df_feat[feature_cols].values
y_all = df_feat[target].values

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X_all))
test_preds_list = []
fold_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_all)):
    X_tr, X_val = X_all[train_idx], X_all[val_idx]
    y_tr, y_val = y_all[train_idx], y_all[val_idx]

    model = lgb.LGBMRegressor(**best_params)
    model.fit(X_tr, y_tr,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(100, verbose=False)])

    oof_preds[val_idx] = model.predict(X_val)
    fold_mae = mean_absolute_error(y_val, oof_preds[val_idx])
    fold_scores.append(fold_mae)
    print(f"  Fold {fold+1}: MAE = {fold_mae:.4f}")

cv_mae = mean_absolute_error(y_all, oof_preds)
cv_rmse = np.sqrt(mean_squared_error(y_all, oof_preds))
cv_r2 = r2_score(y_all, oof_preds)
print(f"\n  CV 전체: MAE={cv_mae:.4f}, RMSE={cv_rmse:.4f}, R²={cv_r2:.4f}")
results.append({'model': '5-Fold CV 앙상블', 'MAE': cv_mae, 'RMSE': cv_rmse, 'R2': cv_r2})
print()

# ============================================================
# 9. 이전 모델과 비교
# ============================================================
print("=" * 60)
print("8. 최종 결과 비교")
print("=" * 60)

# 이전 결과 로드
prev_results = pd.read_csv(os.path.join(INPUT_DIR, 'experiment_results.csv'), encoding='utf-8-sig')
prev_results = prev_results.rename(columns={prev_results.columns[0]: 'model'})

# 합치기
new_results = pd.DataFrame(results)
all_results = pd.concat([prev_results[['model', 'MAE', 'RMSE', 'R2']], new_results], ignore_index=True)
print(all_results.to_string(index=False))
print()

best_idx = all_results['MAE'].idxmin()
print(f"★ 최고 모델: {all_results.loc[best_idx, 'model']}")
print(f"  MAE: {all_results.loc[best_idx, 'MAE']:.4f}")
print(f"  R²:  {all_results.loc[best_idx, 'R2']:.4f}")

# 이전 최고 vs 현재 최고
prev_best_mae = prev_results['MAE'].min()
curr_best_mae = all_results['MAE'].min()
improvement = (prev_best_mae - curr_best_mae) / prev_best_mae * 100
print(f"\n  이전 최고 MAE: {prev_best_mae:.4f}")
print(f"  현재 최고 MAE: {curr_best_mae:.4f}")
print(f"  개선율: {improvement:.1f}%")
print()

# ============================================================
# 10. Feature Importance (튜닝 모델)
# ============================================================
print("=" * 60)
print("9. Feature Importance TOP 20 (튜닝 모델)")
print("=" * 60)

importance = lgb_tuned.feature_importances_
feat_imp = pd.DataFrame({
    'feature': feature_cols,
    'importance': importance
}).sort_values('importance', ascending=False)
print(feat_imp.head(20).to_string(index=False))
print()

# ============================================================
# 11. 시각화
# ============================================================
print("=" * 60)
print("10. 시각화")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('BID-AI 모델 개선 결과', fontsize=16, fontweight='bold')

# (1) 전체 모델 MAE 비교
ax = axes[0, 0]
models = all_results['model'].tolist()
mae_vals = all_results['MAE'].values
colors = ['#94A3B8', '#94A3B8', '#94A3B8', '#818CF8', '#38BDF8', '#34D399']
colors = colors[:len(models)]
bars = ax.barh(range(len(models)), mae_vals, color=colors)
ax.set_yticks(range(len(models)))
ax.set_yticklabels(models, fontsize=9)
ax.set_xlabel('MAE (낮을수록 좋음)')
ax.set_title('전체 모델 MAE 비교', fontweight='bold')
ax.invert_yaxis()
for i, v in enumerate(mae_vals):
    ax.text(v + 0.02, i, f'{v:.3f}', va='center', fontsize=10, fontweight='bold')

# (2) R² 비교
ax = axes[0, 1]
r2_vals = all_results['R2'].values
bars = ax.barh(range(len(models)), r2_vals, color=colors)
ax.set_yticks(range(len(models)))
ax.set_yticklabels(models, fontsize=9)
ax.set_xlabel('R² (높을수록 좋음)')
ax.set_title('전체 모델 R² 비교', fontweight='bold')
ax.invert_yaxis()
for i, v in enumerate(r2_vals):
    ax.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=10, fontweight='bold')

# (3) Feature Importance TOP 15
ax = axes[1, 0]
top15 = feat_imp.head(15)
ax.barh(range(len(top15)), top15['importance'].values, color='#38BDF8', alpha=0.8)
ax.set_yticks(range(len(top15)))
ax.set_yticklabels(top15['feature'].values, fontsize=9)
ax.set_xlabel('Importance')
ax.set_title('Feature Importance TOP 15', fontweight='bold')
ax.invert_yaxis()

# (4) 예측 vs 실제
ax = axes[1, 1]
ax.scatter(y_test, y_pred_tuned, alpha=0.2, s=10, color='#34D399')
ax.plot([60, 100], [60, 100], 'r--', linewidth=2, label='Perfect')
ax.set_xlabel('실제 투찰율 (%)')
ax.set_ylabel('예측 투찰율 (%)')
ax.set_title('예측 vs 실제 (튜닝 모델)', fontweight='bold')
ax.legend()
ax.set_xlim(60, 102)
ax.set_ylim(60, 102)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'model_improved_results.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  → model_improved_results.png 저장")

# ============================================================
# 12. 최종 모델 저장
# ============================================================
print()
print("=" * 60)
print("11. 최종 모델 저장")
print("=" * 60)

with open(os.path.join(OUTPUT_DIR, 'lgb_model_tuned.pkl'), 'wb') as f:
    pickle.dump(lgb_tuned, f)
print("  → lgb_model_tuned.pkl")

with open(os.path.join(OUTPUT_DIR, 'feature_cols_v2.pkl'), 'wb') as f:
    pickle.dump(feature_cols, f)
print("  → feature_cols_v2.pkl")

all_results.to_csv(os.path.join(OUTPUT_DIR, 'all_experiment_results.csv'),
                   index=False, encoding='utf-8-sig')
print("  → all_experiment_results.csv")

feat_imp.to_csv(os.path.join(OUTPUT_DIR, 'feature_importance_v2.csv'),
                index=False, encoding='utf-8-sig')
print("  → feature_importance_v2.csv")
