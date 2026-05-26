"""
BID-AI Step 3: 모델 학습 및 실험
- Baseline: Linear Regression
- Model A: Ridge + TF-IDF
- Model B: LightGBM (Main)
- 피처 중요도 분석
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# ============================================================
# 1. 데이터 로드 & 분할
# ============================================================
print("=" * 60)
print("1. 데이터 로드 & Train/Test 분할")
print("=" * 60)

df = pd.read_csv(os.path.join(OUTPUT_DIR, 'features.csv'), encoding='utf-8-sig')
print(f"데이터: {len(df):,}건, {df.shape[1]}개 컬럼")

target = '투찰율'
feature_cols = [c for c in df.columns if c != target]

X = df[feature_cols].values
y = df[target].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train: {len(X_train):,}건 / Test: {len(X_test):,}건")
print(f"피처 수: {X.shape[1]}개")
print()


def evaluate(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²:   {r2:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    return {'model': name, 'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}


results = []

# ============================================================
# 2. 실험 1: Baseline - Linear Regression (수치 피처만)
# ============================================================
print("=" * 60)
print("2. 실험 1: Baseline - Linear Regression (수치 피처만)")
print("=" * 60)

# 수치 피처만 (TF-IDF 제외)
numeric_only = [c for c in feature_cols if not c.startswith('tfidf_')]
numeric_idx = [feature_cols.index(c) for c in numeric_only]

X_train_num = X_train[:, numeric_idx]
X_test_num = X_test[:, numeric_idx]

scaler = StandardScaler()
X_train_num_s = scaler.fit_transform(X_train_num)
X_test_num_s = scaler.transform(X_test_num)

lr = LinearRegression()
lr.fit(X_train_num_s, y_train)
y_pred_lr = lr.predict(X_test_num_s)
r = evaluate("Linear Regression (수치만)", y_test, y_pred_lr)
results.append(r)
print()

# ============================================================
# 3. 실험 2: Ridge + TF-IDF (전체 피처)
# ============================================================
print("=" * 60)
print("3. 실험 2: Ridge Regression (전체 피처)")
print("=" * 60)

scaler_full = StandardScaler()
X_train_s = scaler_full.fit_transform(X_train)
X_test_s = scaler_full.transform(X_test)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train_s, y_train)
y_pred_ridge = ridge.predict(X_test_s)
r = evaluate("Ridge (전체 피처)", y_test, y_pred_ridge)
results.append(r)
print()

# ============================================================
# 4. 실험 3: LightGBM (수치 피처만)
# ============================================================
print("=" * 60)
print("4. 실험 3: LightGBM (수치 피처만)")
print("=" * 60)

lgb_num = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=7,
    num_leaves=63,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    verbose=-1,
    n_jobs=-1,
)
lgb_num.fit(
    X_train_num, y_train,
    eval_set=[(X_test_num, y_test)],
    callbacks=[lgb.early_stopping(50, verbose=False)],
)
y_pred_lgb_num = lgb_num.predict(X_test_num)
r = evaluate("LightGBM (수치만)", y_test, y_pred_lgb_num)
results.append(r)
print(f"  Best iteration: {lgb_num.best_iteration_}")
print()

# ============================================================
# 5. 실험 4: LightGBM (전체 피처) - Main Model
# ============================================================
print("=" * 60)
print("5. 실험 4: LightGBM (전체 피처) ★ Main Model")
print("=" * 60)

lgb_full = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=8,
    num_leaves=127,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.6,
    reg_alpha=0.1,
    reg_lambda=0.5,
    random_state=42,
    verbose=-1,
    n_jobs=-1,
)
lgb_full.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[lgb.early_stopping(50, verbose=False)],
)
y_pred_lgb_full = lgb_full.predict(X_test)
r = evaluate("LightGBM (전체 피처)", y_test, y_pred_lgb_full)
results.append(r)
print(f"  Best iteration: {lgb_full.best_iteration_}")
print()

# ============================================================
# 6. 실험 결과 비교
# ============================================================
print("=" * 60)
print("6. 실험 결과 비교")
print("=" * 60)

results_df = pd.DataFrame(results)
results_df = results_df.set_index('model')
print(results_df.to_string())
print()

# 최고 모델
best = results_df['MAE'].idxmin()
print(f"★ 최고 성능 모델: {best}")
print(f"  MAE: {results_df.loc[best, 'MAE']:.4f}, R²: {results_df.loc[best, 'R2']:.4f}")
print()

# ============================================================
# 7. Feature Importance (LightGBM 전체 피처 모델)
# ============================================================
print("=" * 60)
print("7. Feature Importance (TOP 30)")
print("=" * 60)

importance = lgb_full.feature_importances_
feat_imp = pd.DataFrame({
    'feature': feature_cols,
    'importance': importance
}).sort_values('importance', ascending=False)

print(feat_imp.head(30).to_string(index=False))
print()

# ============================================================
# 8. 시각화
# ============================================================
print("=" * 60)
print("8. 시각화 생성")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('BID-AI 모델 실험 결과', fontsize=16, fontweight='bold')

# (1) 모델 성능 비교 (MAE)
ax = axes[0, 0]
models = results_df.index.tolist()
mae_vals = results_df['MAE'].values
colors = ['#94A3B8', '#818CF8', '#38BDF8', '#34D399']
bars = ax.barh(range(len(models)), mae_vals, color=colors)
ax.set_yticks(range(len(models)))
ax.set_yticklabels(models, fontsize=10)
ax.set_xlabel('MAE (낮을수록 좋음)')
ax.set_title('모델별 MAE 비교', fontweight='bold')
ax.invert_yaxis()
for i, v in enumerate(mae_vals):
    ax.text(v + 0.05, i, f'{v:.3f}', va='center', fontsize=11, fontweight='bold')

# (2) 모델 성능 비교 (R²)
ax = axes[0, 1]
r2_vals = results_df['R2'].values
bars = ax.barh(range(len(models)), r2_vals, color=colors)
ax.set_yticks(range(len(models)))
ax.set_yticklabels(models, fontsize=10)
ax.set_xlabel('R² (높을수록 좋음)')
ax.set_title('모델별 R² 비교', fontweight='bold')
ax.invert_yaxis()
for i, v in enumerate(r2_vals):
    ax.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=11, fontweight='bold')

# (3) Feature Importance TOP 20
ax = axes[1, 0]
top20 = feat_imp.head(20)
ax.barh(range(len(top20)), top20['importance'].values, color='#38BDF8', alpha=0.8)
ax.set_yticks(range(len(top20)))
ax.set_yticklabels(top20['feature'].values, fontsize=9)
ax.set_xlabel('Importance')
ax.set_title('Feature Importance TOP 20 (LightGBM)', fontweight='bold')
ax.invert_yaxis()

# (4) 예측 vs 실제 산점도 (LightGBM 전체)
ax = axes[1, 1]
ax.scatter(y_test, y_pred_lgb_full, alpha=0.15, s=10, color='#818CF8')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
        'r--', linewidth=2, label='Perfect Prediction')
ax.set_xlabel('실제 투찰율 (%)')
ax.set_ylabel('예측 투찰율 (%)')
ax.set_title('예측 vs 실제 (LightGBM 전체 피처)', fontweight='bold')
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'model_results.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  → model_results.png 저장 완료")

# 잔차 분석
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('LightGBM (전체 피처) 잔차 분석', fontsize=14, fontweight='bold')

residuals = y_test - y_pred_lgb_full

ax = axes[0]
ax.hist(residuals, bins=50, color='#38BDF8', edgecolor='white', alpha=0.8)
ax.axvline(0, color='red', linestyle='--')
ax.set_title('잔차 분포', fontweight='bold')
ax.set_xlabel('잔차 (실제 - 예측)')
ax.set_ylabel('건수')

ax = axes[1]
ax.scatter(y_pred_lgb_full, residuals, alpha=0.15, s=10, color='#818CF8')
ax.axhline(0, color='red', linestyle='--')
ax.set_title('예측값 vs 잔차', fontweight='bold')
ax.set_xlabel('예측 투찰율 (%)')
ax.set_ylabel('잔차')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'residual_analysis.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  → residual_analysis.png 저장 완료")
print()

# ============================================================
# 9. 모델 저장
# ============================================================
print("=" * 60)
print("9. 모델 저장")
print("=" * 60)

model_path = os.path.join(OUTPUT_DIR, 'lgb_model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(lgb_full, f)
print(f"  → lgb_model.pkl")

results_path = os.path.join(OUTPUT_DIR, 'experiment_results.csv')
results_df.to_csv(results_path, encoding='utf-8-sig')
print(f"  → experiment_results.csv")

feat_imp_path = os.path.join(OUTPUT_DIR, 'feature_importance.csv')
feat_imp.to_csv(feat_imp_path, index=False, encoding='utf-8-sig')
print(f"  → feature_importance.csv")

# 피처 목록 저장 (서빙용)
with open(os.path.join(OUTPUT_DIR, 'feature_cols.pkl'), 'wb') as f:
    pickle.dump(feature_cols, f)
print(f"  → feature_cols.pkl")

print()
print("=" * 60)
print("모델링 완료 요약")
print("=" * 60)
print()
print(results_df.to_string())
print()
print(f"★ Best Model: {best}")
print(f"  MAE={results_df.loc[best,'MAE']:.4f}, RMSE={results_df.loc[best,'RMSE']:.4f}, R²={results_df.loc[best,'R2']:.4f}")
