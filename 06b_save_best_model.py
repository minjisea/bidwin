"""
06 실험 결과 기반 최종 모델 학습 & 저장
- 전체 데이터로 LightGBM 학습 (CV가 아닌 최종 배포용)
- 모델 + 피처 리스트 저장
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
import lightgbm as lgb
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
OUTPUT_DIR = os.path.join(BASE_DIR, '06output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 데이터 로드 & 전처리 (06과 동일)
df_full = pd.read_csv(os.path.join(BASE_DIR, '03output', 'full_featured.csv'), encoding='utf-8-sig')
for col in ['개찰예정일자', '공고게시일자', '투찰일자', '입찰서접수시작일자', '입찰서접수마감일자']:
    if col in df_full.columns:
        df_full[col] = pd.to_datetime(df_full[col], errors='coerce')

df = df_full[df_full['투찰율'] >= 60].copy()

# 추가 피처
df['is_private'] = (df['입찰참여자수'] == 1).astype(int)
df['competition_intensity'] = np.log1p(df['입찰참여자수']) * np.log1p(df['배정예산금액'])
df['budget_per_participant'] = df['배정예산금액'] / df['입찰참여자수'].clip(lower=1)
df['budget_per_participant_log'] = np.log1p(df['budget_per_participant'])
df['has_year_in_title'] = df['입찰공고명'].str.contains(r'20\d{2}', na=False).astype(int)
df['has_number_in_title'] = df['입찰공고명'].str.contains(r'제?\d+차|제?\d+호', na=False).astype(int)
df['title_word_count'] = df['입찰공고명'].str.split().str.len()
df['lower_limit_gap'] = (df['투찰율'] - df['낙찰하한율']).fillna(0)

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

all_numeric = [
    'budget_log', 'participants_log', 'base_to_budget_ratio', 'estimate_to_budget',
    'has_lower_limit', 'has_base_price', 'has_industry_limit', 'has_region_limit',
    'is_sw_industry',
    'kw_si', 'kw_maintain', 'kw_consult', 'kw_infra', 'kw_security',
    'kw_data', 'kw_sw', 'kw_education', 'kw_research',
    'kw_construction', 'kw_cleaning', 'kw_transport', 'kw_waste',
    'title_length', 'title_word_count', 'has_year_in_title', 'has_number_in_title',
    'is_private', 'competition_intensity', 'budget_per_participant_log', 'lower_limit_gap',
    'category_mean_rate', 'budget_tier_mean_rate', 'participants_tier_mean_rate',
    'month_mean_rate', 'cat_mean', 'cat_std', 'cat_median', 'cat_count',
    'part_exact_mean', 'part_exact_count', 'budget_cat_mean_rate',
    'category_grouped_encoded', 'industry_code_count',
    'month', 'dayofweek', 'quarter', 'submission_days', 'announce_to_open_days',
]
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

# Train/Test split으로 최종 성능 확인
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("최종 모델 학습 중...")
final_model = lgb.LGBMRegressor(
    n_estimators=500, learning_rate=0.03, max_depth=8,
    num_leaves=127, colsample_bytree=0.7,
    min_child_samples=20, subsample=0.8,
    reg_alpha=0.05, reg_lambda=0.5,
    random_state=42, verbose=-1, n_jobs=-1
)
final_model.fit(X_train, y_train,
                eval_set=[(X_test, y_test)],
                callbacks=[lgb.early_stopping(50, verbose=False)])

y_pred = final_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"  Test MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

# 전체 데이터 예측 (스코어링용)
full_pred = final_model.predict(X)
df['predicted_rate'] = np.nan
df.loc[df.index[df_feat.index], 'predicted_rate'] = full_pred
print(f"  전체 데이터 예측 완료: {len(full_pred):,}건")

# 저장
with open(os.path.join(OUTPUT_DIR, 'lgb_final_model.pkl'), 'wb') as f:
    pickle.dump(final_model, f)
print("  → lgb_final_model.pkl")

with open(os.path.join(OUTPUT_DIR, 'feature_cols_final.pkl'), 'wb') as f:
    pickle.dump(feature_cols, f)
print("  → feature_cols_final.pkl")

with open(os.path.join(OUTPUT_DIR, 'tfidf_char.pkl'), 'wb') as f:
    pickle.dump(tfidf_char, f)
with open(os.path.join(OUTPUT_DIR, 'tfidf_word.pkl'), 'wb') as f:
    pickle.dump(tfidf_word, f)
print("  → tfidf_char.pkl, tfidf_word.pkl")

# Feature Importance 저장
feat_imp = pd.DataFrame({
    'feature': feature_cols,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)
feat_imp.to_csv(os.path.join(OUTPUT_DIR, 'feature_importance_final.csv'),
                index=False, encoding='utf-8-sig')
print("  → feature_importance_final.csv")

# 전체 데이터에 예측값 포함해서 저장
df_full_out = df_full.copy()
df_full_out['predicted_rate'] = np.nan
pred_map = dict(zip(df.index, full_pred))
for idx, val in pred_map.items():
    if idx in df_full_out.index:
        df_full_out.loc[idx, 'predicted_rate'] = val
df_full_out.to_csv(os.path.join(OUTPUT_DIR, 'full_with_predictions.csv'),
                   index=False, encoding='utf-8-sig')
print(f"  → full_with_predictions.csv ({len(df_full_out):,}건)")

print("\n최종 모델 저장 완료!")
print(f"  MAE: {mae:.4f} / R²: {r2:.4f}")
