# BID-AI 설계문서

> 공공입찰 낙찰률 예측 및 AI 제안서 생성 시스템

---

## 1. AI 활용 기술스택 구분

### 구분 기준

| 구분 | 정의 |
|------|------|
| **AI 활용** | 외부 AI 서비스·모델을 호출하여 기능 구현 |
| **오픈소스 솔루션** | 검증된 오픈소스 라이브러리·프레임워크 활용 |
| **자체 개발 코드** | 도메인 로직·파이프라인을 팀이 직접 설계·구현 |

---

### AI 활용 영역

| 기술 | 제공사 | 활용 목적 | 적용 방식 |
|------|--------|---------|----------|
| **Claude API** (claude-sonnet-4 계열) | Anthropic | 기술제안서 초안 자동 생성 | REST API 호출, 프롬프트 엔지니어링 |
| **Claude Code** | Anthropic | 개발 보조 (코드 생성·리뷰·디버깅) | CLI 도구 (IDE 통합) |

**Claude API 활용 상세**

```
시스템 프롬프트: "당신은 공공입찰 기술제안서 전문 작성자입니다."
사용자 프롬프트 구조:
  - 공고명 / 수요기관 / 배정예산 / 업종코드 / 핵심 요구사항
  - 유사 낙찰 사례 (TF-IDF 유사도 기반 검색 결과)
  - 작성 요청 섹션: 사업이해 / 추진전략 / 기술제안 / 수행계획 / 차별화전략

프롬프트 엔지니어링 기법:
  - Few-shot Learning: 우수 제안서 구조를 예시로 제공
  - Chain-of-Thought: 분석 → 전략 수립 → 작성의 단계적 생성
  - 섹션별 분리 생성: 각 섹션을 독립적으로 구조화
```

**Claude Code 활용 상세**

- 코드 스캐폴딩: 전처리·피처 엔지니어링·모델링 파이프라인 초기 구조 생성
- 버그 디버깅: pandas 희소행렬 처리, Streamlit 캐시 최적화
- 코드 리뷰: 성능 병목 분석, 메모리 사용 최적화 제안
- 실험 자동화: 3×5-Fold CV 반복 코드, 결과 저장 로직

---

### 오픈소스 솔루션

| 기술 | 버전 | 라이선스 | 용도 |
|------|------|---------|------|
| Python | 3.10+ | PSF | 핵심 개발 언어 |
| **LightGBM** | 4.x | MIT | 낙찰률 예측 핵심 모델 (GBDT) |
| **scikit-learn** | 1.x | BSD | TF-IDF 벡터화, 전처리, 교차검증 |
| pandas | 2.x | BSD | 데이터 처리·분석 |
| numpy | 1.x | BSD | 수치 연산 |
| **Streamlit** | 1.x | Apache 2.0 | 웹 대시보드 UI 프레임워크 |
| matplotlib | 3.x | PSF | 데이터 시각화 |
| scipy | 1.x | BSD | 통계 검정 (t-test, Spearman) |
| optuna | 3.x | MIT | 하이퍼파라미터 베이지안 최적화 |
| catboost | 1.x | Apache 2.0 | 비교 실험용 (최종 미채택) |
| xgboost | 1.x | Apache 2.0 | 비교 실험용 (최종 미채택) |

---

### 자체 개발 코드

| 파일 | 역할 | 핵심 로직 |
|------|------|---------|
| `01_preprocessing_eda.py` | 데이터 전처리 파이프라인 | 결측치 처리, 이상치 IQR 제거, EDA 시각화 |
| `02_feature_engineering.py` | 피처 파생 엔지니어링 | 기관별/업종별 역대 낙찰률, 경쟁강도, 참여자당 예산 등 14종 |
| `03_modeling.py` | Baseline 모델링 | Linear Regression / Ridge 비교 실험 |
| `04_model_improve.py` | 모델 고도화 | LightGBM 하이퍼파라미터 Grid Search 튜닝 |
| `05_scoring.py` | 입찰 적합도 스코어링 | Fit Score 공식: `낙찰률×0.487 + 예산×0.268 + 경쟁강도×0.246` |
| `06_paper_experiments.py` | 다중 모델 비교 검증 | 3×5-Fold CV, Paired t-test, Ablation Study |
| `06b_save_best_model.py` | 최적 모델 저장 | 최종 모델 pkl 직렬화 |
| `app.py` | Streamlit 대시보드 | Dashboard / Prediction / Proposal / Analysis 4페이지 |

---

## 2. 배포 구조 및 방법

### 시스템 배포 아키텍처

```
┌─────────────────────────────────────────────────────┐
│                   사용자 환경                         │
│                 웹 브라우저 (Chrome/Edge)             │
└───────────────────┬─────────────────────────────────┘
                    │ HTTP (localhost:8501 또는 공개 URL)
┌───────────────────▼─────────────────────────────────┐
│              Streamlit 서버 (app.py)                 │
│  ┌──────────┐  ┌──────────┐  ┌────────┐  ┌────────┐│
│  │Dashboard │  │Prediction│  │Proposal│  │Analysis││
│  └──────────┘  └──────────┘  └────────┘  └────────┘│
└──────┬──────────────────┬──────────────────┬────────┘
       │                  │                  │
┌──────▼──────┐  ┌────────▼──────┐  ┌───────▼───────┐
│ 데이터 파일  │  │ 모델 파일     │  │  Claude API   │
│05output/    │  │06output/      │  │  (Anthropic)  │
│scored_all.  │  │lgb_final_     │  │  (Proposal    │
│csv          │  │model.pkl 등   │  │   페이지 한정) │
└─────────────┘  └───────────────┘  └───────────────┘
```

### 배포 방법 3가지

---

#### 방법 1: 로컬 직접 실행 (기본)

**사전 조건**: Python 3.10+ 설치

```bash
# 1. 의존성 설치
pip install streamlit pandas numpy matplotlib scikit-learn lightgbm scipy

# 2. 앱 실행
cd C:\Users\ohdon\Downloads\bid
streamlit run app.py

# 3. 브라우저 접속
# http://localhost:8501
```

**특징**: 가장 간단, Python 설치 필요

---

#### 방법 2: Docker 컨테이너 실행 (Python 없이)

**사전 조건**: Docker Desktop 설치 (Python 불필요)

아래 내용으로 `Dockerfile` 생성:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir streamlit pandas numpy matplotlib scikit-learn lightgbm scipy
EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
# 이미지 빌드
docker build -t bid-ai .

# 컨테이너 실행
docker run -p 8501:8501 bid-ai

# 브라우저 접속
# http://localhost:8501
```

**특징**: Python·VSCode 없이 실행 가능, 환경 재현 완벽

---

#### 방법 3: Streamlit Community Cloud 배포 (★ 발표 시연 권장)

**사전 조건**: GitHub 계정, 인터넷 연결

```
1. GitHub 레포지토리 생성 및 코드 push
   git init && git add . && git commit -m "BID-AI 최초 배포"
   git remote add origin https://github.com/YOUR_ID/bid-ai.git
   git push -u origin main

2. requirements.txt 작성 (루트 디렉토리)
   streamlit
   pandas
   numpy
   matplotlib
   scikit-learn
   lightgbm
   scipy

3. streamlit.io/cloud 접속 → "New app" → GitHub 레포 연결
   Main file: app.py
   → 자동 빌드 및 배포

4. 공개 URL 생성 (예: https://bid-ai.streamlit.app)
   → 어느 PC·환경에서도 브라우저로 즉시 접속 가능
```

**특징**: Python/VSCode/Docker 모두 불필요, **발표 시연 최적**

---

#### 방법 비교

| 항목 | 로컬 실행 | Docker | Streamlit Cloud |
|------|---------|--------|-----------------|
| Python 필요 | ✅ 필요 | ❌ 불필요 | ❌ 불필요 |
| VSCode 필요 | ❌ 불필요 | ❌ 불필요 | ❌ 불필요 |
| 인터넷 필요 | ❌ | ❌ | ✅ |
| 설치 복잡도 | 낮음 | 중간 | 낮음 |
| 발표 시연 적합성 | 보통 | 보통 | **최적** |
| 공개 URL | ❌ | ❌ | ✅ |

---

## 3. 운영 체크리스트

### 3.1 데이터 업데이트 (월 1회 권장)

```
□ 조달청 공공데이터포털 접속
  URL: data.go.kr → "일반용역 입찰공고 결과내역" 검색
□ 최신 CSV 다운로드 (전월 데이터)
□ 기존 데이터와 중복 제거 후 병합
□ python 01_preprocessing_eda.py 실행
  → 01output/ 결과 확인
□ python 02_feature_engineering.py 실행
  → 02output/full_featured.csv 확인
□ python 05_scoring.py 실행
  → 05output/scored_all.csv 갱신 확인
```

### 3.2 모델 재학습 (분기 1회 또는 데이터 2만 건 추가 시)

```
□ 전처리·피처 생성 완료 확인 (3.1 완료 후)
□ python 03_modeling.py 실행
□ python 04_model_improve.py 실행
□ python 06_paper_experiments.py 실행
  → 06output/model_comparison.csv 확인
  → 목표: MAE < 2.5, R² > 0.5
□ python 06b_save_best_model.py 실행
  → 06output/lgb_final_model.pkl 저장 확인
□ app.py 재시작 → Prediction 페이지에서 예측 동작 확인
```

### 3.3 발표·시연 전 점검 체크리스트

**시연 30분 전**

```
□ streamlit run app.py 명령 정상 실행
□ http://localhost:8501 접속 확인
□ [Dashboard] 공고 목록 로드 확인 (총 공고 수 표시)
□ [Prediction] 예측 버튼 클릭 → 결과 표시 확인
□ [Proposal] 프롬프트 생성 버튼 클릭 → 텍스트 출력 확인
□ [Analysis] Feature Importance 차트 렌더링 확인
□ 한글 폰트 깨짐 없음 확인 (모든 차트)
□ SW/IT 필터 체크박스 동작 확인
```

**Streamlit Cloud 시연 시**

```
□ 공개 URL 접속 테스트 (다른 기기에서 확인)
□ 데이터 로드 속도 확인 (첫 로딩 20~30초 예상)
□ 인터넷 연결 상태 확인
```

### 3.4 이상 상황 대응

| 증상 | 원인 | 해결 방법 |
|------|------|----------|
| 한글 차트 깨짐 | matplotlib 폰트 미설정 | `app.py` 상단 `matplotlib.rcParams['font.family']='Malgun Gothic'` 확인 |
| "데이터 파일 없음" | CSV 경로 오류 | `05output/scored_all.csv` 또는 `03output/full_featured.csv` 존재 확인 |
| "모델 로드 실패" | pkl 파일 없음 | `python 06b_save_best_model.py` 실행 후 `06output/lgb_final_model.pkl` 확인 |
| 포트 충돌 | 8501 포트 사용 중 | `streamlit run app.py --server.port 8502` |
| Streamlit Cloud 빌드 실패 | requirements.txt 누락 | 루트에 requirements.txt 존재 확인 |
| 예측 결과 이상 | 모델 버전 불일치 | 모델 재학습 후 pkl 재저장 |

### 3.5 파일 구조 및 의존성

```
bid/
├── app.py                          # Streamlit 대시보드 (진입점)
├── 01_preprocessing_eda.py         # Step 1: 전처리
├── 02_feature_engineering.py       # Step 2: 피처 생성
├── 03_modeling.py                  # Step 3: Baseline 모델
├── 04_model_improve.py             # Step 4: 모델 고도화
├── 05_scoring.py                   # Step 5: 스코어링
├── 06_paper_experiments.py         # Step 6: 다중 모델 검증
├── 06b_save_best_model.py          # Step 6b: 최적 모델 저장
├── 251008~260407 일반용역 입찰공고 결과내역.csv  # 원본 데이터
├── 05output/
│   ├── scored_all.csv              # ★ app.py가 로드하는 주 데이터
│   └── scored_sw_it.csv
├── 06output/
│   ├── lgb_final_model.pkl         # ★ 최종 예측 모델
│   ├── feature_importance_final.csv
│   ├── model_comparison.csv
│   ├── tfidf_char.pkl
│   ├── tfidf_word.pkl
│   └── feature_cols_final.pkl
└── requirements.txt                # 의존성 목록 (배포 시 필요)
```

### 3.6 성능 모니터링 지표

| 지표 | 목표값 | 측정 방법 |
|------|--------|---------|
| 예측 MAE | < 2.5%p | `06output/model_comparison.csv` 확인 |
| 예측 R² | > 0.50 | 동일 |
| Fit Score Spearman | > 0.90 | `06output/scoring_method_comparison.csv` 확인 |
| 앱 로딩 시간 | < 5초 | 브라우저 개발자 도구 Network 탭 |
| A등급 공고 비율 | 전체의 약 2.5% | Dashboard 확인 |
