# BID-WIN 시스템 설계 문서 (Architecture & Operations Specification)

본 문서는 **BID-WIN (Bidding Insight Data - We Innovation Now)** 프로젝트의 통합 시스템 설계 명세 및 운영 관리를 위한 가이드라인을 정의합니다. 본 시스템은 데이터 기반의 공공입찰 낙찰률 예측 및 생성형 AI 기반의 제안서 자동 생성 파이프라인을 End-to-End로 지원합니다.

---

## 1. AI 활용 기술 스택 분류 (Technology Stack Overview)

시스템의 안정성과 유지보수성을 극대화하기 위해 핵심 엔진(솔루션), 오픈소스 프레임워크, 그리고 비즈니스 로직이 담긴 자체 개발 코드를 명확히 분리하여 설계하였습니다.

### 1.1 AI 솔루션 및 클라우드 도구 (AI Solutions & Cloud Tools)
* **Claude 3.5 Sonnet API (Anthropic):** 고품질의 기술제안서 본문 및 초안 생성을 위한 핵심 대규모 언어 모델(LLM) 엔진입니다.
* **Claude Code / Anthropic API:** 개발 과정에서의 코드 리뷰, 최적화 보조 및 추론 파이프라인의 API 커넥터 역할을 수행합니다.

### 1.2 오픈소스 라이브러리 및 프레임워크 (Open-Source Ecosystem)
* **LightGBM 4.x:** 대규모 정형 데이터(조달청 입찰 데이터)의 고속 가공 및 낙찰가/낙찰 확률 정밀 예측을 위한 트리 기반 부스팅 모델입니다. CPU 환경 최적화를 통해 가성비를 극대화했습니다.
* **Streamlit Framework:** 파이프라인 전체 결과를 시각화하고 실무자가 웹 브라우저에서 시뮬레이션할 수 있도록 지원하는 통합 프론트엔드 UI 프레임워크입니다.
* **Pandas & Scikit-Learn:** 데이터 파생 피처 엔지니어링, 결측치 정제, TF-IDF 벡터화 알고리즘 처리를 담당합니다.

### 1.3 자체 개발 코드 및 산출물 (Custom Developed Source Code)
* `01_preprocessing.py`: 조달청 로우 데이터(CSV) 수집, 정제 및 핵심 파생 변수(경쟁강도, 참여자당 예산 등) 14개를 자동으로 생성하는 데이터 파이프라인 엔진입니다.
* `05_scoring.py`: LightGBM 모델의 변수 중요도(Feature Importance) 데이터를 100% 만점 기준으로 정규화(Normalization)하여 종합 점수를 산출하는 **Fit Score(입찰 적합도) 공식 연산 모듈**입니다.
* `app.py`: 대시보드 통계, 낙찰률 시뮬레이션, AI 제안서 생성 창, 데이터 분석 리포트 등 총 4개 페이지 인터페이스를 제어하는 메인 어플리케이션 코드입니다.

---

## 2. 배포 구조 및 방법 (Deployment Architecture)

운영 환경 및 확장성에 따라 세 가지 형태의 배포 파이프라인을 지원하도록 아키텍처가 설계되었습니다.

```
[Raw Data / CSV] ──> [자체 개발 전처리 엔진] ──> [LightGBM / Claude API]
                                                         │
                                                         ▼
[배포 환경 선택] ──> ① Local PC (런타임 테스팅)
                 ──> ② Docker 컨테이너 환경 (독립성 확보)
                 ──> ③ Streamlit Cloud / 사내 웹 서버 (실무 배포)
```

1.  **로컬 개발 환경 (Local Runtime Environment)**
    * **방법:** 로컬 PC에서 의존성 패키지 설치 후 파이썬 명령어로 직접 구동합니다. 코드 수정 및 신규 피처 실험 시 활용합니다.
    * **명령어:** ```bash
        pip install -r requirements.txt
        streamlit run app.py
        ```
2.  **컨테이너 배포 환경 (Containerized via Docker)**
    * **방법:** 운영체제나 파이썬 버전 독립성을 확보하기 위해 Docker 이미지를 빌드하여 가상화 배포합니다. 사내 인프라 이관 시 표준 규격으로 작동합니다.
    * **명령어:**
        ```bash
        docker build -t bid-win-app .
        docker run -p 8501:8501 bid-win-app
        ```
3.  **클라우드 스테이징 환경 (Cloud Deployment)**
    * **방법:** Git 저장소와 연동하여 코드 푸시 시 자동으로 빌드되는 구조를 갖추고 있으며, 외부 평가위원 및 사내 실무진이 URL을 통해 즉시 데모에 접근할 수 있도록 서빙합니다.

---

## 3. 시스템 운영 체크리스트 (Operations Checklist)

시스템이 지속적으로 높은 예측력을 유지하고, 실무에 안정적으로 서빙되기 위해 관리자가 주기적으로 확인해야 할 필수 체크리스트입니다.

### 3.1 데이터 수집 및 전처리 관리 (Data Pipeline)
* [ ] **조달청 데이터 스키마 일치 여부:** 정기적으로 다운로드하는 신규 입찰 데이터의 열(Column) 구조가 기존 `01_preprocessing.py` 파이프라인과 일치하는지 확인합니다.
* [ ] **결측치 및 이상치 필터링:** 입찰 금액이 0원이거나 참여 업체 수가 공란인 불량 레코드가 정상적으로 드롭 또는 대체 처리되는지 로그를 검증합니다.

### 3.2 모델 성능 및 재학습 가이드 (MLOps / Model Maintenance)
* [ ] **MAE(평균 절대 오차) 모니터링:** 신규 입찰 결과를 적용했을 때 예측 오차 지표인 MAE가 기준치(1.89)에서 크게 벗어나는지 분기별로 테스트합니다.
* [ ] **파생 변수 유효성 검증:** 시장 환경 변화로 인해 '경쟁강도' 파생 피처의 중요도가 하락하는지 변수 중요도(Feature Importance) 플롯을 주기적으로 확인합니다. 오차가 커질 경우 전체 4.5만 건 데이터를 기반으로 모델 재학습(`fit`)을 수행합니다.

### 3.3 API 인프라 및 트래픽 관리 (API & Infrastructure)
* [ ] **Claude API 가동 상태 및 잔여 크레딧:** 제안서 생성 모듈 호출 시 Anthropic 서버와의 통신 에러(Timeout) 혹은 API 토큰 한도 초과 오류가 발생하는지 모니터링합니다.
* [ ] **Rate Limit(트래픽 제한) 대응:** 실무자가 동시에 제안서 생성을 요청할 경우를 대비하여 프롬프트 큐(Queue) 시스템 혹은 요청 제한 예외 처리가 안정적으로 작동하는지 확인합니다.

### 3.4 보안 및 자원 관리 (Security & Resource Optimization)
* [ ] **API Key 보안:** Claude API 호출을 위한 Secret Key가 `app.py` 소스코드 내에 하드코딩되지 않고, 환경 변수(`.env`)나 클라우드 Secrets 매니저를 통해 안전하게 로드되는지 검증합니다.
* [ ] **메모리 오버플로우 방지:** Streamlit 대시보드 가동 중 수만 건의 대용량 데이터를 브라우저 캐시에 적재할 때, CPU 및 메모리 가용 자원이 고갈되지 않도록 가벼운 상태를 유지하는지 확인합니다.