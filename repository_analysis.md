# 리파지토리 분석 보고서

## 리파지토리 목적

이 리파지토리는 **경제학/금융 분야의 데이터 분석 및 예측 모델링을 위한 교육 및 연구 목적**의 프로젝트입니다.

## 주요 구성요소

### 1. 교육용 자료
- **`intro.ipynb`**: Python과 Jupyter 노트북 기본 사용법 (한국어)
- **`pandas.ipynb`**: Pandas 라이브러리 활용법
- **`numpy.ipynb`**: NumPy 라이브러리 활용법 
- **`data_analysis.ipynb`**: 데이터 분석 실습

### 2. 인플레이션 분석 모듈 (`inflation/`)
- **목적**: 인플레이션 예측 모델링 (nowcasting & forecasting)
- **주요 기능**:
  - 빈티지 데이터를 활용한 시계열 예측
  - 다양한 머신러닝 모델 (RandomForest, ExtraTree, ElasticNet 등)
  - S3 클라우드 스토리지 연동
  - 예측 성능 평가 및 시각화
- **핵심 파일**: `inflation.py`, `train_models.ipynb`, `inf_forecasting.ipynb`

### 3. 조기경보시스템 (`ews/`)
- **목적**: 금융·외환 조기경보모형 (Early Warning System)
- **기반**: 한국은행(BOK) 이슈노트 No.2024-11 논문
- **주요 기능**:
  - CFPI(Composite Financial Pressure Index) 기반 위기 예측
  - Bidas 시계열 데이터 API 연동
  - 신호추출법(Signal Extraction) 모형 구현
  - 다양한 금융 지표를 활용한 조기경보
- **핵심 파일**: `ews.py`, `ews.ipynb`, `EWS_final.ipynb`

### 4. 데이터 저장소 (`data/`)
- 경기침체 데이터 (`recessions.csv`)
- 주식시장 데이터 (S&P 500)
- 부동산 관련 데이터 (`reb.xlsx`)
- DSR(Debt Service Ratio) 데이터
- 세계 투입산출표(WIOT) 데이터
- 기타 경제 데이터

## 활용 분야

1. **학술 연구**: 경제학, 금융학 연구를 위한 데이터 분석 도구
2. **정책 수립**: 중앙은행이나 정부기관의 경제 정책 지원
3. **교육**: 경제 데이터 분석 방법론 교육
4. **예측 모델링**: 인플레이션 및 금융 위기 예측

## 기술 스택

- **Python**: 주요 프로그래밍 언어
- **Jupyter Notebook**: 분석 및 시각화 환경
- **주요 라이브러리**: pandas, numpy, scikit-learn, matplotlib, statsmodels
- **클라우드**: AWS S3 스토리지 연동
- **데이터 소스**: Bidas API (한국은행 데이터 시스템)

## 결론

이 리파지토리는 한국의 경제 연구기관(특히 한국은행)에서 개발된 것으로 보이며, 경제 데이터 분석과 예측 모델링을 통한 정책 지원 및 연구 목적으로 활용되고 있는 것으로 판단됩니다.