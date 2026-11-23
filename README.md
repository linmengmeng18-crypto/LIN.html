# 쿠팡 PBL 문제 3: 배송기사 이탈 예측 및 리텐션 정책 분석
## Coupang Delivery Driver Attrition Prediction & Retention Policy Machine Learning Project

본 프로젝트는 쿠팡 배송기사 데이터(`P3_driver_attrition.csv`, 3,000행 × 17열)를 기반으로 배송기사의 6개월 내 이탈(`attrition_6m`)을 조기에 예측하고, 통계 검정과 머신러닝 기법(SHAP, Permutation Importance)을 통해 핵심 이탈 요인을 규명하며, What-if 시뮬레이션 및 비용·효과(Cost-Benefit) 분석을 통해 데이터 기반의 리텐션(근무 유지) 정책을 수립하는 대학 PBL(Problem-Based Learning) 프로젝트입니다.

---

## 1. 프로젝트 폴더 구조 (Project Structure)

```text
E:\p3/
│
├─ data/
│   └─ P3_driver_attrition.csv      # 배송기사 원본 데이터셋 (3,000명)
│
├─ outputs/                         # 모든 분석 결과 및 산출물 자동 저장
│   ├─ 01_data_profile.csv          # 데이터 기초 통계 및 프로파일 요약표
│   ├─ 02_group_comparison.csv      # 재직자 vs 이탈자 통계적 집단 차이 검정표
│   ├─ 03_model_comparison.csv      # 4개 머신러닝 모델 성능 비교표
│   ├─ 04_feature_importance.csv    # 특성 중요도 (Permutation / SHAP) 순위표
│   ├─ 05_top10_high_risk_drivers.csv # 상위 10% 고위험군 기사 선별 및 위험 점수표
│   ├─ 06_whatif_summary.csv        # What-if 가상 시나리오별 예측 이탈률 변화표
│   ├─ 07_policy_cost_benefit.csv   # 리텐션 정책 비용·효과 및 추정 ROI 분석표
│   ├─ report_summary.md            # 실제 실행 결과 기반 9대 항목 요약 보고서
│   ├─ P3_배송기사_이탈예측_최종분석보고서.md # 대학 PBL 제출 및 발표용 종합 보고서
│   │
│   └─ figures/                     # 300 DPI 고해상도 시각화 그래프 PNG
│       ├─ fig01_attrition_distribution.png
│       ├─ fig02_tenure_distribution.png
│       ├─ fig03_work_hours_days_off_income.png
│       ├─ fig04_attrition_by_employment.png
│       ├─ fig05_attrition_by_region.png
│       ├─ fig06_night_shift_ratio.png
│       ├─ fig07_correlation_heatmap.png
│       ├─ fig08_roc_curves.png
│       ├─ fig09_pr_curves.png
│       ├─ fig10_confusion_matrix.png
│       ├─ fig11_shap_summary.png
│       ├─ fig12_permutation_importance.png
│       ├─ fig13_risk_score_distribution.png
│       ├─ fig14_whatif_simulation.png
│       └─ fig15_cost_benefit_roi.png
│
├─ config.py                        # 변수, 모델 파라미터, 비용 가정치 설정 파일
├─ main.py                          # 전체 분석 파이프라인 원클릭 실행 스크립트
├─ requirements.txt                 # 실행 의존성 라이브러리 목록
└─ README.md                        # 프로젝트 설명 및 매뉴얼
```

---

## 2. 환경 설정 및 실행 방법 (How to Run)

### 2.1 필수 패키지 설치
Python 3.10 이상 환경(Windows / macOS / Linux)에서 터미널을 열고 다음 명령어를 실행합니다:

```bash
pip install -r requirements.txt
```

### 2.2 전체 파이프라인 원클릭 실행
```bash
python main.py
```
실행이 완료되면 콘솔에 단계별 분석 과정이 출력되며, `outputs/` 폴더에 7개의 CSV 결과표, 15개의 고해상도 시각화 차트(PNG), 그리고 2개의 마크다운 보고서가 자동으로 생성됩니다.

---

## 3. 핵심 분석 방법론 및 파이프라인 (Methodology)

### 3.1 데이터 누수 방지 원칙 (No Data Leakage)
- 식별 번호(`driver_id`)는 모델 훈련 및 입력 변수에서 엄격히 제외되었습니다.
- 모든 스케일러(StandardScaler) 및 인코더(OneHotEncoder)는 `ColumnTransformer`와 `Pipeline`을 통해 **학습 세트(Train Set)에만 적합(fit)**되었으며, 테스트 세트는 전처리 누출 없이 엄격히 1회 최종 검증에만 사용되었습니다.
- 학습 세트 내에서 `StratifiedKFold(5-Fold)`를 적용하여 모델의 과적합 여부를 검증하였습니다.

### 3.2 불균형 데이터 평가 지표 (Imbalanced Classification)
- 전체 이탈률이 27.53%인 불균형 데이터셋 특성을 감안하여, 단순 정확도(Accuracy)를 배제하고 **PR-AUC (Precision-Recall Curve 하 면적)**, **Recall (재현율)**, **ROC-AUC**, **F1-Score**를 종합적인 평가 척도로 채택하였습니다.
- 비교 모델: Logistic Regression, Decision Tree, Random Forest, XGBoost

### 3.3 변수 중요도 분석 (Model Explainability)
- **Permutation Importance:** 독립 테스트 세트에서 각 특성을 무작위로 섞었을 때의 ROC-AUC 성능 감소 폭을 10회 반복 측정하여 계산.
- **SHAP (SHapley Additive exPlanations):** `TreeExplainer`를 활용하여 각 특성이 개별 기사의 이탈 예측 확률을 얼마나 증가시키거나 감소시키는지 방향성과 강도를 규명.

### 3.4 What-if 가상 시뮬레이션
- **시나리오 A:** 주당 휴무일 +1일 (최대 4일 제한)
- **시나리오 B:** 일평균 근무시간 -1시간 (최소 6시간 제한)
- **시나리오 C:** 최근 6개월 관리자 면담 +2회 (최대 12회 제한)
- **시나리오 D:** A + B + C 복합 적용
- *주의사항:* What-if 분석은 관측 데이터 기반의 예측 점수 변화를 시뮬레이션한 가상 시나리오이며, 실제 현장 도입 시 인과적 효과를 검증하기 위해서는 A/B 테스트 및 시범 캠프 운영이 필수적입니다.

### 3.5 비용·효과(Cost-Benefit) 분석 가정
- 본 데이터셋에는 실제 인사 채용/교육비가 포함되어 있지 않으므로, `config.py`에 명시된 가상 시뮬레이션 가정값(신규 기사 채용 250만원, 교육 150만원 등)을 바탕으로 추정치를 계산하였습니다.
- `config.py`의 `enable_cost_analysis = False`로 설정 시, 임의의 금액을 생성하지 않고 정성적 실행 난이도 평가 모드로 자동 전환됩니다.

---

## 4. 윤리 및 개인정보 보호 가이드 (Ethics & Privacy)
- 본 모델이 산출한 이탈 위험 점수(`attrition_risk_score`)는 기사 개인에 대한 불이익(인사 평가 하향, 배차 불이익, 권고사직 등)에 사용되어서는 절대 안 됩니다.
- 오직 캠프 매니저의 **1on1 고충 상담 대상자 추천, 과로 완화 조치, 심리/업무 지원 프로그램 제공 등 리텐션 및 근로환경 개선 목적**으로만 제한되어야 합니다.
