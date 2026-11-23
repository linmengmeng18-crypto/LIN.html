"""
config.py
쿠팡 배송기사 이탈 예측 및 리텐션 정책 분석 프로젝트 설정 파일
"""

import os

# --- 기본 경로 설정 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "P3_driver_attrition.csv")
# 만약 data 폴더에 없으면 루트 폴더 경로를 폴백으로 사용
if not os.path.exists(DATA_PATH):
    DATA_PATH = os.path.join(BASE_DIR, "P3_driver_attrition.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")

# --- 데이터 변수 정의 ---
ID_COL = "driver_id"
TARGET_COL = "attrition_6m"

NUMERICAL_COLS = [
    "age",
    "tenure_months",
    "avg_daily_deliveries",
    "avg_daily_work_hours",
    "weekly_days_off",
    "route_difficulty",
    "night_shift_ratio",
    "monthly_income_10k_krw",
    "incentive_ratio",
    "injury_count_12m",
    "commute_minutes",
    "manager_1on1_count_6m",
    "job_satisfaction"
]

CATEGORICAL_COLS = [
    "employment_type",
    "region"
]

# --- 모델링 설정 ---
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS_CV = 5

# --- What-if 시뮬레이션 설정 ---
# 현실적인 변수 상한/하한 적용
WHATIF_CONSTRAINTS = {
    "weekly_days_off_max": 4,      # 주당 휴무일 최대 4일로 제한
    "avg_daily_work_hours_min": 6.0, # 일평균 근무시간 최소 6시간으로 제한
    "manager_1on1_count_6m_max": 12 # 최근 6개월 면담 횟수 최대 12회로 제한
}

# --- 리텐션 정책 비용·효과 분석 가정값 (설정 가능) ---
# 주의: 본 수치는 가상 시뮬레이션을 위한 '가정값(Hypothetical Assumptions)'이며,
# 실제 쿠팡의 공식 재무/인사 데이터가 아닙니다.
COST_BENEFIT_CONFIG = {
    "enable_cost_analysis": True,  # False로 변경 시 '비용 데이터 미입력' 상태로 정성평가 수행
    "recruitment_cost_per_driver": 2_500_000,   # 신규 기사 1명 채용 비용(원)
    "training_cost_per_driver": 1_500_000,      # 신규 기사 1명 교육 및 온보딩 비용(원)
    "scenario_a_monthly_cost_per_driver": 350_000,  # 휴무 1일 확대 시 기사 1명당 월 대체 인력/인건비(원)
    "scenario_b_monthly_cost_per_driver": 200_000,  # 근무 1시간 단축 시 기사 1명당 월 운영비용(원)
    "scenario_c_cost_per_meeting": 50_000,          # 면담 1회당 관리자 인건비/시간 환산액(원)
    "evaluation_period_months": 6,                  # 분석 및 정책 적용 기간(6개월)
    "target_cohort": "top10_high_risk"              # 정책 적용 대상: 'top10_high_risk' (상위 10%) 또는 'all' (전체)
}
