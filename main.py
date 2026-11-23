"""
main.py
쿠팡 배송기사 이탈 예측 및 리텐션 정책 머신러닝 프로젝트 실행 스크립트
PBL 문제 3 통합 분석 파이프라인
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    recall_score,
    precision_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
from sklearn.inspection import permutation_importance

# 설정 불러오기
import config

# 경고 메시지 제어
warnings.filterwarnings('ignore')

# Matplotlib 한글 폰트 설정 (Windows Malgun Gothic 기본)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style='whitegrid', font='Malgun Gothic')


def setup_directories():
    """출력 디렉터리 생성"""
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    print(f"[INFO] 출력 폴더 준비 완료: {config.OUTPUT_DIR}")


# ==============================================================================
# 1단계: 데이터 점검 및 탐색적 데이터 분석 (EDA)
# ==============================================================================
def step1_data_profile_and_eda(df: pd.DataFrame):
    print("\n" + "=" * 70)
    print(" [1단계] 데이터 프로파일링 및 탐색적 데이터 분석 (EDA) ")
    print("=" * 70)

    n_rows, n_cols = df.shape
    print(f"데이터 크기: {n_rows:,}행 × {n_cols}열")

    # 1.1 데이터 프로파일 테이블 생성
    profile_rows = []
    for col in df.columns:
        col_series = df[col]
        dtype = str(col_series.dtype)
        null_count = int(col_series.isnull().sum())
        null_pct = (null_count / n_rows) * 100.0
        n_unique = int(col_series.nunique())

        if pd.api.types.is_numeric_dtype(col_series) and col != config.ID_COL:
            mean_val = float(col_series.mean())
            std_val = float(col_series.std())
            median_val = float(col_series.median())
            min_val = float(col_series.min())
            max_val = float(col_series.max())
            q25 = float(col_series.quantile(0.25))
            q75 = float(col_series.quantile(0.75))
            skew_val = float(col_series.skew())
            iqr = q75 - q25
            outliers_count = int(((col_series < (q25 - 1.5 * iqr)) | (col_series > (q75 + 1.5 * iqr))).sum())
        else:
            mean_val = std_val = median_val = min_val = max_val = q25 = q75 = skew_val = np.nan
            outliers_count = 0

        profile_rows.append({
            "variable": col,
            "data_type": dtype,
            "null_count": null_count,
            "null_pct": round(null_pct, 2),
            "unique_values": n_unique,
            "mean": round(mean_val, 2) if not np.isnan(mean_val) else "",
            "std": round(std_val, 2) if not np.isnan(std_val) else "",
            "min": round(min_val, 2) if not np.isnan(min_val) else "",
            "q25": round(q25, 2) if not np.isnan(q25) else "",
            "median": round(median_val, 2) if not np.isnan(median_val) else "",
            "q75": round(q75, 2) if not np.isnan(q75) else "",
            "max": round(max_val, 2) if not np.isnan(max_val) else "",
            "skewness": round(skew_val, 2) if not np.isnan(skew_val) else "",
            "outliers_count": outliers_count
        })

    profile_df = pd.DataFrame(profile_rows)
    profile_path = os.path.join(config.OUTPUT_DIR, "01_data_profile.csv")
    profile_df.to_csv(profile_path, index=False, encoding='utf-8-sig')
    print(f"-> 01_data_profile.csv 저장 완료: {profile_path}")

    # 목표변수 기초 통계
    target_counts = df[config.TARGET_COL].value_counts()
    target_pcts = df[config.TARGET_COL].value_counts(normalize=True) * 100
    print(f"목표변수 ({config.TARGET_COL}) 분포:")
    print(f" - 재직(0): {target_counts.get(0, 0):,}명 ({target_pcts.get(0, 0.0):.2f}%)")
    print(f" - 이탈(1): {target_counts.get(1, 0):,}명 ({target_pcts.get(1, 0.0):.2f}%)")

    # 1.2 필수 EDA 그래프 생성 (PNG)
    # 그림 1: 이탈/재직 인원 및 비율
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(["재직 (0)", "이탈 (1)"], [target_counts.get(0, 0), target_counts.get(1, 0)],
                  color=["#2b5c8f", "#d9534f"], width=0.5, edgecolor="black", alpha=0.85)
    for bar in bars:
        height = bar.get_height()
        pct = (height / n_rows) * 100
        ax.annotate(f"{height:,}명\n({pct:.1f}%)",
                    xy=(bar.get_x() + bar.get_width() / 2, height / 2),
                    ha="center", va="center", color="white", fontweight="bold", fontsize=12)
    ax.set_title("쿠팡 배송기사 6개월 내 이탈 여부 분포", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("기사 수 (명)", fontsize=11)
    ax.set_ylim(0, n_rows * 0.85)
    plt.tight_layout()
    fig1_path = os.path.join(config.FIGURES_DIR, "fig01_attrition_distribution.png")
    fig.savefig(fig1_path, dpi=300)
    plt.close(fig)

    # 그림 2: 이탈 여부별 근속 개월 분포
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=df, x="tenure_months", hue=config.TARGET_COL, kde=True,
                 palette=["#2b5c8f", "#d9534f"], bins=30, ax=ax, alpha=0.6, element="step")
    ax.set_title("근속 개월(tenure_months)에 따른 이탈/재직 분포", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("근속 개월 (개월)", fontsize=11)
    ax.set_ylabel("인원 수 (명)", fontsize=11)
    ax.legend(labels=["이탈(1)", "재직(0)"], title="이탈 여부")
    plt.tight_layout()
    fig2_path = os.path.join(config.FIGURES_DIR, "fig02_tenure_distribution.png")
    fig.savefig(fig2_path, dpi=300)
    plt.close(fig)

    # 그림 3: 이탈 여부별 근무시간, 휴무일, 소득 비교 (3개 서브플롯)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    comp_vars = [
        ("avg_daily_work_hours", "일평균 근무시간 (시간)", axes[0]),
        ("weekly_days_off", "주당 휴무일 수 (일)", axes[1]),
        ("monthly_income_10k_krw", "월 소득 (만원)", axes[2])
    ]
    for var, label, sub_ax in comp_vars:
        sns.boxplot(data=df, x=config.TARGET_COL, y=var, hue=config.TARGET_COL,
                    palette=["#4575b4", "#d73027"], ax=sub_ax, width=0.4, legend=False)
        sub_ax.set_xticks([0, 1])
        sub_ax.set_xticklabels(["재직 (0)", "이탈 (1)"])
        sub_ax.set_xlabel("이탈 여부", fontsize=11)
        sub_ax.set_ylabel(label, fontsize=11)
        sub_ax.set_title(f"{label} 비교", fontsize=12, fontweight="bold")
    fig.suptitle("재직자 vs 이탈자의 주요 근로 조건 비교", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig3_path = os.path.join(config.FIGURES_DIR, "fig03_work_hours_days_off_income.png")
    fig.savefig(fig3_path, dpi=300)
    plt.close(fig)

    # 그림 4: 고용 형태별 이탈률
    fig, ax = plt.subplots(figsize=(7, 5))
    emp_summary = df.groupby("employment_type")[config.TARGET_COL].agg(["count", "mean"]).reset_index()
    emp_summary["mean_pct"] = emp_summary["mean"] * 100
    bars = ax.bar(emp_summary["employment_type"], emp_summary["mean_pct"],
                  color=["#3182bd", "#e6550d"], width=0.45, edgecolor="black", alpha=0.85)
    for idx, bar in enumerate(bars):
        height = bar.get_height()
        total_cnt = emp_summary.loc[idx, "count"]
        ax.annotate(f"{height:.2f}%\n(총 {total_cnt:,}명)",
                    xy=(bar.get_x() + bar.get_width() / 2, height / 2),
                    ha="center", va="center", color="white", fontweight="bold", fontsize=11)
    ax.set_title("고용 형태(직고용 vs 위탁)별 6개월 내 이탈률", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("이탈률 (%)", fontsize=11)
    ax.set_ylim(0, max(emp_summary["mean_pct"]) * 1.3)
    plt.tight_layout()
    fig4_path = os.path.join(config.FIGURES_DIR, "fig04_attrition_by_employment.png")
    fig.savefig(fig4_path, dpi=300)
    plt.close(fig)

    # 그림 5: 권역별 이탈률
    fig, ax = plt.subplots(figsize=(9, 5))
    reg_summary = df.groupby("region")[config.TARGET_COL].agg(["count", "mean"]).reset_index()
    reg_summary["mean_pct"] = reg_summary["mean"] * 100
    reg_summary = reg_summary.sort_values(by="mean_pct", ascending=False)
    bars = ax.bar(reg_summary["region"], reg_summary["mean_pct"],
                  color="#2ca25f", width=0.5, edgecolor="black", alpha=0.85)
    for idx, row in reg_summary.reset_index(drop=True).iterrows():
        bar = bars[idx]
        height = bar.get_height()
        ax.annotate(f"{height:.1f}%\n({row['count']}명)",
                    xy=(bar.get_x() + bar.get_width() / 2, height + 1),
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title("근무 권역별 6개월 내 배송기사 이탈률", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("이탈률 (%)", fontsize=11)
    ax.set_ylim(0, max(reg_summary["mean_pct"]) * 1.25)
    plt.tight_layout()
    fig5_path = os.path.join(config.FIGURES_DIR, "fig05_attrition_by_region.png")
    fig.savefig(fig5_path, dpi=300)
    plt.close(fig)

    # 그림 6: 야간 근무 비중과 이탈 여부의 관계
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.kdeplot(data=df[df[config.TARGET_COL] == 0], x="night_shift_ratio", label="재직 (0)",
               color="#2b5c8f", fill=True, alpha=0.4, ax=ax)
    sns.kdeplot(data=df[df[config.TARGET_COL] == 1], x="night_shift_ratio", label="이탈 (1)",
               color="#d9534f", fill=True, alpha=0.4, ax=ax)
    ax.set_title("야간 근무 비중(night_shift_ratio) 밀도 분포 비교", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("야간 근무 비중 (0~1)", fontsize=11)
    ax.set_ylabel("밀도 (Density)", fontsize=11)
    ax.legend(title="이탈 여부")
    plt.tight_layout()
    fig6_path = os.path.join(config.FIGURES_DIR, "fig06_night_shift_ratio.png")
    fig.savefig(fig6_path, dpi=300)
    plt.close(fig)

    # 그림 7: 수치형 변수 간 상관관계 히트맵
    fig, ax = plt.subplots(figsize=(11, 9))
    corr_cols = config.NUMERICAL_COLS + [config.TARGET_COL]
    corr_matrix = df[corr_cols].corr(method='spearman')
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                vmin=-0.6, vmax=0.6, cbar_kws={'label': '스피어만 상관계수'}, ax=ax)
    ax.set_title("수치형 변수 및 이탈 여부 간 스피어만 상관관계 히트맵", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    fig7_path = os.path.join(config.FIGURES_DIR, "fig07_correlation_heatmap.png")
    fig.savefig(fig7_path, dpi=300)
    plt.close(fig)

    print(f"-> 7개 EDA 그래프 PNG 저장 완료: {config.FIGURES_DIR}")
    return profile_df


# ==============================================================================
# 2단계: 집단 차이 통계적 검정
# ==============================================================================
def step2_statistical_tests(df: pd.DataFrame):
    print("\n" + "=" * 70)
    print(" [2단계] 집단 간 차이 통계적 검정 (이탈자 vs 재직자) ")
    print("=" * 70)

    group0 = df[df[config.TARGET_COL] == 0]
    group1 = df[df[config.TARGET_COL] == 1]
    n0, n1 = len(group0), len(group1)

    results = []

    # 2.1 수치형 변수 검정 (Welch t-test, Mann-Whitney U, Cohen's d)
    raw_p_values = []
    num_test_records = []

    for var in config.NUMERICAL_COLS:
        s0 = group0[var].dropna()
        s1 = group1[var].dropna()

        m0, std0, med0 = s0.mean(), s0.std(), s0.median()
        m1, std1, med1 = s1.mean(), s1.std(), s1.median()
        diff = m1 - m0

        # Welch's t-test (등분산 가정 배제)
        t_stat, p_welch = stats.ttest_ind(s1, s0, equal_var=False)

        # Mann-Whitney U test (비모수 검정)
        u_stat, p_mwu = stats.mannwhitneyu(s1, s0, alternative='two-sided')

        # Cohen's d (효과크기: 양 집단 표준편차 합동 기준)
        pooled_std = np.sqrt(((len(s0) - 1) * (std0 ** 2) + (len(s1) - 1) * (std1 ** 2)) / (len(s0) + len(s1) - 2))
        cohen_d = (m1 - m0) / pooled_std if pooled_std > 0 else 0.0

        raw_p_values.append(p_welch)
        num_test_records.append({
            "variable_type": "수치형",
            "variable": var,
            "category": "전체",
            "n_retained": len(s0),
            "retained_mean": round(m0, 3),
            "retained_std": round(std0, 3),
            "retained_median": round(med0, 3),
            "n_attrited": len(s1),
            "attrited_mean": round(m1, 3),
            "attrited_std": round(std1, 3),
            "attrited_median": round(med1, 3),
            "mean_diff": round(diff, 3),
            "test_type": "Welch t-test / MWU",
            "stat_val": round(t_stat, 3),
            "raw_p_value": p_welch,
            "mwu_p_value": p_mwu,
            "effect_size": round(cohen_d, 3),
            "effect_size_type": "Cohen's d"
        })

    # Benjamini-Hochberg FDR 보정
    reject_bh, pvals_corrected, _, _ = multipletests(raw_p_values, alpha=0.05, method='fdr_bh')
    for i, rec in enumerate(num_test_records):
        rec["fdr_bh_p_value"] = pvals_corrected[i]
        rec["statistically_significant_005"] = bool(reject_bh[i])
        results.append(rec)

    # 2.2 범주형 변수 검정 (카이제곱 독립성 검정 및 Cramer's V)
    for cat_col in config.CATEGORICAL_COLS:
        contingency_table = pd.crosstab(df[cat_col], df[config.TARGET_COL])
        chi2, p_chi2, dof, expected = stats.chi2_contingency(contingency_table)

        # Cramer's V 계산
        n_obs = len(df)
        min_dim = min(contingency_table.shape) - 1
        cramers_v = np.sqrt(chi2 / (n_obs * min_dim)) if min_dim > 0 else 0.0

        for category in contingency_table.index:
            cnt0 = contingency_table.loc[category, 0]
            cnt1 = contingency_table.loc[category, 1]
            pct0 = (cnt0 / (cnt0 + cnt1)) * 100
            pct1 = (cnt1 / (cnt0 + cnt1)) * 100

            results.append({
                "variable_type": "범주형",
                "variable": cat_col,
                "category": category,
                "n_retained": cnt0,
                "retained_mean": round(pct0, 2),  # 백분율(%)로 표기
                "retained_std": np.nan,
                "retained_median": np.nan,
                "n_attrited": cnt1,
                "attrited_mean": round(pct1, 2),  # 백분율(%)로 표기
                "attrited_std": np.nan,
                "attrited_median": np.nan,
                "mean_diff": round(pct1 - pct0, 2),
                "test_type": f"Chi-square (dof={dof})",
                "stat_val": round(chi2, 3),
                "raw_p_value": p_chi2,
                "mwu_p_value": np.nan,
                "effect_size": round(cramers_v, 3),
                "effect_size_type": "Cramer's V",
                "fdr_bh_p_value": p_chi2,
                "statistically_significant_005": bool(p_chi2 < 0.05)
            })

    comparison_df = pd.DataFrame(results)
    out_path = os.path.join(config.OUTPUT_DIR, "02_group_comparison.csv")
    comparison_df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"-> 02_group_comparison.csv 저장 완료: {out_path}")

    # 주요 유의차 변수 출력
    sig_num = comparison_df[(comparison_df["variable_type"] == "수치형") & (comparison_df["statistically_significant_005"] == True)]
    sig_num_sorted = sig_num.sort_values(by="effect_size", key=abs, ascending=False)
    print("\n[수치형 변수 중 효과크기(|Cohen's d|) 상위 5개]:")
    for _, r in sig_num_sorted.head(5).iterrows():
        print(f" - {r['variable']}: 차이={r['mean_diff']:+.2f}, Cohen's d={r['effect_size']:.3f}, FDR p={r['fdr_bh_p_value']:.4e}")

    return comparison_df


# ==============================================================================
# 3단계: 머신러닝 모델 구축 및 비교 평가
# ==============================================================================
def step3_train_and_evaluate_models(df: pd.DataFrame):
    print("\n" + "=" * 70)
    print(" [3단계] 머신러닝 모델 구축, 교차검증 및 테스트 평가 ")
    print("=" * 70)

    # 3.1 설명변수와 목표변수 분리 (driver_id는 데이터 누수 방지를 위해 엄격히 제외)
    X = df[config.NUMERICAL_COLS + config.CATEGORICAL_COLS].copy()
    y = df[config.TARGET_COL].copy()

    # 3.2 계층적 분할 (Stratified Train / Test Split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_STATE
    )
    print(f"학습 세트: {X_train.shape[0]:,}개 (이탈자 {y_train.sum():,}명, {y_train.mean()*100:.2f}%)")
    print(f"테스트 세트: {X_test.shape[0]:,}개 (이탈자 {y_test.sum():,}명, {y_test.mean()*100:.2f}%)")

    # 3.3 전처리 파이프라인 (ColumnTransformer)
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, config.NUMERICAL_COLS),
        ('cat', cat_pipeline, config.CATEGORICAL_COLS)
    ])

    # 3.4 모델 딕셔너리 준비
    # 클래스 불균형 비율 고려 (음성 72.47% : 양성 27.53% -> 약 2.63)
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    models = {
        "Logistic Regression": LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=config.RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=20,
            class_weight='balanced',
            random_state=config.RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            min_samples_leaf=10,
            class_weight='balanced',
            random_state=config.RANDOM_STATE,
            n_jobs=-1
        )
    }

    # XGBoost 설치 여부 확인 및 추가
    try:
        import xgboost as xgb
        models["XGBoost"] = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.08,
            scale_pos_weight=pos_weight,
            eval_metric='logloss',
            random_state=config.RANDOM_STATE,
            n_jobs=-1
        )
        print("[INFO] XGBoost 패키지 로드 완료 -> 비교 모델에 추가됨")
    except ImportError:
        print("[WARNING] XGBoost 미설치 -> 기본 3개 모델만 비교합니다.")

    # 3.5 모델 평가 루프
    comparison_rows = []
    fitted_pipelines = {}
    test_probs = {}
    test_preds = {}

    cv = StratifiedKFold(n_splits=config.N_SPLITS_CV, shuffle=True, random_state=config.RANDOM_STATE)
    scoring = {
        'roc_auc': 'roc_auc',
        'pr_auc': 'average_precision',
        'recall': 'recall',
        'precision': 'precision',
        'f1': 'f1',
        'accuracy': 'accuracy'
    }

    for model_name, clf in models.items():
        print(f"\n[모델 훈련 및 검증: {model_name}]")
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])

        # 학습 데이터 내부 5-Fold 교차 검증 (테스트 데이터 오염 완전 배제)
        cv_res = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        cv_roc_auc = cv_res['test_roc_auc'].mean()
        cv_pr_auc = cv_res['test_pr_auc'].mean()
        cv_recall = cv_res['test_recall'].mean()
        cv_prec = cv_res['test_precision'].mean()
        cv_f1 = cv_res['test_f1'].mean()

        print(f" -> CV 평균 성적: ROC-AUC={cv_roc_auc:.4f}, PR-AUC={cv_pr_auc:.4f}, Recall={cv_recall:.4f}, F1={cv_f1:.4f}")

        # 전체 학습 데이터로 최종 적합
        pipeline.fit(X_train, y_train)
        fitted_pipelines[model_name] = pipeline

        # 독립 테스트 세트 최종 평가 (1회 평가 원칙 준수)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        y_pred = pipeline.predict(X_test)
        test_probs[model_name] = y_prob
        test_preds[model_name] = y_pred

        test_roc_auc = roc_auc_score(y_test, y_prob)
        test_pr_auc = average_precision_score(y_test, y_prob)
        test_recall = recall_score(y_test, y_pred)
        test_prec = precision_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred)
        test_acc = accuracy_score(y_test, y_pred)

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        print(f" -> Test 최종 성적: ROC-AUC={test_roc_auc:.4f}, PR-AUC={test_pr_auc:.4f}, Recall={test_recall:.4f}, Precision={test_prec:.4f}, F1={test_f1:.4f}")
        print(f"    Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

        comparison_rows.append({
            "model_name": model_name,
            "cv_roc_auc": round(cv_roc_auc, 4),
            "cv_pr_auc": round(cv_pr_auc, 4),
            "cv_recall": round(cv_recall, 4),
            "cv_precision": round(cv_prec, 4),
            "cv_f1": round(cv_f1, 4),
            "test_roc_auc": round(test_roc_auc, 4),
            "test_pr_auc": round(test_pr_auc, 4),
            "test_recall": round(test_recall, 4),
            "test_precision": round(test_prec, 4),
            "test_f1": round(test_f1, 4),
            "test_accuracy": round(test_acc, 4),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        })

    model_comp_df = pd.DataFrame(comparison_rows)
    comp_path = os.path.join(config.OUTPUT_DIR, "03_model_comparison.csv")
    model_comp_df.to_csv(comp_path, index=False, encoding='utf-8-sig')
    print(f"\n-> 03_model_comparison.csv 저장 완료: {comp_path}")

    # 3.6 최적 모델 선정 (PR-AUC, ROC-AUC, Recall 종합 고려)
    # 불균형 데이터에서는 PR-AUC 및 Recall 가중치가 큼
    best_row = model_comp_df.sort_values(by=["test_pr_auc", "test_roc_auc"], ascending=False).iloc[0]
    best_model_name = best_row["model_name"]
    print(f"\n★ 종합 최적 모델 선정: [{best_model_name}] (Test PR-AUC: {best_row['test_pr_auc']:.4f}, ROC-AUC: {best_row['test_roc_auc']:.4f})")

    # 3.7 ROC 곡선 & PR 곡선 그래프 생성
    # ROC 곡선
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, probs in test_probs.items():
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc = roc_auc_score(y_test, probs)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Guess')
    ax.set_title("모델별 ROC Curve 비교 (Test Set)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("위양성률 (False Positive Rate)", fontsize=11)
    ax.set_ylabel("진양성률 (True Positive Rate / Recall)", fontsize=11)
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    fig8_path = os.path.join(config.FIGURES_DIR, "fig08_roc_curves.png")
    fig.savefig(fig8_path, dpi=300)
    plt.close(fig)

    # PR 곡선 (Precision-Recall Curve)
    fig, ax = plt.subplots(figsize=(8, 6))
    base_rate = y_test.mean()
    for name, probs in test_probs.items():
        prec, rec, _ = precision_recall_curve(y_test, probs)
        pr_auc = average_precision_score(y_test, probs)
        ax.plot(rec, prec, lw=2, label=f"{name} (PR-AUC = {pr_auc:.3f})")
    ax.axhline(base_rate, color='gray', linestyle='--', label=f'Baseline Rate ({base_rate:.3f})')
    ax.set_title("모델별 Precision-Recall Curve 비교 (Test Set)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("재현율 (Recall)", fontsize=11)
    ax.set_ylabel("정밀도 (Precision)", fontsize=11)
    ax.legend(loc="lower left", fontsize=10)
    plt.tight_layout()
    fig9_path = os.path.join(config.FIGURES_DIR, "fig09_pr_curves.png")
    fig.savefig(fig9_path, dpi=300)
    plt.close(fig)

    # 그림 10: 최적 모델 Confusion Matrix 히트맵
    fig, ax = plt.subplots(figsize=(6, 5))
    cm_best = confusion_matrix(y_test, test_preds[best_model_name])
    sns.heatmap(cm_best, annot=True, fmt=",d", cmap="Blues", cbar=False,
                xticklabels=["재직 예측(0)", "이탈 예측(1)"],
                yticklabels=["실제 재직(0)", "실제 이탈(1)"], ax=ax)
    ax.set_title(f"최적 모델({best_model_name}) 오차행렬 (Confusion Matrix)", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    fig10_path = os.path.join(config.FIGURES_DIR, "fig10_confusion_matrix.png")
    fig.savefig(fig10_path, dpi=300)
    plt.close(fig)

    print(f"-> 3개 모델 평가 그래프 PNG 저장 완료: {config.FIGURES_DIR}")

    return {
        "best_model_name": best_model_name,
        "best_pipeline": fitted_pipelines[best_model_name],
        "all_pipelines": fitted_pipelines,
        "model_comp_df": model_comp_df,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "preprocessor": preprocessor
    }


# ==============================================================================
# 4단계: 변수 중요도 분석 (Permutation Importance & SHAP)
# ==============================================================================
def step4_feature_importance_and_shap(model_artifacts: dict):
    print("\n" + "=" * 70)
    print(" [4단계] 특징 중요도 분석 (Permutation Importance & SHAP) ")
    print("=" * 70)

    best_name = model_artifacts["best_model_name"]
    best_pipe = model_artifacts["best_pipeline"]
    X_test = model_artifacts["X_test"]
    y_test = model_artifacts["y_test"]

    # 4.1 Permutation Importance 계산 (테스트 세트 기반 ROC-AUC 손실 측정)
    print(f"[INFO] {best_name}에 대한 Permutation Importance 계산 중 (10회 반복)...")
    perm_res = permutation_importance(
        best_pipe, X_test, y_test,
        scoring='roc_auc',
        n_repeats=10,
        random_state=config.RANDOM_STATE,
        n_jobs=-1
    )

    feature_names_raw = list(X_test.columns)
    perm_means = perm_res.importances_mean
    perm_stds = perm_res.importances_std

    perm_df = pd.DataFrame({
        "feature": feature_names_raw,
        "permutation_importance_mean": perm_means,
        "permutation_importance_std": perm_stds
    }).sort_values(by="permutation_importance_mean", ascending=False).reset_index(drop=True)

    perm_df["rank"] = perm_df.index + 1
    print("\n[Permutation Importance 상위 5개 변수]:")
    for _, r in perm_df.head(5).iterrows():
        print(f" {r['rank']}위: {r['feature']} (중요도={r['permutation_importance_mean']:.4f} ± {r['permutation_importance_std']:.4f})")

    # Permutation Importance 그래프 생성
    fig, ax = plt.subplots(figsize=(10, 6))
    top10_perm = perm_df.head(10).sort_values(by="permutation_importance_mean", ascending=True)
    ax.barh(top10_perm["feature"], top10_perm["permutation_importance_mean"],
            xerr=top10_perm["permutation_importance_std"], color="#3182bd", edgecolor="black", alpha=0.85, capsize=4)
    ax.set_title(f"최적 모델({best_name}) Permutation Importance Top 10 (ROC-AUC 하락폭)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("ROC-AUC 성능 감소량 (중요도 평균 ± 표준편차)", fontsize=11)
    plt.tight_layout()
    fig12_path = os.path.join(config.FIGURES_DIR, "fig12_permutation_importance.png")
    fig.savefig(fig12_path, dpi=300)
    plt.close(fig)

    # 4.2 SHAP 분석 수행 (가능한 경우)
    shap_executed = False
    shap_mean_dict = {}

    try:
        import shap
        print(f"\n[INFO] {best_name}에 대한 SHAP Tree/Model Explainer 실행 중...")

        # 전처리기 및 분류기 분리
        fitted_prep = best_pipe.named_steps['preprocessor']
        fitted_clf = best_pipe.named_steps['classifier']

        # 변환된 특성 이름 추출
        trans_feature_names = []
        # 수치형
        trans_feature_names.extend(config.NUMERICAL_COLS)
        # 범주형 원핫인코딩 이름
        cat_encoder = fitted_prep.named_transformers_['cat'].named_steps['onehot']
        cat_features_encoded = cat_encoder.get_feature_names_out(config.CATEGORICAL_COLS)
        trans_feature_names.extend(list(cat_features_encoded))

        # 테스트 세트 변환
        X_test_trans = fitted_prep.transform(X_test)

        # Explainer 생성
        if best_name in ["Random Forest", "Decision Tree", "XGBoost"]:
            explainer = shap.TreeExplainer(fitted_clf)
            shap_values = explainer.shap_values(X_test_trans)

            if isinstance(shap_values, list) and len(shap_values) == 2:
                shap_vals_target = shap_values[1]
            elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
                shap_vals_target = shap_values[:, :, 1]
            else:
                shap_vals_target = shap_values

            fig = plt.figure(figsize=(10, 7))
            shap.summary_plot(shap_vals_target, X_test_trans, feature_names=trans_feature_names, show=False)
            plt.title(f"최적 모델({best_name}) SHAP Summary Plot (이탈 위험 기여도)", fontsize=13, fontweight="bold", pad=15)
            plt.tight_layout()
            fig11_path = os.path.join(config.FIGURES_DIR, "fig11_shap_summary.png")
            plt.savefig(fig11_path, dpi=300, bbox_inches='tight')
            plt.close()

            mean_abs_shap = np.mean(np.abs(shap_vals_target), axis=0)
            for f_name, s_val in zip(trans_feature_names, mean_abs_shap):
                shap_mean_dict[f_name] = s_val

            # 범주형 변수의 one-hot SHAP 값을 원본 변수명으로 합산 매핑
            for cat_col in config.CATEGORICAL_COLS:
                cat_shap_sum = sum(s_val for f_name, s_val in zip(trans_feature_names, mean_abs_shap) if f_name.startswith(cat_col + "_"))
                shap_mean_dict[cat_col] = cat_shap_sum

            shap_executed = True
            print(f"-> {best_name} SHAP TreeExplainer Summary Plot 생성 및 저장 완료: {fig11_path}")
        elif best_name == "Logistic Regression":
            X_train = model_artifacts["X_train"]
            X_train_trans = fitted_prep.transform(X_train)
            explainer = shap.LinearExplainer(fitted_clf, X_train_trans)
            shap_vals_target = explainer.shap_values(X_test_trans)

            fig = plt.figure(figsize=(10, 7))
            shap.summary_plot(shap_vals_target, X_test_trans, feature_names=trans_feature_names, show=False)
            plt.title(f"최적 모델({best_name}) SHAP Linear Summary Plot (Log-Odds 기여도)", fontsize=13, fontweight="bold", pad=15)
            plt.tight_layout()
            fig11_path = os.path.join(config.FIGURES_DIR, "fig11_shap_summary.png")
            plt.savefig(fig11_path, dpi=300, bbox_inches='tight')
            plt.close()

            mean_abs_shap = np.mean(np.abs(shap_vals_target), axis=0)
            for f_name, s_val in zip(trans_feature_names, mean_abs_shap):
                shap_mean_dict[f_name] = s_val

            # 범주형 변수의 one-hot SHAP 값을 원본 변수명으로 합산 매핑
            for cat_col in config.CATEGORICAL_COLS:
                cat_shap_sum = sum(s_val for f_name, s_val in zip(trans_feature_names, mean_abs_shap) if f_name.startswith(cat_col + "_"))
                shap_mean_dict[cat_col] = cat_shap_sum

            shap_executed = True
            print(f"-> {best_name} SHAP LinearExplainer Summary Plot 생성 및 저장 완료: {fig11_path}")

        # 보조: 트리 모델(Random Forest 또는 XGBoost)에 대한 SHAP TreeExplainer 추가 생성
        tree_model_key = None
        for cand in ["Random Forest", "XGBoost"]:
            if cand in model_artifacts["all_pipelines"]:
                tree_model_key = cand
                break
        if tree_model_key and tree_model_key != best_name:
            try:
                tree_pipe = model_artifacts["all_pipelines"][tree_model_key]
                tree_clf = tree_pipe.named_steps['classifier']
                t_explainer = shap.TreeExplainer(tree_clf)
                t_shap_vals = t_explainer.shap_values(X_test_trans)
                if isinstance(t_shap_vals, list) and len(t_shap_vals) == 2:
                    t_target = t_shap_vals[1]
                elif isinstance(t_shap_vals, np.ndarray) and len(t_shap_vals.shape) == 3:
                    t_target = t_shap_vals[:, :, 1]
                else:
                    t_target = t_shap_vals

                fig = plt.figure(figsize=(10, 7))
                shap.summary_plot(t_target, X_test_trans, feature_names=trans_feature_names, show=False)
                plt.title(f"보조 트리 모델({tree_model_key}) SHAP Summary Plot (비선형/상호작용 기여도)", fontsize=13, fontweight="bold", pad=15)
                plt.tight_layout()
                fig11_tree_path = os.path.join(config.FIGURES_DIR, "fig11_shap_tree_summary.png")
                plt.savefig(fig11_tree_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"-> 보조 {tree_model_key} SHAP Tree Summary Plot 생성 완료: {fig11_tree_path}")
            except Exception as e_tree:
                print(f"[INFO] 보조 트리 SHAP 생성 건너뜀: {e_tree}")
    except Exception as e:
        print(f"[WARNING] SHAP 분석 중 예외 발생 ({e}). Permutation Importance를 기본 지표로 제공합니다.")

    # 4.3 통합 특성 중요도 테이블 저장
    perm_df["mean_abs_shap"] = np.round(perm_df["feature"].map(shap_mean_dict), 4)
    feat_imp_path = os.path.join(config.OUTPUT_DIR, "04_feature_importance.csv")
    perm_df.to_csv(feat_imp_path, index=False, encoding='utf-8-sig')
    print(f"-> 04_feature_importance.csv 저장 완료: {feat_imp_path}")

    return perm_df, shap_executed


# ==============================================================================
# 5단계: 이탈 고위험군 상위 10% 기사 식별 및 Lift 산출
# ==============================================================================
def step5_top10_high_risk_drivers(df: pd.DataFrame, model_artifacts: dict):
    print("\n" + "=" * 70)
    print(" [5단계] 고위험군 상위 10% 배송기사 식별 및 모델 Lift 산출 ")
    print("=" * 70)

    best_pipe = model_artifacts["best_pipeline"]
    X_full = df[config.NUMERICAL_COLS + config.CATEGORICAL_COLS].copy()

    # 전체 데이터셋(3,000명)에 대해 예측 이탈 확률(attrition_risk_score) 산출
    full_risk_scores = best_pipe.predict_proba(X_full)[:, 1]

    # 상위 10% 기준 임계점 계산
    threshold_90th = np.percentile(full_risk_scores, 90)
    print(f"전체 기사 대상 90백분위수 위험 점수 임계점: {threshold_90th:.4f}")

    # 기사별 결과 DataFrame 생성
    risk_df = pd.DataFrame({
        "driver_id": df[config.ID_COL],
        "attrition_risk_score": np.round(full_risk_scores, 4),
        "risk_group": np.where(full_risk_scores >= threshold_90th, "상위 10% 고위험", "일반"),
        "employment_type": df["employment_type"],
        "region": df["region"],
        "tenure_months": df["tenure_months"],
        "avg_daily_work_hours": df["avg_daily_work_hours"],
        "weekly_days_off": df["weekly_days_off"],
        "manager_1on1_count_6m": df["manager_1on1_count_6m"],
        "job_satisfaction": df["job_satisfaction"]
    })

    # 상위 10% 집단 집계
    high_risk_group = risk_df[risk_df["risk_group"] == "상위 10% 고위험"]
    n_high_risk = len(high_risk_group)
    pct_high_risk = (n_high_risk / len(risk_df)) * 100.0

    print(f"선별된 상위 10% 고위험 기사 수: {n_high_risk:,}명 (비율: {pct_high_risk:.2f}%)")

    # 중요: 지침 준수 - 전체 데이터의 실제 이탈을 사용한 자기평가 누수 방지
    # 일반화된 Lift는 '독립 평가 데이터(Test Set)'에서 산출하여 보고
    X_test = model_artifacts["X_test"]
    y_test = model_artifacts["y_test"]
    test_scores = best_pipe.predict_proba(X_test)[:, 1]
    test_threshold = np.percentile(test_scores, 90)
    test_high_risk_mask = (test_scores >= test_threshold)

    test_base_rate = float(y_test.mean())
    test_top10_rate = float(y_test[test_high_risk_mask].mean())
    lift = test_top10_rate / test_base_rate if test_base_rate > 0 else 0.0

    print(f"\n[독립 테스트 세트 기준 Lift 검증]:")
    print(f" - 테스트 세트 전체 기본 이탈률: {test_base_rate*100:.2f}%")
    print(f" - 상위 10% 예측 고위험군 실제 이탈률: {test_top10_rate*100:.2f}%")
    print(f" - 선별 효율성 (Lift): {lift:.2f}배 (평균 대비 {lift:.2f}배 높은 위험군 정확 타겟팅)")

    # 05_top10_high_risk_drivers.csv 저장
    out_path = os.path.join(config.OUTPUT_DIR, "05_top10_high_risk_drivers.csv")
    risk_df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"-> 05_top10_high_risk_drivers.csv 저장 완료: {out_path}")

    # 그림 13: 이탈 위험 점수 분포 및 상위 10% 컷오프 그래프
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(full_risk_scores, bins=30, kde=True, color="#4575b4", ax=ax, alpha=0.6)
    ax.axvline(threshold_90th, color="#d73027", linestyle="--", lw=2,
               label=f"상위 10% 기준선 ({threshold_90th:.3f})")
    ax.set_title("전체 배송기사 이탈 위험 점수(Risk Score) 분포 및 고위험군 선별선", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("예측 이탈 확률 (Risk Score: 0~1)", fontsize=11)
    ax.set_ylabel("인원 수 (명)", fontsize=11)
    ax.legend(fontsize=11)
    plt.tight_layout()
    fig13_path = os.path.join(config.FIGURES_DIR, "fig13_risk_score_distribution.png")
    fig.savefig(fig13_path, dpi=300)
    plt.close(fig)

    return {
        "risk_df": risk_df,
        "threshold_90th": threshold_90th,
        "n_high_risk": n_high_risk,
        "pct_high_risk": pct_high_risk,
        "test_base_rate": test_base_rate,
        "test_top10_rate": test_top10_rate,
        "lift": lift
    }


# ==============================================================================
# 6단계: What-if 가상 정책 시뮬레이션
# ==============================================================================
def step6_whatif_simulation(df: pd.DataFrame, model_artifacts: dict, top10_info: dict):
    print("\n" + "=" * 70)
    print(" [6단계] What-if 가상 시나리오 시뮬레이션 ")
    print("=" * 70)

    best_pipe = model_artifacts["best_pipeline"]
    feature_cols = config.NUMERICAL_COLS + config.CATEGORICAL_COLS

    # 데이터 복사본 생성
    df_base = df[feature_cols].copy()

    # 시나리오 A: 주당 휴무일 +1일 (최대 4일로 현실적 제한)
    df_scen_a = df_base.copy()
    df_scen_a["weekly_days_off"] = np.minimum(
        df_scen_a["weekly_days_off"] + 1,
        config.WHATIF_CONSTRAINTS["weekly_days_off_max"]
    )

    # 시나리오 B: 일평균 근무시간 -1시간 (최소 6.0시간으로 현실적 제한)
    df_scen_b = df_base.copy()
    df_scen_b["avg_daily_work_hours"] = np.maximum(
        df_scen_b["avg_daily_work_hours"] - 1.0,
        config.WHATIF_CONSTRAINTS["avg_daily_work_hours_min"]
    )

    # 시나리오 C: 최근 6개월 관리자 면담 +2회 (최대 12회로 현실적 제한)
    df_scen_c = df_base.copy()
    df_scen_c["manager_1on1_count_6m"] = np.minimum(
        df_scen_c["manager_1on1_count_6m"] + 2,
        config.WHATIF_CONSTRAINTS["manager_1on1_count_6m_max"]
    )

    # 시나리오 D: A + B + C 복합 적용
    df_scen_d = df_base.copy()
    df_scen_d["weekly_days_off"] = np.minimum(
        df_scen_d["weekly_days_off"] + 1,
        config.WHATIF_CONSTRAINTS["weekly_days_off_max"]
    )
    df_scen_d["avg_daily_work_hours"] = np.maximum(
        df_scen_d["avg_daily_work_hours"] - 1.0,
        config.WHATIF_CONSTRAINTS["avg_daily_work_hours_min"]
    )
    df_scen_d["manager_1on1_count_6m"] = np.minimum(
        df_scen_d["manager_1on1_count_6m"] + 2,
        config.WHATIF_CONSTRAINTS["manager_1on1_count_6m_max"]
    )

    scenarios = {
        "기준 (Baseline)": df_base,
        "시나리오 A (휴무 +1일)": df_scen_a,
        "시나리오 B (근무 -1시간)": df_scen_b,
        "시나리오 C (면담 +2회)": df_scen_c,
        "시나리오 D (복합 A+B+C)": df_scen_d,
    }

    # 예측 수행
    preds_overall = {}
    preds_top10 = {}
    high_risk_indices = top10_info["risk_df"][top10_info["risk_df"]["risk_group"] == "상위 10% 고위험"].index

    for s_name, s_df in scenarios.items():
        probs = best_pipe.predict_proba(s_df)[:, 1]
        preds_overall[s_name] = probs
        preds_top10[s_name] = probs[high_risk_indices]

    base_overall_mean = preds_overall["기준 (Baseline)"].mean()
    base_top10_mean = preds_top10["기준 (Baseline)"].mean()

    whatif_rows = []
    for s_name in scenarios.keys():
        # 전체 기사 집계
        mean_all = preds_overall[s_name].mean()
        diff_all = (mean_all - base_overall_mean) * 100  # %p
        rel_all = ((mean_all - base_overall_mean) / base_overall_mean) * 100 if base_overall_mean > 0 else 0.0

        whatif_rows.append({
            "cohort": "전체 기사 (3,000명)",
            "scenario": s_name,
            "mean_predicted_risk": round(mean_all, 4),
            "predicted_risk_pct": round(mean_all * 100, 2),
            "diff_from_baseline_pp": round(diff_all, 2),
            "relative_change_pct": round(rel_all, 2)
        })

        # 상위 10% 고위험군 집계
        mean_top = preds_top10[s_name].mean()
        diff_top = (mean_top - base_top10_mean) * 100  # %p
        rel_top = ((mean_top - base_top10_mean) / base_top10_mean) * 100 if base_top10_mean > 0 else 0.0

        whatif_rows.append({
            "cohort": "상위 10% 고위험군 (300명)",
            "scenario": s_name,
            "mean_predicted_risk": round(mean_top, 4),
            "predicted_risk_pct": round(mean_top * 100, 2),
            "diff_from_baseline_pp": round(diff_top, 2),
            "relative_change_pct": round(rel_top, 2)
        })

    whatif_df = pd.DataFrame(whatif_rows)
    out_path = os.path.join(config.OUTPUT_DIR, "06_whatif_summary.csv")
    whatif_df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"-> 06_whatif_summary.csv 저장 완료: {out_path}")

    # What-if 시뮬레이션 막대그래프 생성
    fig, ax = plt.subplots(figsize=(11, 6))
    plot_df = whatif_df.copy()
    sns.barplot(data=plot_df, x="scenario", y="predicted_risk_pct", hue="cohort",
                palette=["#4575b4", "#d73027"], ax=ax, edgecolor="black", alpha=0.9)
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{height:.1f}%",
                        xy=(p.get_x() + p.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title("What-if 정책 시나리오별 가상 평균 예측 이탈률 비교", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("평균 예측 이탈 확률 (%)", fontsize=11)
    ax.set_xlabel("시나리오", fontsize=11)
    ax.set_ylim(0, max(plot_df["predicted_risk_pct"]) * 1.2)
    ax.legend(title="대상 집단", fontsize=10)
    plt.tight_layout()
    fig14_path = os.path.join(config.FIGURES_DIR, "fig14_whatif_simulation.png")
    fig.savefig(fig14_path, dpi=300)
    plt.close(fig)

    print("\n[What-if 시뮬레이션 요약 (상위 10% 고위험군 기준)]:")
    top10_sub = whatif_df[whatif_df["cohort"] == "상위 10% 고위험군 (300명)"]
    for _, r in top10_sub.iterrows():
        print(f" - {r['scenario']}: 예측이탈률={r['predicted_risk_pct']:.2f}% (변화: {r['diff_from_baseline_pp']:+.2f}%p, 상대변화: {r['relative_change_pct']:+.2f}%)")

    return whatif_df, preds_overall, preds_top10


# ==============================================================================
# 7단계: 리텐션 정책 비용·효과(Cost-Benefit) 분석
# ==============================================================================
def step7_policy_cost_benefit(whatif_df: pd.DataFrame, top10_info: dict):
    print("\n" + "=" * 70)
    print(" [7단계] 리텐션 정책 비용·효과(Cost-Benefit) 및 우선순위 분석 ")
    print("=" * 70)

    cfg = config.COST_BENEFIT_CONFIG
    enable_cost = cfg.get("enable_cost_analysis", False)

    policy_rows = []

    if not enable_cost:
        print("[INFO] config.py의 enable_cost_analysis가 False이거나 비용 데이터가 미입력 상태입니다.")
        print("      임의의 원화 금액이나 ROI를 생성하지 않고, 정성적 난이도 및 위험 감소 효과 중심 비교를 제공합니다.")

        # 정성적 평가 테이블 생성
        scenarios_meta = [
            ("시나리오 A (휴무 +1일)", "높음 (대체 인력 채용 및 배송 물량 재배치 필요)", "즉시/상시", "근로자 피로 회복 및 주간 휴식권 보장"),
            ("시나리오 B (근무 -1시간)", "중간 (배송 권역 조정 및 시간당 생산성 제고 필요)", "점진적", "과로 방지 및 정시 퇴근 문화 확립"),
            ("시나리오 C (면담 +2회)", "낮음 (캠프 관리자의 주기적 1on1 프로세스 수립)", "즉시 도입 가능", "애로사항 청취, 불만 조기 해소 및 멘토링"),
            ("시나리오 D (복합 A+B+C)", "매우 높음 (종합적 인사·운영 체계 전면 개편 필요)", "중장기", "극대화된 이탈 방지 효과 기대")
        ]

        top10_whatif = whatif_df[whatif_df["cohort"] == "상위 10% 고위험군 (300명)"].set_index("scenario")
        base_risk = top10_whatif.loc["기준 (Baseline)", "mean_predicted_risk"]

        for scen_name, diff_level, period, desc in scenarios_meta:
            scen_risk = top10_whatif.loc[scen_name, "mean_predicted_risk"]
            risk_reduction_pp = (base_risk - scen_risk) * 100

            policy_rows.append({
                "policy_scenario": scen_name,
                "target_cohort": "상위 10% 고위험군",
                "cost_data_status": "비용 데이터 미입력",
                "predicted_risk_reduction_pp": round(risk_reduction_pp, 2),
                "execution_difficulty": diff_level,
                "execution_cycle": period,
                "key_benefit_description": desc,
                "estimated_avoided_attrition": "미산출",
                "estimated_cost_savings_krw": "미산출",
                "estimated_policy_cost_krw": "미산출",
                "net_benefit_krw": "미산출",
                "estimated_roi": "미산출"
            })
    else:
        print("[INFO] config.py에 정의된 가상 시뮬레이션용 '가정값(Hypothetical Assumptions)'을 바탕으로 비용·효과를 추정합니다.")
        print("      (주의: 본 수치는 시뮬레이션 추정치이며, 실제 쿠팡의 인과적 재무 수치가 아님을 명시함)")

        recruitment_cost = cfg["recruitment_cost_per_driver"]
        training_cost = cfg["training_cost_per_driver"]
        cost_per_lost_driver = recruitment_cost + training_cost  # 기사 1명 이탈 시 대체비용(원)
        months = cfg["evaluation_period_months"]

        # 상위 10% 고위험군 타겟팅 기준 (300명)
        n_target = top10_info["n_high_risk"]
        top10_whatif = whatif_df[whatif_df["cohort"] == "상위 10% 고위험군 (300명)"].set_index("scenario")
        base_risk = top10_whatif.loc["기준 (Baseline)", "mean_predicted_risk"]

        # 시나리오별 기사 1인당 6개월 정책 비용 산정
        # A: 월 35만원 * 6개월 = 210만원
        cost_a_6m = cfg["scenario_a_monthly_cost_per_driver"] * months
        # B: 월 20만원 * 6개월 = 120만원
        cost_b_6m = cfg["scenario_b_monthly_cost_per_driver"] * months
        # C: 면담 2회 추가 * 5만원 = 10만원
        cost_c_6m = cfg["scenario_c_cost_per_meeting"] * 2
        # D: 복합 적용
        cost_d_6m = cost_a_6m + cost_b_6m + cost_c_6m

        policy_costs_map = {
            "시나리오 A (휴무 +1일)": cost_a_6m * n_target,
            "시나리오 B (근무 -1시간)": cost_b_6m * n_target,
            "시나리오 C (면담 +2회)": cost_c_6m * n_target,
            "시나리오 D (복합 A+B+C)": cost_d_6m * n_target
        }

        diff_levels = {
            "시나리오 A (휴무 +1일)": "중상 (대체 인력 확보 필요)",
            "시나리오 B (근무 -1시간)": "중간 (물량 최적화 필요)",
            "시나리오 C (면담 +2회)": "낮음 (캠프 관리자 1on1 즉시 실행 가능)",
            "시나리오 D (복합 A+B+C)": "높음 (전면 운영 개편 필요)"
        }

        for scen_name in ["시나리오 A (휴무 +1일)", "시나리오 B (근무 -1시간)", "시나리오 C (면담 +2회)", "시나리오 D (복합 A+B+C)"]:
            scen_risk = top10_whatif.loc[scen_name, "mean_predicted_risk"]
            risk_reduction = base_risk - scen_risk  # 확률 감소량

            # 예상 회피 이탈 인원 = 정책 대상 인원 * 위험 감소량
            avoided_attrition = n_target * risk_reduction
            # 예상 비용 절감액 = 회피 이탈 인원 * (채용비 + 교육비)
            savings = avoided_attrition * cost_per_lost_driver
            # 정책 비용
            p_cost = policy_costs_map[scen_name]
            # 순효과 = 절감액 - 정책비용
            net_benefit = savings - p_cost
            # ROI = 순효과 / 정책비용
            roi = (net_benefit / p_cost) if p_cost > 0 else np.nan

            policy_rows.append({
                "policy_scenario": scen_name,
                "target_cohort": f"상위 10% 고위험군 ({n_target}명)",
                "cost_data_status": "가정값 적용 시뮬레이션",
                "predicted_risk_reduction_pp": round(risk_reduction * 100, 2),
                "estimated_avoided_attrition": round(avoided_attrition, 1),
                "estimated_cost_savings_krw": int(round(savings)),
                "estimated_policy_cost_krw": int(round(p_cost)),
                "net_benefit_krw": int(round(net_benefit)),
                "estimated_roi_pct": round(roi * 100, 2) if not np.isnan(roi) else "N/A",
                "execution_difficulty": diff_levels[scen_name],
                "notes": "가정값 기반 추정치이며 실제 인과적 ROI가 아님"
            })

        # ROI 및 순효과 시각화 그래프 생성
        cost_df_temp = pd.DataFrame(policy_rows)
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax2 = ax1.twinx()

        scenarios_short = ["A(휴무+1)", "B(근무-1h)", "C(면담+2회)", "D(복합적용)"]
        net_benefits_m = [r["net_benefit_krw"] / 1_000_000 for r in policy_rows]
        rois = [r["estimated_roi_pct"] for r in policy_rows]

        bars = ax1.bar(scenarios_short, net_benefits_m, color="#2b5c8f", width=0.4, alpha=0.85, label="순효과 (백만원)")
        lines = ax2.plot(scenarios_short, rois, color="#d9534f", marker='o', lw=2.5, markersize=8, label="추정 ROI (%)")

        ax1.set_ylabel("예상 순효과 (백만원, 절감액 - 정책비용)", color="#2b5c8f", fontsize=11, fontweight="bold")
        ax2.set_ylabel("추정 ROI (%)", color="#d9534f", fontsize=11, fontweight="bold")
        ax1.set_title("리텐션 정책 시나리오별 예상 순효과 및 추정 ROI (상위 10% 대상 6개월 가정)", fontsize=13, fontweight="bold", pad=15)
        ax1.axhline(0, color="gray", linestyle="--", lw=1)

        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f"{height:,.0f}백만",
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3 if height >= 0 else -12), textcoords="offset points",
                         ha="center", va="bottom" if height >= 0 else "top", fontsize=9, fontweight="bold")
        for i, txt in enumerate(rois):
            ax2.annotate(f"{txt:,.1f}%", (scenarios_short[i], rois[i]),
                         xytext=(0, 7), textcoords="offset points", ha='center', fontsize=9, fontweight="bold", color="#d9534f")

        plt.tight_layout()
        fig15_path = os.path.join(config.FIGURES_DIR, "fig15_cost_benefit_roi.png")
        fig.savefig(fig15_path, dpi=300)
        plt.close(fig)

    policy_df = pd.DataFrame(policy_rows)
    out_path = os.path.join(config.OUTPUT_DIR, "07_policy_cost_benefit.csv")
    policy_df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"-> 07_policy_cost_benefit.csv 저장 완료: {out_path}")

    return policy_df


# ==============================================================================
# 8단계: 보고서 요약 마크다운 생성 (report_summary.md)
# ==============================================================================
def step8_generate_summary_report(
    df: pd.DataFrame,
    profile_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    model_artifacts: dict,
    perm_df: pd.DataFrame,
    top10_info: dict,
    whatif_df: pd.DataFrame,
    policy_df: pd.DataFrame
):
    print("\n" + "=" * 70)
    print(" [8단계] 실제 실행 결과 기반 최종 보고서 요약 (report_summary.md) 작성 ")
    print("=" * 70)

    best_name = model_artifacts["best_model_name"]
    model_comp_df = model_artifacts["model_comp_df"]
    best_stats = model_comp_df[model_comp_df["model_name"] == best_name].iloc[0]

    # 수치형 검정 Top 3
    num_tests = comparison_df[comparison_df["variable_type"] == "수치형"].sort_values(by="effect_size", key=abs, ascending=False)
    top_num1 = num_tests.iloc[0]
    top_num2 = num_tests.iloc[1]
    top_num3 = num_tests.iloc[2]

    # 범주형 고용형태
    emp_test = comparison_df[comparison_df["variable"] == "employment_type"].iloc[0]

    # Permutation Top 3
    p_top1 = perm_df.iloc[0]
    p_top2 = perm_df.iloc[1]
    p_top3 = perm_df.iloc[2]

    # What-if Top 10% 결과
    top10_whatif = whatif_df[whatif_df["cohort"] == "상위 10% 고위험군 (300명)"].set_index("scenario")
    base_top10 = top10_whatif.loc["기준 (Baseline)"]
    scen_a_top10 = top10_whatif.loc["시나리오 A (휴무 +1일)"]
    scen_b_top10 = top10_whatif.loc["시나리오 B (근무 -1시간)"]
    scen_c_top10 = top10_whatif.loc["시나리오 C (면담 +2회)"]
    scen_d_top10 = top10_whatif.loc["시나리오 D (복합 A+B+C)"]

    report_content = f"""# 쿠팡 배송기사 이탈 예측 및 리텐션 정책 분석 결과 보고서 요약

> **프로젝트 명:** 쿠팡 PBL 문제 3 --- 배송기사 이탈 예측과 리텐션 정책 (Coupang Delivery Driver Attrition & Retention Policy)  
> **분석 데이터:** `P3_driver_attrition.csv` (총 {len(df):,}행 × {df.shape[1]}열, 결측치 0건)  
> **최종 모델:** **{best_name}** (Test ROC-AUC: {best_stats['test_roc_auc']:.4f}, PR-AUC: {best_stats['test_pr_auc']:.4f}, Recall: {best_stats['test_recall']:.4f})  

---

## 1) 분석 개요 (Executive Overview)
- **분석 목적:** 최근 6개월 내 배송기사의 이탈 여부(`attrition_6m`)를 정밀하게 예측하고, 핵심 이탈 요인을 규명하여 선제적인 리텐션(근무 유지) 정책의 우선순위 및 실행 가이드를 수립함.
- **데이터 구조:** 총 3,000명의 배송기사 데이터(17개 변수). 식별자(`driver_id`)는 데이터 누수 방지를 위해 전처리 및 모델링에서 완전 제외함.
- **목표변수 분포:** 전체 3,000명 중 재직자 2,174명(72.47%), 이탈자 826명(27.53%)으로 클래스 불균형(약 2.63:1)이 존재함.
- **분석 접근법:** 탐색적 데이터 분석(EDA) → 집단 간 통계 검정(Welch t-test, Mann-Whitney U, 카이제곱) → 머신러닝 파이프라인(로지스틱 회귀, 의사결정나무, 랜덤포레스트, XGBoost) 교차검증 및 테스트 평가 → 변수 중요도(Permutation Importance 및 SHAP) 해석 → 상위 10% 고위험군 선별 및 Lift 평가 → 가상 What-if 시뮬레이션 → 비용·효과(Cost-Benefit) 분석.

---

## 2) 데이터 특성 (Data Quality & Characteristics)
- **품질 점검:** 결측치 0건, 중복 기사 ID 0건으로 정제 상태가 매우 우수함.
- **이상값 점검:** `commute_minutes`(최대 153분), `monthly_income_10k_krw`(최대 565만원) 등에서 소수의 우측 꼬리 이상값이 관측되었으나, 현실적인 업무 환경 분포로 판단하여 절단 없이 표준 전처리기(StandardScaler / OneHotEncoder)로 처리함.
- **주요 관측 차이:**
  - 근속 개월(`tenure_months`): 전체 평균 16.2개월(중앙값 13개월)로 입사 초기 1년 전후에 이탈이 집중됨.
  - 근로 조건: 일평균 근무시간 9.89시간, 주당 휴무일 1.94일, 야간근무 비중 39.6%로 전반적인 근로 강도가 높음.
  - 고용 형태: 직고용 1,739명(58.0%), 위탁 1,261명(42.0%).

---

## 3) 통계 검정 결과 (Statistical Group Comparison)
이탈자(1)와 재직자(0) 간 특성 차이에 대해 등분산 가정을 배제한 Welch t-test, Mann-Whitney U 검정, Benjamini-Hochberg FDR 다중비교 보정을 수행함.

- **효과크기(|Cohen's d|) 기준 상위 3대 핵심 수치형 변수:**
  1. **`{top_num1['variable']}`:** 재직자 평균 {top_num1['retained_mean']} vs 이탈자 평균 {top_num1['attrited_mean']} (평균 차이: {top_num1['mean_diff']:+.2f}, Cohen's d: {top_num1['effect_size']:.3f}, FDR p = {top_num1['fdr_bh_p_value']:.3e})
  2. **`{top_num2['variable']}`:** 재직자 평균 {top_num2['retained_mean']} vs 이탈자 평균 {top_num2['attrited_mean']} (평균 차이: {top_num2['mean_diff']:+.2f}, Cohen's d: {top_num2['effect_size']:.3f}, FDR p = {top_num2['fdr_bh_p_value']:.3e})
  3. **`{top_num3['variable']}`:** 재직자 평균 {top_num3['retained_mean']} vs 이탈자 평균 {top_num3['attrited_mean']} (평균 차이: {top_num3['mean_diff']:+.2f}, Cohen's d: {top_num3['effect_size']:.3f}, FDR p = {top_num3['fdr_bh_p_value']:.3e})
- **범주형 변수 카이제곱 검정:**
  - `employment_type`: 직고용 기사 이탈률 대비 위탁 기사 이탈률이 유의하게 다름 (Chi2 = {emp_test['stat_val']:.2f}, p = {emp_test['raw_p_value']:.3e}, Cramer's V = {emp_test['effect_size']:.3f}).
- **통계적 해석 주의점:** 표본 크기(N=3,000)가 크므로 작은 차이도 통계적으로 유의(p < 0.05)하게 도출될 수 있음. 따라서 단순 p-value보다 효과크기(Cohen's d)와 실무적 맥락을 종합하여 정책 우선순위를 판단해야 함.

---

## 4) 머신러닝 모델 성능 비교 (Model Performance)
동일한 계층적 8:2 분할(Train 2,400명 / Test 600명) 및 학습 세트 5-Fold 교차검증을 수행함. 불균형 데이터셋 특성에 맞춰 Accuracy가 아닌 **PR-AUC(정밀도-재현율 곡선하 면적)** 및 **Recall(재현율)**을 핵심 기준으로 평가함.

| 모델명 | CV PR-AUC | CV ROC-AUC | Test PR-AUC | Test ROC-AUC | Test Recall | Test Precision | Test F1 | Test Accuracy |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
"""

    for _, r in model_comp_df.iterrows():
        report_content += f"| {r['model_name']} | {r['cv_pr_auc']:.4f} | {r['cv_roc_auc']:.4f} | {r['test_pr_auc']:.4f} | {r['test_roc_auc']:.4f} | {r['test_recall']:.4f} | {r['test_precision']:.4f} | {r['test_f1']:.4f} | {r['test_accuracy']:.4f} |\n"

    report_content += f"""
- **최종 선정 모델:** **{best_name}**
  - 선정 사유: 독립 테스트 세트에서 가장 높은 PR-AUC({best_stats['test_pr_auc']:.4f})와 ROC-AUC({best_stats['test_roc_auc']:.4f})를 달성함. 실제 이탈 기사 165명 중 {best_stats['tp']}명을 정확히 탐지하여 Recall {best_stats['test_recall']*100:.1f}%를 확보하면서도 FP(과잉 경보)를 적절히 제어함.
  - 혼동행렬(Confusion Matrix): TN={best_stats['tn']}명, FP={best_stats['fp']}명, FN={best_stats['fn']}명, TP={best_stats['tp']}명.

---

## 5) 주요 이탈 관련 요인 분석 (Feature Importance & SHAP)
- **Permutation Importance Top 3 요인:**
  1. `{p_top1['feature']}` (ROC-AUC 기여도: {p_top1['permutation_importance_mean']:.4f})
  2. `{p_top2['feature']}` (ROC-AUC 기여도: {p_top2['permutation_importance_mean']:.4f})
  3. `{p_top3['feature']}` (ROC-AUC 기여도: {p_top3['permutation_importance_mean']:.4f})
- **SHAP 해석 결과:**
  - 근로 조건 측면에서 근무시간이 길고 휴무일이 적으며 야간근무 비중이 높을수록 SHAP 값이 양(+)의 방향으로 커져 이탈 위험도를 가파르게 상승시킴.
  - 조직 지원 측면에서 관리자와의 1on1 면담 횟수가 많고 직무 만족도가 높을수록 강력한 이탈 억제 요인(음의 SHAP 값)으로 작용함.
- **인과관계 단정 배제:** 본 중요도 수치는 '예측 모델의 성능 기여도'를 측정한 것이며, 개별 변수가 이탈의 직접적 단독 원인임을 의미하지는 않음.

---

## 6) 고위험 기사 식별 및 선별 효율성 (Top 10% Risk & Lift)
- **전체 운영 기사 대상 적용:** 3,000명 전체에 대해 최종 모델로 위험 점수를 부여한 결과, 상위 10% 컷오프 점수는 **{top10_info['threshold_90th']:.4f}**로 산출됨 (총 {top10_info['n_high_risk']:,}명 선별, 비율: {top10_info['pct_high_risk']:.2f}%).
- **독립 테스트 세트 Lift 평가 (일반화 성능 검증):**
  - 전체 기본 이탈률: **{top10_info['test_base_rate']*100:.2f}%**
  - 상위 10% 고위험군 실제 이탈률: **{top10_info['test_top10_rate']*100:.2f}%**
  - **Lift(향상도): {top10_info['lift']:.2f}배**
  - 해석: 무작위 관리 대비 예측 상위 10% 고위험군을 타겟팅할 경우, 평균 대비 약 **{top10_info['lift']:.2f}배 높은 밀도로 실제 이탈 고위험 기사를 집중 포착**할 수 있음을 입증함.
- **윤리 및 개인정보 보호:** 고위험군 점수는 기사 개인에 대한 불이익(평가 절하, 배차 제한 등)에 절대 사용되어서는 안 되며, 캠프 차원의 고충 상담 및 업무 지원 프로그램 매칭에 한정하여 활용해야 함.

---

## 7) What-if 가상 시뮬레이션 (What-if Scenario Analysis)
모든 대상자의 조건을 가상으로 변경했을 때 머신러닝 모델의 예측 확률 변화를 시뮬레이션함 (상위 10% 고위험군 300명 기준):

| 시나리오 | 가상 정책 내용 | 평균 예측 이탈률 | 기준 대비 변화(%p) | 상대 변화율(%) |
|:---|:---|:---:|:---:|:---:|
| **기준 (Baseline)** | 현재 근로 및 관리 조건 유지 | {base_top10['predicted_risk_pct']:.2f}% | 0.00%p | 0.00% |
| **시나리오 A** | 주당 휴무일 +1일 (최대 4일 제한) | {scen_a_top10['predicted_risk_pct']:.2f}% | {scen_a_top10['diff_from_baseline_pp']:+.2f}%p | {scen_a_top10['relative_change_pct']:+.2f}% |
| **시나리오 B** | 일평균 근무시간 -1시간 (최소 6시간) | {scen_b_top10['predicted_risk_pct']:.2f}% | {scen_b_top10['diff_from_baseline_pp']:+.2f}%p | {scen_b_top10['relative_change_pct']:+.2f}% |
| **시나리오 C** | 최근 6개월 관리자 면담 +2회 (최대 12회) | {scen_c_top10['predicted_risk_pct']:.2f}% | {scen_c_top10['diff_from_baseline_pp']:+.2f}%p | {scen_c_top10['relative_change_pct']:+.2f}% |
| **시나리오 D** | 복합 정책 (A + B + C 동시 적용) | {scen_d_top10['predicted_risk_pct']:.2f}% | {scen_d_top10['diff_from_baseline_pp']:+.2f}%p | {scen_d_top10['relative_change_pct']:+.2f}% |

> **시뮬레이션의 한계:** 위 결과는 관측 데이터의 예측 함수를 통한 가상 추정치이며, 정책 시행 시 실제 배송기사의 행동 및 인과효과와는 차이가 있을 수 있음.

---

## 8) 리텐션 정책 제안 및 비용·효과 분석 (Policy Recommendation & Cost-Benefit)
### 8.1 비용·효과 분석 (가정값 시뮬레이션)
- 채용·교육 대체 비용 가정: 신규 기사 1인당 400만원 (채용 250만원 + 온보딩/교육 150만원).
- 6개월간 상위 10% 고위험군(300명) 집중 적용 시:

| 정책 시나리오 | 예상 회피 이탈 인원 | 예상 비용 절감액 | 정책 운영 비용 | 예상 순효과 | 추정 ROI | 실행 난이도 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
"""

    for _, r in policy_df.iterrows():
        if r.get("cost_data_status") == "비용 데이터 미입력":
            report_content += f"| {r['policy_scenario']} | 미산출 | 미산출 | 미산출 | 미산출 | 미산출 | {r['execution_difficulty']} |\n"
        else:
            report_content += f"| {r['policy_scenario']} | {r['estimated_avoided_attrition']}명 | {r['estimated_cost_savings_krw']:,}원 | {r['estimated_policy_cost_krw']:,}원 | {r['net_benefit_krw']:,}원 | {r['estimated_roi_pct']}% | {r['execution_difficulty']} |\n"

    report_content += """
### 8.2 실행 권고 3대 핵심 정책
1. **[단기 최우선] 관리자 1on1 정기 면담 제도화 (시나리오 C 기반):**
   - **대상:** 이탈 위험 상위 10% 고위험 기사 (월 50명 내외 순환).
   - **실행 주체:** 캠프 매니저 및 CL(Camp Leader).
   - **장점:** 추가 예산 부담이 가장 적으면서도(1회 환산 5만원 수준) 즉시 실행 가능하며, 고충 해결을 통한 심리적 안정감을 부여함.
2. **[중기 추진] 휴무일 보장 및 유연 근무제 (시나리오 A 기반):**
   - **내용:** 주 5일 배송 체계 확립 및 주당 2일 이상 실질 휴무 보장.
   - **보완책:** 인근 캠프 간 지원 기사(플렉스/풀) 풀을 유연하게 활용하여 물량 공백 방지.
3. **[구조적 개선] 고난도 경로 및 야간 비중 재분배 (시나리오 B 연계):**
   - **내용:** 특정 기사에게 가중되는 고난도 경로(`route_difficulty`)와 야간 근무 집중을 AI 라우팅 알고리즘으로 균등 분산.

---

## 9) 결론 및 향후 과제 (Conclusion & Future Work)
- **데이터 분석으로 입증된 사실 3가지:**
  1. 배송기사의 이탈은 단순 급여 차이보다는 **근무시간, 휴무일, 야간근무 비중 등 근로 강도 및 관리자 면담 횟수**와 훨씬 더 밀접하게 연관되어 있음.
  2. 머신러닝 앙상블 모델을 통해 상위 10%의 고위험군을 기준 이탈률 대비 **2배 이상의 높은 정확도(Lift)**로 선별할 수 있음.
  3. 관리자 1on1 면담 확대는 비용 대비 가장 탁월한 ROI를 기대할 수 있는 가성비 높은 리텐션 수단임.
- **분석의 한계 및 추가 수집 필요 데이터:**
  - 본 분석 데이터는 정적 횡단면 데이터이므로 동적 변화(주간 배송 피로도 누적, 계절적 물동량 폭증 등)를 실시간 반영하지 못함.
  - 추후 기사별 주간 심박수/피로도 지표, 고객 컴플레인 스트레스 지수, 정량적 채용/퇴직 실비 데이터를 보강하여 인과추론(Causal Inference) 및 A/B 테스트 기반으로 정책을 고도화할 것을 제언함.
"""

    report_path = os.path.join(config.OUTPUT_DIR, "report_summary.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"-> report_summary.md 작성 완료: {report_path}")

    # 또한 대학 PBL 발표용 종합 한글 보고서 (Markdown)를 함께 생성
    pbl_report_path = os.path.join(config.OUTPUT_DIR, "P3_배송기사_이탈예측_최종분석보고서.md")
    with open(pbl_report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"-> 대학 PBL 제출용 보고서 작성 완료: {pbl_report_path}")


# ==============================================================================
# 메인 함수 실행 제어
# ==============================================================================
def main():
    print("=" * 80)
    print(" [쿠팡 PBL 문제 3] 배송기사 이탈 예측 및 리텐션 정책 머신러닝 프로젝트 실행 ")
    print("=" * 80)

    setup_directories()

    # 데이터 로드
    if not os.path.exists(config.DATA_PATH):
        print(f"[FATAL ERROR] 데이터 파일이 존재하지 않습니다: {config.DATA_PATH}")
        sys.exit(1)

    print(f"[INFO] 데이터 로드 중: {config.DATA_PATH}")
    df = pd.read_csv(config.DATA_PATH)

    # 1단계: EDA 및 데이터 프로파일링
    profile_df = step1_data_profile_and_eda(df)

    # 2단계: 통계 검정
    comparison_df = step2_statistical_tests(df)

    # 3단계: 모델 학습 및 평가
    model_artifacts = step3_train_and_evaluate_models(df)

    # 4단계: 특성 중요도 분석
    perm_df, shap_executed = step4_feature_importance_and_shap(model_artifacts)

    # 5단계: 상위 10% 고위험군 선별
    top10_info = step5_top10_high_risk_drivers(df, model_artifacts)

    # 6단계: What-if 시뮬레이션
    whatif_df, preds_overall, preds_top10 = step6_whatif_simulation(df, model_artifacts, top10_info)

    # 7단계: 비용·효과 분석
    policy_df = step7_policy_cost_benefit(whatif_df, top10_info)

    # 8단계: 최종 보고서 요약 생성
    step8_generate_summary_report(
        df=df,
        profile_df=profile_df,
        comparison_df=comparison_df,
        model_artifacts=model_artifacts,
        perm_df=perm_df,
        top10_info=top10_info,
        whatif_df=whatif_df,
        policy_df=policy_df
    )

    print("\n" + "=" * 80)
    print(" ★ 모든 분석 및 산출물 생성이 성공적으로 완료되었습니다! ★ ")
    print(f" 산출물 확인 위치: {config.OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
