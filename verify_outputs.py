import pandas as pd
import numpy as np

print("=== 1. CHECKING 01_data_profile.csv ===")
df1 = pd.read_csv("outputs/01_data_profile.csv")
print(f"Rows: {len(df1)}, Columns: {df1.columns.tolist()}")
print(df1[["variable", "data_type", "null_count", "null_pct", "unique_values", "mean", "std", "min", "max"]].head(5))

print("\n=== 2. CHECKING 02_group_comparison.csv ===")
df2 = pd.read_csv("outputs/02_group_comparison.csv")
print(f"Total rows: {len(df2)}, Variables: {df2['variable'].nunique()}")
num_df = df2[df2["variable_type"] == "수치형"].sort_values(by="effect_size", key=abs, ascending=False)
print("Top 5 effect size in numerical:")
print(num_df[["variable", "retained_mean", "attrited_mean", "mean_diff", "effect_size", "fdr_bh_p_value"]].head(5))

print("\n=== 3. CHECKING 03_model_comparison.csv ===")
df3 = pd.read_csv("outputs/03_model_comparison.csv")
print(df3[["model_name", "cv_pr_auc", "cv_roc_auc", "test_pr_auc", "test_roc_auc", "test_recall", "test_precision", "test_f1", "test_accuracy", "tn", "fp", "fn", "tp"]])

print("\n=== 4. CHECKING 04_feature_importance.csv ===")
df4 = pd.read_csv("outputs/04_feature_importance.csv")
print(df4)

print("\n=== 5. CHECKING 05_top10_high_risk_drivers.csv ===")
df5 = pd.read_csv("outputs/05_top10_high_risk_drivers.csv")
print(f"Total rows: {len(df5)}")
print(df5["risk_group"].value_counts())
p90 = np.percentile(df5["attrition_risk_score"], 90)
print(f"Risk score min: {df5['attrition_risk_score'].min():.4f}, median: {df5['attrition_risk_score'].median():.4f}, 90th: {p90:.4f}, max: {df5['attrition_risk_score'].max():.4f}")

print("\n=== 6. CHECKING 06_whatif_summary.csv ===")
df6 = pd.read_csv("outputs/06_whatif_summary.csv")
print(df6[["cohort", "scenario", "predicted_risk_pct", "diff_from_baseline_pp", "relative_change_pct"]])

print("\n=== 7. CHECKING 07_policy_cost_benefit.csv ===")
df7 = pd.read_csv("outputs/07_policy_cost_benefit.csv")
print(df7[["policy_scenario", "target_cohort", "predicted_risk_reduction_pp", "estimated_avoided_attrition", "estimated_cost_savings_krw", "estimated_policy_cost_krw", "net_benefit_krw", "estimated_roi_pct", "execution_difficulty"]])
