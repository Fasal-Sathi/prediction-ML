import pandas as pd
import numpy as np

print("1/3 Loading dataset...")
df = pd.read_csv("Model2_yield_rain_irr_fert_soil.csv")
print(f"Total Rows: {len(df):,}")
print(f"Total Columns: {df.shape[1]}")

# Identify exact yield column name regardless of casing
yield_col = [c for c in df.columns if "yield" in c.lower()][0]
print(f"Detected target column: '{yield_col}'")

print("\n" + "="*50)
print(f"2/3 TARGET VARIABLE AUDIT: {yield_col}")
print("="*50)
target = df[yield_col]
print(f"Missing values (NaN): {target.isna().sum():,} ({target.isna().mean()*100:.2f}%)")
print(f"Negative values (< 0): {(target < 0).sum():,}")
print(f"Zero values (== 0): {(target == 0).sum():,}")
print("\nPercentiles:")
print(target.describe(percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]))

print("\n" + "="*50)
print("3/3 FEATURE MISSINGNESS REPORT")
print("="*50)
null_counts = df.isna().sum()
null_pct = (df.isna().mean() * 100).round(2)
missing_report = pd.DataFrame({"Missing_Count": null_counts, "Missing_Percent": null_pct})
print(missing_report[missing_report["Missing_Count"] > 0].sort_values(by="Missing_Percent", ascending=False))

print("\n" + "="*50)
print("YEAR DISTRIBUTION (For Train / Test Split)")
print("="*50)
print(f"Year range: {df['Year'].min()} to {df['Year'].max()}")
print(f"Rows 1990-2014 (Historical Train): {(df['Year'] <= 2014).sum():,} ({(df['Year'] <= 2014).mean()*100:.1f}%)")
print(f"Rows 2015-2019 (Out-of-time Test): {(df['Year'] >= 2015).sum():,} ({(df['Year'] >= 2015).mean()*100:.1f}%)")
