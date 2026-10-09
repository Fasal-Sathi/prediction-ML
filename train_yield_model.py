import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("1/5 Loading master dataset...")
df = pd.read_csv("Model2_yield_rain_irr_fert_soil.csv")

# Filter out uncultivated zero yield entries
df_clean = df[df["Yield_kg_per_ha"] > 0].copy()

# Fix the 2018-2019 Sugarcane 10x unit reporting shift in ICRISAT
cane_shift_mask = (df_clean["Crop"] == "SUGARCANE") & (
    df_clean["Year"] >= 2018
)
df_clean.loc[cane_shift_mask, "Yield_kg_per_ha"] = (
    df_clean.loc[cane_shift_mask, "Yield_kg_per_ha"] / 10.0
)

# Remove unrealistic statistical noise/typos (> 20,000 kg/ha for non-sugarcane)
unrealistic_mask = (df_clean["Crop"] != "SUGARCANE") & (
    df_clean["Yield_kg_per_ha"] > 15000
)
df_clean = df_clean[~unrealistic_mask].copy()

print(f"Clean valid active records: {len(df_clean):,}")

print("2/5 Feature Engineering & Imputation...")
df_clean["Crop_Irrigated_Area_ha"] = df_clean["Crop_Irrigated_Area_ha"].fillna(0)
df_clean["Is_Irrigated"] = (df_clean["Crop_Irrigated_Area_ha"] > 0).astype(int)

# District total irrigation imputation
df_clean["Total_District_Irrigated_1000_ha"] = df_clean.groupby(
    ["State Name", "Year"]
)["Total_District_Irrigated_1000_ha"].transform(lambda x: x.fillna(x.median()))
df_clean["Total_District_Irrigated_1000_ha"] = df_clean[
    "Total_District_Irrigated_1000_ha"
].fillna(df_clean["Total_District_Irrigated_1000_ha"].median())

# Fertilizer imputation
fert_cols = [
    "Fertilizer_N_kg_per_ha",
    "Fertilizer_P_kg_per_ha",
    "Fertilizer_K_kg_per_ha",
    "Fertilizer_NPK_Total_kg_per_ha",
]
for col in fert_cols:
    df_clean[col] = df_clean.groupby(["State Name", "Year"])[col].transform(
        lambda x: x.fillna(x.median())
    )
    df_clean[col] = df_clean[col].fillna(df_clean[col].median())

# Soil imputation
df_clean["Dominant_Soil_Type"] = df_clean["Dominant_Soil_Type"].fillna(
    "Unknown"
)
df_clean["Dominant_Soil_Percent"] = df_clean["Dominant_Soil_Percent"].fillna(
    df_clean["Dominant_Soil_Percent"].median()
)

# Feature subsets
cat_cols = ["Crop", "State Name", "Dominant_Soil_Type"]
rainfall_cols = [c for c in df_clean.columns if c.startswith("Rainfall_")]
num_cols = (
    ["Crop_Irrigated_Area_ha", "Total_District_Irrigated_1000_ha"]
    + fert_cols
    + ["Dominant_Soil_Percent", "Is_Irrigated"]
    + rainfall_cols
)

target_col = "Yield_kg_per_ha"

# One-hot encoding
df_encoded = pd.get_dummies(df_clean[cat_cols], drop_first=False)
X_all = pd.concat([df_clean[["Year"] + num_cols], df_encoded], axis=1)
y_all = df_clean[target_col]
X_all = X_all.loc[:, ~X_all.columns.duplicated()]

# 3. Temporal Train/Test Split
print("3/5 Performing Temporal Train/Test Split (1990-2014 vs 2015-2019)...")
train_mask = (X_all["Year"] <= 2014).values
test_mask = (X_all["Year"] >= 2015).values

X_train = X_all.loc[train_mask].drop(columns=["Year"])
y_train = y_all.loc[train_mask]

X_test = X_all.loc[test_mask].drop(columns=["Year"])
y_test = y_all.loc[test_mask]

print(f"Training samples: {len(X_train):,}")
print(f"Testing samples:  {len(X_test):,}")

# 4. Train Model
print("\n4/5 Training Gradient Boosting Model...")
model = HistGradientBoostingRegressor(
    max_iter=300,
    learning_rate=0.07,
    max_depth=12,
    min_samples_leaf=20,
    random_state=42,
)
model.fit(X_train, y_train)

# 5. Evaluate Overall & Per-Crop Performance
print("5/5 Evaluating on unseen Out-of-Time Test Set (2015-2019)...")
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

train_mae = mean_absolute_error(y_train, y_pred_train)
test_mae = mean_absolute_error(y_test, y_pred_test)

train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

print("\n" + "=" * 55)
print("CORRECTED MODEL 2: CROP YIELD PREDICTOR REPORT")
print("=" * 55)
print(f"Train R² Score:   {train_r2:.4f}")
print(f"Test  R² Score:   {test_r2:.4f}")
print("-" * 55)
print(f"Train MAE:        {train_mae:.2f} kg/ha")
print(f"Test  MAE:        {test_mae:.2f} kg/ha")
print("-" * 55)
print(f"Train RMSE:       {train_rmse:.2f} kg/ha")
print(f"Test  RMSE:       {test_rmse:.2f} kg/ha")
print("=" * 55)

# Per-Crop Performance Breakdown on the Test Set
test_df_eval = pd.DataFrame(
    {
        "Crop": df_clean.loc[test_mask, "Crop"].values,
        "Actual": y_test.values,
        "Predicted": y_pred_test,
    }
)

crop_metrics = []
for crop, grp in test_df_eval.groupby("Crop"):
    r2 = r2_score(grp["Actual"], grp["Predicted"])
    mae = mean_absolute_error(grp["Actual"], grp["Predicted"])
    rmse = np.sqrt(mean_squared_error(grp["Actual"], grp["Predicted"]))
    crop_metrics.append(
        {
            "Crop": crop,
            "Test_Samples": len(grp),
            "Mean_Yield": round(grp["Actual"].mean(), 1),
            "MAE": round(mae, 1),
            "RMSE": round(rmse, 1),
            "R2_Score": round(r2, 4),
        }
    )

print("\nPer-Crop Evaluation Breakdown on Unseen Test Years (2015-2019):")
print(pd.DataFrame(crop_metrics).to_string(index=False))