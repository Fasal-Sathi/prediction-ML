import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("1/4 Loading dataset and engineering features...")
df = pd.read_csv("Model2_yield_rain_irr_fert_soil.csv")
df_clean = df[df["Yield_kg_per_ha"] > 0].copy()

# A. Standardize Sugarcane unit shift & filter statistical typos
cane_mask = (df_clean["Crop"] == "SUGARCANE") & (df_clean["Year"] >= 2018)
df_clean.loc[cane_mask, "Yield_kg_per_ha"] /= 10.0
df_clean = df_clean[
    ~(
        (df_clean["Crop"] != "SUGARCANE")
        & (df_clean["Yield_kg_per_ha"] > 15000)
    )
].copy()

# B. Imputations
df_clean["Crop_Irrigated_Area_ha"] = df_clean["Crop_Irrigated_Area_ha"].fillna(0)
df_clean["Is_Irrigated"] = (df_clean["Crop_Irrigated_Area_ha"] > 0).astype(int)

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

df_clean["Dominant_Soil_Type"] = df_clean["Dominant_Soil_Type"].fillna(
    "Unknown"
)
df_clean["Dominant_Soil_Percent"] = df_clean["Dominant_Soil_Percent"].fillna(
    df_clean["Dominant_Soil_Percent"].median()
)
df_clean["Total_District_Irrigated_1000_ha"] = df_clean[
    "Total_District_Irrigated_1000_ha"
].fillna(df_clean["Total_District_Irrigated_1000_ha"].median())

# C. Keep Agronomic Ratios & Interactions
total_npk = df_clean["Fertilizer_NPK_Total_kg_per_ha"].replace(0, np.nan)
df_clean["N_Ratio"] = (
    df_clean["Fertilizer_N_kg_per_ha"] / total_npk
).fillna(0)
df_clean["P_Ratio"] = (
    df_clean["Fertilizer_P_kg_per_ha"] / total_npk
).fillna(0)
df_clean["K_Ratio"] = (
    df_clean["Fertilizer_K_kg_per_ha"] / total_npk
).fillna(0)

if "Rainfall_Kharif_Monsoon" in df_clean.columns:
    df_clean["Moisture_Index"] = df_clean["Rainfall_Kharif_Monsoon"] * (
        1.0 + 0.5 * df_clean["Is_Irrigated"]
    )
else:
    df_clean["Moisture_Index"] = 0.0

# D. Feature Matrix Assembly
cat_cols = ["Crop", "State Name", "Dominant_Soil_Type"]
rainfall_cols = [c for c in df_clean.columns if c.startswith("Rainfall_")]
engineered_cols = ["N_Ratio", "P_Ratio", "K_Ratio", "Moisture_Index"]
num_cols = (
    [
        "Crop_Irrigated_Area_ha",
        "Total_District_Irrigated_1000_ha",
        "Dominant_Soil_Percent",
        "Is_Irrigated",
    ]
    + fert_cols
    + engineered_cols
    + rainfall_cols
)

df_encoded = pd.get_dummies(df_clean[cat_cols], drop_first=False)
X_all = pd.concat([df_clean[["Year"] + num_cols], df_encoded], axis=1)
y_all = df_clean["Yield_kg_per_ha"]  # TRAIN ON TRUE PHYSICAL SCALE
X_all = X_all.loc[:, ~X_all.columns.duplicated()]

# E. Temporal Split (1990-2014 vs 2015-2019)
train_mask = (X_all["Year"] <= 2014).values
test_mask = (X_all["Year"] >= 2015).values

X_train, y_train = X_all.loc[train_mask].drop(columns=["Year"]), y_all.loc[train_mask]
X_test, y_test = X_all.loc[test_mask].drop(columns=["Year"]), y_all.loc[test_mask]

feature_names = X_train.columns.tolist()

# F. Optimized Regressor without Log Distortion
print("2/4 Training optimized gradient booster on raw scale...")
model = HistGradientBoostingRegressor(
    max_iter=350,
    learning_rate=0.07,
    max_depth=12,
    min_samples_leaf=20,
    l2_regularization=0.1,  # Light regularization to avoid underfitting
    random_state=42,
)
model.fit(X_train, y_train)

# G. Evaluate
print("3/4 Evaluating on unseen Test Set (2015-2019)...")
y_pred_test = model.predict(X_test)

r2 = r2_score(y_test, y_pred_test)
mae = mean_absolute_error(y_test, y_pred_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

print("\n" + "=" * 55)
print("UPDATED LINEAR OPTIMIZED EVALUATION REPORT")
print("=" * 55)
print(f"Test R² Score:   {r2:.4f}  (Baseline was: 0.7206)")
print(f"Test MAE:        {mae:.2f} kg/ha  (Baseline was: 647.75 kg/ha)")
print(f"Test RMSE:       {rmse:.2f} kg/ha (Baseline was: 1103.83 kg/ha)")
print("=" * 55)

# Save artifact
joblib.dump(
    {"model": model, "feature_names": feature_names},
    "model2_crop_yield_predictor_tuned.joblib",
)
print("Artifact saved: model2_crop_yield_predictor_tuned.joblib")