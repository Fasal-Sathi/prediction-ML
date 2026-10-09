import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("1/4 Loading dataset and preparing features...")
df = pd.read_csv("Model2_yield_rain_irr_fert_soil.csv")
df_clean = df[df["Yield_kg_per_ha"] > 0].copy()

# Standardize 2018-2019 Sugarcane scale shift and remove statistical typos
cane_mask = (df_clean["Crop"] == "SUGARCANE") & (df_clean["Year"] >= 2018)
df_clean.loc[cane_mask, "Yield_kg_per_ha"] /= 10.0
df_clean = df_clean[
    ~(
        (df_clean["Crop"] != "SUGARCANE")
        & (df_clean["Yield_kg_per_ha"] > 15000)
    )
].copy()

# Imputations
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

# Agronomic Interaction Ratios
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

# Define Families
FAMILIES = {
    "cereals": {
        "crops": ["RICE", "WHEAT", "MAIZE", "BARLEY"],
        "max_iter": 350,
        "lr": 0.06,
        "depth": 12,
        "min_leaf": 20,
    },
    "pulses": {
        "crops": ["CHICKPEA", "MINOR PULSES", "SOYABEAN"],
        "max_iter": 250,
        "lr": 0.05,
        "depth": 8,
        "min_leaf": 30,
    },
    "cash_oilseeds": {
        "crops": ["SUGARCANE", "GROUNDNUT", "SUNFLOWER", "OILSEEDS", "LINSEED"],
        "max_iter": 350,
        "lr": 0.07,
        "depth": 12,
        "min_leaf": 20,
    },
}

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

specialist_artifacts = {}

print("\n2/4 Training Specialist Engines...")

overall_actuals = []
overall_preds = []

for fam_name, config in FAMILIES.items():
    fam_crops = config["crops"]
    sub_df = df_clean[df_clean["Crop"].isin(fam_crops)].copy()

    # One-hot encode for this family
    cat_cols = ["Crop", "State Name", "Dominant_Soil_Type"]
    df_encoded = pd.get_dummies(sub_df[cat_cols], drop_first=False)
    X = pd.concat([sub_df[["Year"] + num_cols], df_encoded], axis=1)
    y = sub_df["Yield_kg_per_ha"]
    X = X.loc[:, ~X.columns.duplicated()]

    train_mask = (X["Year"] <= 2014).values
    test_mask = (X["Year"] >= 2015).values

    X_train = X.loc[train_mask].drop(columns=["Year"])
    y_train = y.loc[train_mask]
    X_test = X.loc[test_mask].drop(columns=["Year"])
    y_test = y.loc[test_mask]

    fam_model = HistGradientBoostingRegressor(
        max_iter=config["max_iter"],
        learning_rate=config["lr"],
        max_depth=config["depth"],
        min_samples_leaf=config["min_leaf"],
        random_state=42,
    )
    fam_model.fit(X_train, y_train)

    y_pred = fam_model.predict(X_test)
    fam_r2 = r2_score(y_test, y_pred)
    fam_mae = mean_absolute_error(y_test, y_pred)
    fam_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"\n--- {fam_name.upper()} SPECIALIST ---")
    print(f"Crops: {fam_crops}")
    print(f"Train samples: {len(X_train):,} | Test samples: {len(X_test):,}")
    print(f"Test R²:   {fam_r2:.4f}")
    print(f"Test MAE:  {fam_mae:.2f} kg/ha")
    print(f"Test RMSE: {fam_rmse:.2f} kg/ha")

    # Store artifact
    specialist_artifacts[fam_name] = {
        "model": fam_model,
        "feature_names": X_train.columns.tolist(),
        "crops": fam_crops,
    }

    overall_actuals.extend(y_test.values)
    overall_preds.extend(y_pred)

overall_r2 = r2_score(overall_actuals, overall_preds)
overall_mae = mean_absolute_error(overall_actuals, overall_preds)
overall_rmse = np.sqrt(mean_squared_error(overall_actuals, overall_preds))

print("\n" + "=" * 55)
print("COMBINED SPECIALIST SYSTEM EVALUATION")
print("=" * 55)
print(f"Combined Test R² Score: {overall_r2:.4f}")
print(f"Combined Test MAE:      {overall_mae:.2f} kg/ha")
print(f"Combined Test RMSE:     {overall_rmse:.2f} kg/ha")
print("=" * 55)

print("\n3/4 Saving Multi-Model Specialist Artifact...")
joblib.dump(specialist_artifacts, "model2_specialists.joblib")
print("Saved to: model2_specialists.joblib")