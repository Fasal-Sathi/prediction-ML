import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score

print("1/4 Loading dataset and rebuilding optimal training matrix...")
df = pd.read_csv("Model2_yield_rain_irr_fert_soil.csv")
df_clean = df[df["Yield_kg_per_ha"] > 0].copy()

# 2018-2019 Sugarcane fix
cane_shift_mask = (df_clean["Crop"] == "SUGARCANE") & (
    df_clean["Year"] >= 2018
)
df_clean.loc[cane_shift_mask, "Yield_kg_per_ha"] = (
    df_clean.loc[cane_shift_mask, "Yield_kg_per_ha"] / 10.0
)

# Outlier filtering
unrealistic_mask = (df_clean["Crop"] != "SUGARCANE") & (
    df_clean["Yield_kg_per_ha"] > 15000
)
df_clean = df_clean[~unrealistic_mask].copy()

# Imputations
df_clean["Crop_Irrigated_Area_ha"] = df_clean["Crop_Irrigated_Area_ha"].fillna(0)
df_clean["Is_Irrigated"] = (df_clean["Crop_Irrigated_Area_ha"] > 0).astype(int)

df_clean["Total_District_Irrigated_1000_ha"] = df_clean.groupby(
    ["State Name", "Year"]
)["Total_District_Irrigated_1000_ha"].transform(lambda x: x.fillna(x.median()))
df_clean["Total_District_Irrigated_1000_ha"] = df_clean[
    "Total_District_Irrigated_1000_ha"
].fillna(df_clean["Total_District_Irrigated_1000_ha"].median())

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

cat_cols = ["Crop", "State Name", "Dominant_Soil_Type"]
rainfall_cols = [c for c in df_clean.columns if c.startswith("Rainfall_")]
num_cols = (
    ["Crop_Irrigated_Area_ha", "Total_District_Irrigated_1000_ha"]
    + fert_cols
    + ["Dominant_Soil_Percent", "Is_Irrigated"]
    + rainfall_cols
)

df_encoded = pd.get_dummies(df_clean[cat_cols], drop_first=False)
X_all = pd.concat([df_clean[["Year"] + num_cols], df_encoded], axis=1)
y_all = df_clean["Yield_kg_per_ha"]
X_all = X_all.loc[:, ~X_all.columns.duplicated()]

train_mask = (X_all["Year"] <= 2014).values
test_mask = (X_all["Year"] >= 2015).values

X_train = X_all.loc[train_mask].drop(columns=["Year"])
y_train = y_all.loc[train_mask]
X_test = X_all.loc[test_mask].drop(columns=["Year"])
y_test = y_all.loc[test_mask]

feature_names = X_train.columns.tolist()

print("2/4 Training final model...")
model = HistGradientBoostingRegressor(
    max_iter=300,
    learning_rate=0.07,
    max_depth=12,
    min_samples_leaf=20,
    random_state=42,
)
model.fit(X_train, y_train)

# Save artifact
model_filename = "model2_crop_yield_predictor.joblib"
joblib.dump({"model": model, "feature_names": feature_names}, model_filename)
print(f"Artifact saved: {model_filename}")

print("\n3/4 Calculating Top Permutation Feature Importances on Test Set...")
# Subsample 2500 test points for fast and accurate importance calculation
sample_idx = np.random.RandomState(42).choice(
    len(X_test), size=min(2500, len(X_test)), replace=False
)
X_test_sub = X_test.iloc[sample_idx]
y_test_sub = y_test.iloc[sample_idx]

perm = permutation_importance(
    model,
    X_test_sub,
    y_test_sub,
    n_repeats=5,
    random_state=42,
    scoring="neg_root_mean_squared_error",
)

importance_df = pd.DataFrame(
    {
        "Feature": feature_names,
        "Importance_RMSE_Impact": perm.importances_mean,
    }
).sort_values(by="Importance_RMSE_Impact", ascending=False)

print("\n" + "=" * 55)
print("TOP 15 MOST INFLUENTIAL FEATURES ON YIELD")
print("=" * 55)
print(importance_df.head(15).to_string(index=False))