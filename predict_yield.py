import joblib
import numpy as np
import pandas as pd

# 1. Load model artifact
artifact = joblib.load("model2_crop_yield_predictor.joblib")
model = artifact["model"]
feature_names = artifact["feature_names"]


def predict_crop_yield(
    crop: str,
    state: str,
    crop_irrigated_ha: float = 0.0,
    total_district_irr_1000ha: float = 50.0,
    n_kg_ha: float = 80.0,
    p_kg_ha: float = 40.0,
    k_kg_ha: float = 20.0,
    soil_type: str = "INCEPTISOLS",
    soil_pct: float = 70.0,
    monsoon_rainfall_mm: float = 850.0,
    annual_rainfall_mm: float = 1100.0,
):
    """Predicts expected yield (kg/ha) using the trained Model 2 pipeline."""
    # Build default feature row with zeros
    input_row = pd.DataFrame(0.0, index=[0], columns=feature_names)

    # Populate numerical variables
    input_row["Crop_Irrigated_Area_ha"] = crop_irrigated_ha
    input_row["Is_Irrigated"] = 1.0 if crop_irrigated_ha > 0 else 0.0
    input_row["Total_District_Irrigated_1000_ha"] = total_district_irr_1000ha
    input_row["Fertilizer_N_kg_per_ha"] = n_kg_ha
    input_row["Fertilizer_P_kg_per_ha"] = p_kg_ha
    input_row["Fertilizer_K_kg_per_ha"] = k_kg_ha
    input_row["Fertilizer_NPK_Total_kg_per_ha"] = n_kg_ha + p_kg_ha + k_kg_ha
    input_row["Dominant_Soil_Percent"] = soil_pct

    # Populate primary rainfall aggregations
    if "Rainfall_Annual" in input_row.columns:
        input_row["Rainfall_Annual"] = annual_rainfall_mm
    if "Rainfall_Kharif_Monsoon" in input_row.columns:
        input_row["Rainfall_Kharif_Monsoon"] = monsoon_rainfall_mm

    # Distribute monsoon rain across Jun-Sep if columns exist
    for m in ["Rainfall_Jun", "Rainfall_Jul", "Rainfall_Aug", "Rainfall_Sep"]:
        if m in input_row.columns:
            input_row[m] = monsoon_rainfall_mm / 4.0

    # Activate one-hot encoded flags
    crop_col = f"Crop_{crop.strip().upper()}"
    state_col = f"State Name_{state.strip()}"
    soil_col = f"Dominant_Soil_Type_{soil_type.strip().upper()}"

    for col in [crop_col, state_col, soil_col]:
        if col in input_row.columns:
            input_row[col] = 1.0

    pred_yield = model.predict(input_row)[0]
    return max(0.0, round(float(pred_yield), 2))


# --- Demonstration Simulations ---
print("=" * 65)
print("MODEL 2: INTERACTIVE YIELD SIMULATION DEMONSTRATION")
print("=" * 65)

# Scenario A: Rainfed vs Irrigated Wheat in Madhya Pradesh
yield_rainfed = predict_crop_yield(
    crop="WHEAT",
    state="Madhya Pradesh",
    crop_irrigated_ha=0.0,
    n_kg_ha=50,
    p_kg_ha=20,
    k_kg_ha=10,
)
yield_irrigated = predict_crop_yield(
    crop="WHEAT",
    state="Madhya Pradesh",
    crop_irrigated_ha=2500.0,
    n_kg_ha=120,
    p_kg_ha=60,
    k_kg_ha=30,
)

print("\n[Scenario A: Wheat in Madhya Pradesh]")
print(f"  Rainfed + Low Fertilizer (50-20-10 NPK)  : {yield_rainfed:,} kg/ha")
print(
    f"  Irrigated + Optimal Fertilizer (120-60-30): {yield_irrigated:,} kg/ha"
)
print(
    f"  Gain from Irrigation & Inputs             : +{yield_irrigated - yield_rainfed:,.1f} kg/ha"
)

# Scenario B: Rice under Normal Monsoon vs Drought in Chhattisgarh
yield_rice_normal = predict_crop_yield(
    crop="RICE",
    state="Chhattisgarh",
    monsoon_rainfall_mm=1100.0,
    annual_rainfall_mm=1300.0,
)
yield_rice_drought = predict_crop_yield(
    crop="RICE",
    state="Chhattisgarh",
    monsoon_rainfall_mm=450.0,
    annual_rainfall_mm=600.0,
)

print("\n[Scenario B: Rice in Chhattisgarh under Rainfall Variance]")
print(f"  Normal Monsoon (1100 mm) : {yield_rice_normal:,} kg/ha")
print(f"  Severe Drought (450 mm)  : {yield_rice_drought:,} kg/ha")
print(
    f"  Drought Yield Impact     : {yield_rice_drought - yield_rice_normal:,.1f} kg/ha"
)

# Scenario C: Crop Comparison for same plot (Inceptisols soil, Andhra Pradesh)
print("\n[Scenario C: Crop Comparison (Andhra Pradesh, 80-40-20 NPK)]")
for test_crop in ["RICE", "MAIZE", "GROUNDNUT", "CHICKPEA", "SOYABEAN"]:
    y = predict_crop_yield(
        crop=test_crop,
        state="Andhra Pradesh",
        crop_irrigated_ha=500.0,
        n_kg_ha=80,
        p_kg_ha=40,
        k_kg_ha=20,
    )
    print(f"  {test_crop:12s}: {y:,.1f} kg/ha")
print("=" * 65)