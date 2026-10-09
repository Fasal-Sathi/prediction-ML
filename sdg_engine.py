import joblib
import numpy as np
import pandas as pd

# Load the trained specialist artifact
print("Loading specialist model suite...")
specialists = joblib.load("model2_specialists.joblib")


def get_specialist_prediction(crop: str, state: str, soil: str, inputs: dict):
    crop_upper = crop.strip().upper()
    fam_name = None

    # Route crop to corresponding specialist model
    for f_name, data in specialists.items():
        if crop_upper in data["crops"]:
            fam_name = f_name
            break

    if not fam_name:
        fam_name = "cereals"

    engine = specialists[fam_name]
    model = engine["model"]
    feature_names = engine["feature_names"]

    # Build feature row
    row = pd.DataFrame(0.0, index=[0], columns=feature_names)

    # Assign numerical features
    for col, val in inputs.items():
        if col in row.columns:
            row[col] = float(val)

    # Set one-hot encoded columns
    for prefix, name in [
        ("Crop_", crop_upper),
        ("State Name_", state),
        ("Dominant_Soil_Type_", soil),
    ]:
        col = f"{prefix}{name}"
        if col in row.columns:
            row[col] = 1.0

    pred_kg_ha = max(0.0, float(model.predict(row)[0]))
    return round(pred_kg_ha, 2)


# Crop agronomic benchmark requirements (per hectare)
CROP_AGRONOMY_BENCHMARKS = {
    "RICE": {
        "opt_n": 100,
        "opt_p": 50,
        "opt_k": 50,
        "water_req_m3": 12000,
        "is_legume": False,
    },
    "WHEAT": {
        "opt_n": 120,
        "opt_p": 60,
        "opt_k": 40,
        "water_req_m3": 4500,
        "is_legume": False,
    },
    "MAIZE": {
        "opt_n": 120,
        "opt_p": 60,
        "opt_k": 40,
        "water_req_m3": 5500,
        "is_legume": False,
    },
    "CHICKPEA": {
        "opt_n": 20,
        "opt_p": 40,
        "opt_k": 20,
        "water_req_m3": 3000,
        "is_legume": True,
    },
    "SOYABEAN": {
        "opt_n": 30,
        "opt_p": 60,
        "opt_k": 40,
        "water_req_m3": 4500,
        "is_legume": True,
    },
    "GROUNDNUT": {
        "opt_n": 25,
        "opt_p": 50,
        "opt_k": 75,
        "water_req_m3": 5000,
        "is_legume": True,
    },
    "SUGARCANE": {
        "opt_n": 250,
        "opt_p": 100,
        "opt_k": 120,
        "water_req_m3": 20000,
        "is_legume": False,
    },
    "SUNFLOWER": {
        "opt_n": 60,
        "opt_p": 40,
        "opt_k": 30,
        "water_req_m3": 4000,
        "is_legume": False,
    },
    "BARLEY": {
        "opt_n": 60,
        "opt_p": 30,
        "opt_k": 20,
        "water_req_m3": 3500,
        "is_legume": False,
    },
    "LINSEED": {
        "opt_n": 40,
        "opt_p": 20,
        "opt_k": 20,
        "water_req_m3": 3000,
        "is_legume": False,
    },
    "MINOR PULSES": {
        "opt_n": 20,
        "opt_p": 40,
        "opt_k": 20,
        "water_req_m3": 2800,
        "is_legume": True,
    },
    "OILSEEDS": {
        "opt_n": 50,
        "opt_p": 30,
        "opt_k": 30,
        "water_req_m3": 3800,
        "is_legume": False,
    },
}


def calculate_sdg_metrics(
    crop: str,
    state: str,
    soil: str,
    land_acres: float,
    soil_moisture_pct: float,
    applied_n_kg_ha: float,
    applied_p_kg_ha: float,
    applied_k_kg_ha: float,
    monsoon_rainfall_mm: float = 850.0,
):
    crop_upper = crop.strip().upper()
    land_ha = land_acres * 0.404686
    benchmarks = CROP_AGRONOMY_BENCHMARKS.get(
        crop_upper,
        {
            "opt_n": 80,
            "opt_p": 40,
            "opt_k": 30,
            "water_req_m3": 5000,
            "is_legume": False,
        },
    )

    # 1. Yield Prediction
    is_irrigated = 1 if soil_moisture_pct > 50.0 else 0
    total_npk = applied_n_kg_ha + applied_p_kg_ha + applied_k_kg_ha
    safe_npk = total_npk if total_npk > 0 else 1.0

    model_inputs = {
        "Crop_Irrigated_Area_ha": land_ha if is_irrigated else 0.0,
        "Is_Irrigated": is_irrigated,
        "Total_District_Irrigated_1000_ha": 50.0,
        "Fertilizer_N_kg_per_ha": applied_n_kg_ha,
        "Fertilizer_P_kg_per_ha": applied_p_kg_ha,
        "Fertilizer_K_kg_per_ha": applied_k_kg_ha,
        "Fertilizer_NPK_Total_kg_per_ha": total_npk,
        "N_Ratio": applied_n_kg_ha / safe_npk,
        "P_Ratio": applied_p_kg_ha / safe_npk,
        "K_Ratio": applied_k_kg_ha / safe_npk,
        "Dominant_Soil_Percent": 70.0,
        "Rainfall_Annual": 1100.0,
        "Rainfall_Kharif_Monsoon": monsoon_rainfall_mm,
        "Moisture_Index": monsoon_rainfall_mm * (1.0 + 0.5 * is_irrigated),
    }

    pred_yield_kg_ha = get_specialist_prediction(
        crop_upper, state, soil, model_inputs
    )
    harvest_quintals_per_acre = round(pred_yield_kg_ha / 247.105, 2)
    total_harvest_quintals = round(harvest_quintals_per_acre * land_acres, 1)

    # 2. SDG 6: Clean Water & Sanitation
    baseline_flood_m3 = (benchmarks["water_req_m3"] * 1.35) * land_ha
    effective_rainfall_m3 = (monsoon_rainfall_mm * 10) * 0.65 * land_ha
    actual_water_needed_m3 = max(
        0.0, (benchmarks["water_req_m3"] * land_ha) - effective_rainfall_m3
    )

    if soil_moisture_pct >= 55.0:
        water_saved_m3 = baseline_flood_m3 - (actual_water_needed_m3 * 0.8)
        irrigation_advice = "Soil moisture adequate. Irrigation stopped (Water Saved)."
    else:
        water_saved_m3 = baseline_flood_m3 - actual_water_needed_m3
        irrigation_advice = "Irrigation recommended to reach root threshold."

    water_saved_liters = max(0.0, water_saved_m3 * 1000.0)

    # 3. SDG 12: Responsible Consumption (Nutrient Leaching & Runoff Risk)
    excess_n_kg_ha = max(0.0, applied_n_kg_ha - benchmarks["opt_n"])
    if excess_n_kg_ha == 0:
        chemical_status = "Optimal Chemical Dosage (Zero Runoff Risk)"
        leaching_risk_score = 10.0  # 10 out of 100 risk
    elif excess_n_kg_ha <= 30:
        chemical_status = "Mild Excess Nitrogen (Marginal Leaching Risk)"
        leaching_risk_score = 45.0
    else:
        chemical_status = f"Severe Excess Nitrogen (+{excess_n_kg_ha:.1f} kg/ha above crop uptake capacity)"
        leaching_risk_score = 85.0

    # 4. SDG 15: Life on Land (Soil Regeneration)
    if benchmarks["is_legume"]:
        bio_n_fixed_kg = round(40.0 * land_ha, 1)
        soil_impact = f"Leguminous crop biological fixation: adds ~{bio_n_fixed_kg} kg natural Nitrogen back to soil."
        soil_regeneration_score = 90
    else:
        bio_n_fixed_kg = 0.0
        soil_impact = "Exhaustive crop. Rotate with pulses next season to restore soil microbiome."
        soil_regeneration_score = 55

    # 5. SDG 13: Climate Action (Emissions intensity)
    # Urea production + field N2O emission factor: ~4.5 kg CO2e per kg synthetic N applied
    carbon_emitted_kg = round(applied_n_kg_ha * land_ha * 4.5, 1)

    return {
        "Crop": crop_upper,
        "Land_Acres": land_acres,
        "Predicted_Yield_Quintals_Acre": harvest_quintals_per_acre,
        "Total_Harvest_Quintals": total_harvest_quintals,
        "SDG_6_Water_Saved_Liters": round(water_saved_liters, 0),
        "SDG_6_Irrigation_Advisory": irrigation_advice,
        "SDG_12_Nutrient_Status": chemical_status,
        "SDG_12_Runoff_Risk_Index": leaching_risk_score,
        "SDG_15_Biological_N_Fixed_kg": bio_n_fixed_kg,
        "SDG_15_Soil_Status": soil_impact,
        "SDG_15_Regeneration_Score": soil_regeneration_score,
        "SDG_13_Carbon_Footprint_kg_CO2e": carbon_emitted_kg,
    }


# --- Demonstration Run ---
print("\n" + "=" * 65)
print("       SUSTAINABLE DEVELOPMENT (SDG) REPORT DEMO")
print("=" * 65)

# Example: Farmer with 3 Acres of Rice in Odisha
report_rice = calculate_sdg_metrics(
    crop="RICE",
    state="Odisha",
    soil="INCEPTISOLS",
    land_acres=3.0,
    soil_moisture_pct=62.0,  # Sensor shows adequate moisture
    applied_n_kg_ha=140.0,  # Applied excess Urea (140 vs 100 benchmark)
    applied_p_kg_ha=40.0,
    applied_k_kg_ha=20.0,
)

print(
    f"Crop: {report_rice['Crop']} | Land: {report_rice['Land_Acres']} Acres"
)
print(
    f"Predicted Yield: {report_rice['Predicted_Yield_Quintals_Acre']} Quintals/Acre "
    f"({report_rice['Total_Harvest_Quintals']} Quintals Total)"
)
print("-" * 65)
print(f"[SDG 6 - Clean Water]")
print(
    f"  Water Saved vs Flood Irrigation: {report_rice['SDG_6_Water_Saved_Liters']:,.0f} Liters"
)
print(f"  IoT Advisory: {report_rice['SDG_6_Irrigation_Advisory']}")
print(f"[SDG 12 - Responsible Consumption]")
print(f"  Chemical Runoff Risk: {report_rice['SDG_12_Runoff_Risk_Index']}/100")
print(f"  Nutrient Assessment: {report_rice['SDG_12_Nutrient_Status']}")
print(f"[SDG 15 - Life on Land]")
print(f"  Soil Regeneration Score: {report_rice['SDG_15_Regeneration_Score']}/100")
print(f"  Soil Ecology Note: {report_rice['SDG_15_Soil_Status']}")
print(f"[SDG 13 - Climate Action]")
print(
    f"  Synthesized GHG Footprint: {report_rice['SDG_13_Carbon_Footprint_kg_CO2e']} kg CO2e"
)

# Example: Crop Rotation to Chickpea (Pulse) on 3 Acres
print("\n" + "-" * 65)
print("Scenario 2: Crop Rotation with Chickpea (Sustainable Legume)")
print("-" * 65)
report_pulse = calculate_sdg_metrics(
    crop="CHICKPEA",
    state="Odisha",
    soil="INCEPTISOLS",
    land_acres=3.0,
    soil_moisture_pct=40.0,
    applied_n_kg_ha=20.0,
    applied_p_kg_ha=40.0,
    applied_k_kg_ha=20.0,
)
print(
    f"Predicted Yield: {report_pulse['Predicted_Yield_Quintals_Acre']} Quintals/Acre"
)
print(
    f"[SDG 15] Biological N Fixed: +{report_pulse['SDG_15_Biological_N_Fixed_kg']} kg Natural Nitrogen"
)
print(
    f"[SDG 12] Runoff Risk: {report_pulse['SDG_12_Runoff_Risk_Index']}/100 ({report_pulse['SDG_12_Nutrient_Status']})"
)
print(
    f"[SDG 13] Carbon Footprint: {report_pulse['SDG_13_Carbon_Footprint_kg_CO2e']} kg CO2e (vs {report_rice['SDG_13_Carbon_Footprint_kg_CO2e']} kg for Rice)"
)
print("=" * 65)