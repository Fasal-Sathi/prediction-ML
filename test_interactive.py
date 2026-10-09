import sys
import joblib
import pandas as pd

# Load saved model artifact
print("Loading model artifact...")
try:
    artifact = joblib.load("model2_crop_yield_predictor.joblib")
    model = artifact["model"]
    feature_names = artifact["feature_names"]
    print("Model loaded successfully!\n")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

# Available crops & states known to the model
available_crops = sorted(
    [f.replace("Crop_", "") for f in feature_names if f.startswith("Crop_")]
)
available_states = sorted(
    [
        f.replace("State Name_", "")
        for f in feature_names
        if f.startswith("State Name_")
    ]
)
available_soils = sorted(
    [
        f.replace("Dominant_Soil_Type_", "")
        for f in feature_names
        if f.startswith("Dominant_Soil_Type_")
    ]
)

print("=" * 60)
print("     MANUAL INTERACTIVE CROP YIELD TESTER")
print("=" * 60)


def ask_choice(prompt, options):
    print(f"\n{prompt}:")
    for i, opt in enumerate(options, 1):
        print(f"  [{i:2d}] {opt}")
    while True:
        choice = input(f"Enter number (1-{len(options)}): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
        print("Invalid selection, please try again.")


def ask_float(prompt, default_val):
    val = input(f"{prompt} [default: {default_val}]: ").strip()
    if not val:
        return float(default_val)
    try:
        return float(val)
    except ValueError:
        print(f"Invalid number, using default: {default_val}")
        return float(default_val)


while True:
    crop = ask_choice("Select Crop", available_crops)
    state = ask_choice("Select State", available_states)
    soil = ask_choice("Select Soil Type", available_soils)

    print("\n--- Enter Farm & Climate Conditions ---")
    crop_irr = ask_float("Enter Crop Irrigated Area in Hectares (0 if rainfed)", 0.0)
    monsoon_rain = ask_float("Enter Monsoon Season Rainfall in mm", 850.0)
    annual_rain = ask_float("Enter Total Annual Rainfall in mm", 1100.0)
    n_fert = ask_float("Enter Nitrogen (N) applied in kg/ha", 80.0)
    p_fert = ask_float("Enter Phosphorus (P) applied in kg/ha", 40.0)
    k_fert = ask_float("Enter Potassium (K) applied in kg/ha", 20.0)

    # Prepare input vector
    input_row = pd.DataFrame(0.0, index=[0], columns=feature_names)
    input_row["Crop_Irrigated_Area_ha"] = crop_irr
    input_row["Is_Irrigated"] = 1.0 if crop_irr > 0 else 0.0
    input_row["Total_District_Irrigated_1000_ha"] = 50.0
    input_row["Fertilizer_N_kg_per_ha"] = n_fert
    input_row["Fertilizer_P_kg_per_ha"] = p_fert
    input_row["Fertilizer_K_kg_per_ha"] = k_fert
    input_row["Fertilizer_NPK_Total_kg_per_ha"] = n_fert + p_fert + k_fert
    input_row["Dominant_Soil_Percent"] = 70.0

    if "Rainfall_Annual" in input_row.columns:
        input_row["Rainfall_Annual"] = annual_rain
    if "Rainfall_Kharif_Monsoon" in input_row.columns:
        input_row["Rainfall_Kharif_Monsoon"] = monsoon_rain
    for m in ["Rainfall_Jun", "Rainfall_Jul", "Rainfall_Aug", "Rainfall_Sep"]:
        if m in input_row.columns:
            input_row[m] = monsoon_rain / 4.0

    # Categorical flags
    if f"Crop_{crop}" in input_row.columns:
        input_row[f"Crop_{crop}"] = 1.0
    if f"State Name_{state}" in input_row.columns:
        input_row[f"State Name_{state}"] = 1.0
    if f"Dominant_Soil_Type_{soil}" in input_row.columns:
        input_row[f"Dominant_Soil_Type_{soil}"] = 1.0

    # Predict
    pred_yield = max(0.0, float(model.predict(input_row)[0]))

    print("\n" + "=" * 50)
    print(" PREDICTION RESULT")
    print("=" * 50)
    print(f" Crop:                 {crop}")
    print(f" State:                {state}")
    print(f" Soil Type:            {soil}")
    print(f" Water Regime:         {'Irrigated' if crop_irr > 0 else 'Rainfed'}")
    print(f" N-P-K Applied:        {n_fert:.1f} - {p_fert:.1f} - {k_fert:.1f} kg/ha")
    print("-" * 50)
    print(f" Predicted Yield:      {pred_yield:,.2f} kg/ha")
    print(f" In Tonnes per Hectare:{pred_yield / 1000.0:,.3f} Tonnes/ha")
    print("=" * 50)

    again = input("\nDo you want to test another combination? (y/n): ").strip().lower()
    if again != "y":
        print("\nExiting tester. Done!")
        break