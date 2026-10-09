import sys
import joblib
import pandas as pd

print("Loading specialist model suite...")
try:
    specialists = joblib.load("model2_specialists.joblib")
    print("Specialists loaded successfully!\n")
except Exception as e:
    print(f"Error loading 'model2_specialists.joblib': {e}")
    sys.exit(1)

# Map crops to specialist family engines
CROP_FAMILY_MAP = {}
for fam_name, data in specialists.items():
    for c in data["crops"]:
        CROP_FAMILY_MAP[c] = fam_name

available_crops = sorted(list(CROP_FAMILY_MAP.keys()))

# Get reference states and soils from the cereals model feature list
sample_features = specialists["cereals"]["feature_names"]
available_states = sorted(
    [
        f.replace("State Name_", "")
        for f in sample_features
        if f.startswith("State Name_")
    ]
)
available_soils = sorted(
    [
        f.replace("Dominant_Soil_Type_", "")
        for f in sample_features
        if f.startswith("Dominant_Soil_Type_")
    ]
)

print("=" * 60)
print("     SPECIALIST CROP YIELD & SDG TESTER")
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
        print(f"Invalid input, using default: {default_val}")
        return float(default_val)


while True:
    crop = ask_choice("Select Crop", available_crops)
    state = ask_choice("Select State", available_states)
    soil = ask_choice("Select Soil Type", available_soils)

    print("\n--- Enter Farm & Climate Conditions ---")
    acres = ask_float("Enter Land Size in Acres", 3.0)
    crop_irr = ask_float(
        "Enter Irrigated Area in Hectares (0 if rainfed)", 0.0
    )
    monsoon_rain = ask_float("Enter Monsoon Season Rainfall in mm", 850.0)
    annual_rain = ask_float("Enter Annual Rainfall in mm", 1100.0)
    n_fert = ask_float("Enter Nitrogen (N) applied in kg/ha", 80.0)
    p_fert = ask_float("Enter Phosphorus (P) applied in kg/ha", 40.0)
    k_fert = ask_float("Enter Potassium (K) applied in kg/ha", 20.0)

    # Route to matching specialist model
    fam_name = CROP_FAMILY_MAP[crop]
    engine = specialists[fam_name]
    model = engine["model"]
    feature_names = engine["feature_names"]

    # Build input row
    row = pd.DataFrame(0.0, index=[0], columns=feature_names)

    # Assign numerical and ratio features
    total_npk = n_fert + p_fert + k_fert
    safe_npk = total_npk if total_npk > 0 else 1.0
    is_irrigated = 1.0 if crop_irr > 0 else 0.0

    row["Crop_Irrigated_Area_ha"] = crop_irr
    row["Is_Irrigated"] = is_irrigated
    row["Total_District_Irrigated_1000_ha"] = 50.0
    row["Fertilizer_N_kg_per_ha"] = n_fert
    row["Fertilizer_P_kg_per_ha"] = p_fert
    row["Fertilizer_K_kg_per_ha"] = k_fert
    row["Fertilizer_NPK_Total_kg_per_ha"] = total_npk
    row["N_Ratio"] = n_fert / safe_npk
    row["P_Ratio"] = p_fert / safe_npk
    row["K_Ratio"] = k_fert / safe_npk
    row["Dominant_Soil_Percent"] = 70.0

    if "Rainfall_Annual" in row.columns:
        row["Rainfall_Annual"] = annual_rain
    if "Rainfall_Kharif_Monsoon" in row.columns:
        row["Rainfall_Kharif_Monsoon"] = monsoon_rain
    if "Moisture_Index" in row.columns:
        row["Moisture_Index"] = monsoon_rain * (1.0 + 0.5 * is_irrigated)

    # Distribute rainfall across monsoon months if present
    for m in ["Rainfall_Jun", "Rainfall_Jul", "Rainfall_Aug", "Rainfall_Sep"]:
        if m in row.columns:
            row[m] = monsoon_rain / 4.0

    # Categorical flags
    for prefix, name in [
        ("Crop_", crop),
        ("State Name_", state),
        ("Dominant_Soil_Type_", soil),
    ]:
        col = f"{prefix}{name}"
        if col in row.columns:
            row[col] = 1.0

    # Predict
    pred_kg_ha = max(0.0, float(model.predict(row)[0]))
    quintals_per_acre = pred_kg_ha / 247.105
    total_quintals = quintals_per_acre * acres

    print("\n" + "=" * 55)
    print(" PREDICTION RESULT")
    print("=" * 55)
    print(f" Specialized Engine:    {fam_name.upper()}")
    print(f" Crop:                  {crop}")
    print(f" State:                 {state}")
    print(f" Soil Type:             {soil}")
    print(f" Field Status:          {'Irrigated' if crop_irr > 0 else 'Rainfed'}")
    print(f" N-P-K Applied:         {n_fert:.1f} - {p_fert:.1f} - {k_fert:.1f} kg/ha")
    print("-" * 55)
    print(f" Model Output (Metric): {pred_kg_ha:,.2f} kg/ha ({pred_kg_ha / 1000.0:,.2f} Tonnes/ha)")
    print(f" Farmer Yield Rate:     {quintals_per_acre:,.2f} Quintals / Acre")
    print(f" Total Expected Yield:  {total_quintals:,.1f} Quintals (for {acres} Acres)")
    print("=" * 55)

    again = input("\nTest another scenario? (y/n): ").strip().lower()
    if again != "y":
        print("\nExiting. Done!")
        break