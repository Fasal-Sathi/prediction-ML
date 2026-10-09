import numpy as np
import pandas as pd

print("1/5 Loading files...")
df_base = pd.read_csv("Model2_yield_with_rainfall.csv")
df_irr = pd.read_csv("irrigation.csv.csv")

print(f"Base dataset records: {len(df_base)}")
print(f"Raw irrigation records: {len(df_irr)}")

# 2. Standardize State Name
df_irr["State Name"] = df_irr["State Name"].replace({"Orissa": "Odisha"})

# 3. Map crop-specific irrigation columns to match our standard Crop names
crop_irr_map = {
    "RICE IRRIGATED AREA (1000 ha)": "RICE",
    "WHEAT IRRIGATED AREA (1000 ha)": "WHEAT",
    "MAIZE IRRIGATED AREA (1000 ha)": "MAIZE",
    "BARLEY IRRIGATED AREA (1000 ha)": "BARLEY",
    "CHICKPEA IRRIGATED AREA (1000 ha)": "CHICKPEA",
    "MINOR PULSES IRRIGATED AREA (1000 ha)": "MINOR PULSES",
    "GROUNDNUT IRRIGATED AREA (1000 ha)": "GROUNDNUT",
    "LINSEED IRRIGATED AREA (1000 ha)": "LINSEED",
    "SUGARCANE IRRIGATED AREA (1000 ha)": "SUGARCANE",
}

# Also calculate Total District Irrigated Area across all crops present in this sheet as a general environmental indicator
irr_value_cols = [c for c in df_irr.columns if "IRRIGATED AREA" in c]

# Replace negative sentinel values (-1) with NaN
for col in irr_value_cols:
    df_irr[col] = df_irr[col].apply(lambda x: np.nan if x < 0 else x)

print("2/5 Creating District-level Total Irrigated Area feature...")
# District total irrigated area (sum across available crops for that district-year)
df_irr["Total_District_Irrigated_1000_ha"] = df_irr[irr_value_cols].sum(
    axis=1, min_count=1
)

# Extract district-level summary table (Dist Code, Year -> Total_District_Irrigated)
df_dist_irr = df_irr[
    ["Dist Code", "Year", "Total_District_Irrigated_1000_ha"]
].drop_duplicates()

print("3/5 Melting wide irrigation data into long format...")
# Filter to the keys and the mapped crop columns
id_vars = ["Dist Code", "Year"]
cols_to_keep = id_vars + list(crop_irr_map.keys())
df_irr_subset = df_irr[cols_to_keep]

df_irr_long = df_irr_subset.melt(
    id_vars=id_vars,
    value_vars=list(crop_irr_map.keys()),
    var_name="Irrigation_Col",
    value_name="Crop_Irrigated_Area_1000_ha",
)

# Map back to standard uppercase Crop name
df_irr_long["Crop"] = df_irr_long["Irrigation_Col"].map(crop_irr_map)
df_irr_long = df_irr_long.drop(columns=["Irrigation_Col"])

# Convert 1000 ha to hectares as well for convenience
df_irr_long["Crop_Irrigated_Area_ha"] = (
    df_irr_long["Crop_Irrigated_Area_1000_ha"] * 1000
)

print("4/5 Merging with Base Yield + Rainfall dataset...")
# Left join 1: Merge crop-specific irrigation
df_merged = pd.merge(
    df_base, df_irr_long, on=["Dist Code", "Year", "Crop"], how="left"
)

# Left join 2: Merge total district-level irrigation
df_merged = pd.merge(
    df_merged, df_dist_irr, on=["Dist Code", "Year"], how="left"
)

output_filename = "Model2_yield_rain_irrigation.csv"
print(f"5/5 Exporting to {output_filename}...")
df_merged.to_csv(output_filename, index=False)

print("\n--- Summary ---")
print(f"Saved: {output_filename}")
print(f"Total Rows: {len(df_merged)}")
print(f"Total Columns: {df_merged.shape[1]}")
print(
    f"Crop-specific irrigation coverage (non-nulls): {df_merged['Crop_Irrigated_Area_1000_ha'].notnull().sum()} rows"
)
print(
    f"District-total irrigation coverage (non-nulls): {df_merged['Total_District_Irrigated_1000_ha'].notnull().sum()} rows"
)