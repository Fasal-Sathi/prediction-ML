import pandas as pd
import numpy as np

print("1/4 Loading base dataset and fertilizer dataset...")
df_base = pd.read_csv("Model2_yield_rain_irrigation.csv")
df_fert = pd.read_csv("fertilizer.csv")

print(f"Base dataset records: {len(df_base)}")
print(f"Raw fertilizer records: {len(df_fert)}")

# 2. Standardize State Name
df_fert["State Name"] = df_fert["State Name"].replace({"Orissa": "Odisha"})

# 3. Clean column names
fert_col_rename = {
    "NITROGEN CONSUMPTION (tons)": "Fertilizer_N_tons",
    "NITROGEN PER HA OF GCA (Kg per ha)": "Fertilizer_N_kg_per_ha",
    "PHOSPHATE CONSUMPTION (tons)": "Fertilizer_P_tons",
    "PHOSPHATE PER HA OF GCA (Kg per ha)": "Fertilizer_P_kg_per_ha",
    "POTASH CONSUMPTION (tons)": "Fertilizer_K_tons",
    "POTASH PER HA OF GCA (Kg per ha)": "Fertilizer_K_kg_per_ha",
    "TOTAL CONSUMPTION (tons)": "Fertilizer_NPK_Total_tons",
    "TOTAL PER HA OF GCA (Kg per ha)": "Fertilizer_NPK_Total_kg_per_ha"
}
df_fert = df_fert.rename(columns=fert_col_rename)

numeric_fert_cols = list(fert_col_rename.values())

print("2/4 Cleaning sentinel values (-1 -> NaN)...")
for col in numeric_fert_cols:
    df_fert[col] = df_fert[col].apply(lambda x: np.nan if x < 0 else x)

# 4. Prepare merge keys and columns
merge_cols = ["Dist Code", "Year"] + numeric_fert_cols
df_fert_clean = df_fert[merge_cols].drop_duplicates(subset=["Dist Code", "Year"])

print("3/4 Merging fertilizer features into base dataset...")
df_merged = pd.merge(
    df_base,
    df_fert_clean,
    on=["Dist Code", "Year"],
    how="left"
)

output_filename = "Model2_yield_rain_irr_fert.csv"
print(f"4/4 Exporting to {output_filename}...")
df_merged.to_csv(output_filename, index=False)

print("\n--- Summary ---")
print(f"Saved: {output_filename}")
print(f"Total Rows: {len(df_merged)}")
print(f"Total Columns: {df_merged.shape[1]}")
print(f"Fertilizer NPK kg/ha coverage (non-nulls): {df_merged['Fertilizer_NPK_Total_kg_per_ha'].notnull().sum()} rows ({df_merged['Fertilizer_NPK_Total_kg_per_ha'].notnull().sum() / len(df_merged) * 100:.2f}%)")