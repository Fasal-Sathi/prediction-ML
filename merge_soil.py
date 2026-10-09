import re
import numpy as np
import pandas as pd

print("1/4 Loading base dataset and soil dataset...")
df_base = pd.read_csv("Model2_yield_rain_irr_fert.csv")
df_soil = pd.read_csv("soil.csv")

print(f"Base dataset rows: {len(df_base)}")
print(f"Soil dataset rows: {len(df_soil)}")


# 2. Parsing logic to extract dominant soil type and its percentage
def parse_dominant_soil(val):
    if pd.isna(val) or not isinstance(val, str):
        return pd.Series([np.nan, np.nan])

    val = val.strip()
    if val in ["", "-1", "Not Applicable", "NA"]:
        return pd.Series([np.nan, np.nan])

    # Pattern matches entries like: "LOAMY ALFISOLS - 60%"
    # Splitting by semicolon
    parts = val.split(";")
    best_soil = np.nan
    best_pct = -1.0

    for part in parts:
        match = re.search(r"^(.*?)\s*-\s*(\d+(?:\.\d+)?)\s*%", part.strip())
        if match:
            soil_name = match.group(1).strip().upper()
            pct = float(match.group(2))
            if pct > best_pct:
                best_pct = pct
                best_soil = soil_name

    if best_pct >= 0:
        return pd.Series([best_soil, best_pct])
    else:
        # Fallback if no percentage delimiter was matched
        return pd.Series([val.strip().upper(), np.nan])


print(
    "2/4 Engineering 'Dominant_Soil_Type' and 'Dominant_Soil_Percent' features..."
)
soil_col = "SOIL TYPE PERCENT (Percent)"
df_soil[["Dominant_Soil_Type", "Dominant_Soil_Percent"]] = df_soil[
    soil_col
].apply(parse_dominant_soil)

# Standardize minor naming variants (e.g., ALFISOL vs ALFISOLS)
df_soil["Dominant_Soil_Type"] = df_soil["Dominant_Soil_Type"].replace(
    {"LOAMY ALFISOL": "LOAMY ALFISOLS"}
)

# Rename the raw column for clarity
df_soil = df_soil.rename(columns={soil_col: "Soil_Type_Raw"})

# Retain only unique district-level mapping
soil_features = df_soil[
    [
        "Dist Code",
        "Dominant_Soil_Type",
        "Dominant_Soil_Percent",
        "Soil_Type_Raw",
    ]
].drop_duplicates(subset=["Dist Code"])

print("3/4 Merging soil features into master dataset on 'Dist Code'...")
df_merged = pd.merge(df_base, soil_features, on="Dist Code", how="left")

output_filename = "Model2_yield_rain_irr_fert_soil.csv"
print(f"4/4 Exporting final master dataset to {output_filename}...")
df_merged.to_csv(output_filename, index=False)

print("\n--- Summary ---")
print(f"Saved: {output_filename}")
print(f"Total Rows: {len(df_merged)}")
print(f"Total Columns: {df_merged.shape[1]}")
print(
    f"Dominant Soil Coverage: {df_merged['Dominant_Soil_Type'].notnull().sum()} rows ({df_merged['Dominant_Soil_Type'].notnull().sum() / len(df_merged) * 100:.2f}%)"
)
print("\nTop 5 Most Prevalent Dominant Soil Types:")
print(df_merged["Dominant_Soil_Type"].value_counts().head(5))