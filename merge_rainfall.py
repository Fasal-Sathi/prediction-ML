import pandas as pd

print("1/4 Loading datasets...")
df_yield = pd.read_csv("Model2_yield_clean_long.csv")
df_rain = pd.read_csv("perciption new.csv")

print("2/4 Standardizing names...")
df_rain["State Name"] = df_rain["State Name"].replace({"Orissa": "Odisha"})

clean_precip_names = {
    "JANUARY PERCIPITATION (Millimeters)": "Rainfall_Jan",
    "FEBRUARY PERCIPITATION (Millimeters)": "Rainfall_Feb",
    "MARCH PERCIPITATION (Millimeters)": "Rainfall_Mar",
    "APRIL PERCIPITATION (Millimeters)": "Rainfall_Apr",
    "MAY PERCIPITATION (Millimeters)": "Rainfall_May",
    "JUNE PERCIPITATION (Millimeters)": "Rainfall_Jun",
    "JULY PERCIPITATION (Millimeters)": "Rainfall_Jul",
    "AUGUST PERCIPITATION (Millimeters)": "Rainfall_Aug",
    "SEPTEMBER PERCIPITATION (Millimeters)": "Rainfall_Sep",
    "OCTOBER PERCIPITATION (Millimeters)": "Rainfall_Oct",
    "NOVEMBER PERCIPITATION (Millimeters)": "Rainfall_Nov",
    "DECEMBER PERCIPITATION (Millimeters)": "Rainfall_Dec",
}
df_rain = df_rain.rename(columns=clean_precip_names)

print("3/4 Engineering rainfall features...")
months = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]
df_rain["Rainfall_Annual"] = df_rain[[f"Rainfall_{m}" for m in months]].sum(
    axis=1
)
df_rain["Rainfall_Kharif_Monsoon"] = df_rain[
    ["Rainfall_Jun", "Rainfall_Jul", "Rainfall_Aug", "Rainfall_Sep"]
].sum(axis=1)
df_rain["Rainfall_Rabi_PostMonsoon"] = df_rain[
    ["Rainfall_Oct", "Rainfall_Nov", "Rainfall_Dec"]
].sum(axis=1)
df_rain["Rainfall_Winter"] = df_rain[["Rainfall_Jan", "Rainfall_Feb"]].sum(
    axis=1
)
df_rain["Rainfall_Summer_PreMonsoon"] = df_rain[
    ["Rainfall_Mar", "Rainfall_Apr", "Rainfall_May"]
].sum(axis=1)

print("4/4 Merging and exporting...")
rain_features = [
    c
    for c in df_rain.columns
    if c not in ["State Code", "State Name", "Dist Name"]
]
df_merged = pd.merge(
    df_yield, df_rain[rain_features], on=["Dist Code", "Year"], how="inner"
)

output_filename = "Model2_yield_with_rainfall.csv"
df_merged.to_csv(output_filename, index=False)

print(f"\nSuccess! Generated: {output_filename}")
print(f"Total Rows: {len(df_merged)} | Total Columns: {df_merged.shape[1]}")