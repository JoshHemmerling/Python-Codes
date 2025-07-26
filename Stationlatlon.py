import pandas as pd

# Load NOAA ISD station metadata
df = pd.read_csv("isd-history.csv")

# Drop missing lat/lon
df = df.dropna(subset=["LAT", "LON"])

# Filter lat/lon bounding box for WRF d02 domain
df = df[(df["LAT"] >= 25) & (df["LAT"] <= 38)]
df = df[(df["LON"] >= -105) & (df["LON"] <= -90)]

# Convert END year to numeric and filter for stations active in or after 1995
df["END"] = pd.to_numeric(df["END"], errors="coerce")
df = df[df["END"] >= 19950101]

# Create a clean station ID: ICAO preferred, fallback to USAF-WBAN
df["station_id"] = df.apply(
    lambda row: row["ICAO"] if pd.notna(row["ICAO"]) else f"{int(row['USAF']):06d}-{int(row['WBAN']):05d}",
    axis=1
)

# Select relevant columns
df_out = df[["station_id", "STATION NAME", "LAT", "LON"]].rename(columns={"STATION NAME": "NAME"})

# Output formatted list
with open("stations_formatted.py", "w") as f:
    f.write("stations_list = [\n")
    for _, row in df_out.iterrows():
        f.write(f'    ("{row["station_id"]}", "{row["NAME"]}", {row["LAT"]:.3f}, {row["LON"]:.3f}),\n')
    f.write("]\n")

print(f"✅ Saved {len(df_out)} stations in Python list format to stations_formatted.py")
