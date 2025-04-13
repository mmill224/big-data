import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import sys

start_total = time.time()

# Load shapefile with county geometries
start = time.time()
counties_gdf = gpd.read_file("Data/cb_2018_us_county_20m.shp")
end = time.time()
print(f"Shapefile load time: {end - start:.2f} seconds")
print(counties_gdf.columns)

# State FIPS codes to state names
state_fips_to_name = {
    "01": "alabama", "02": "alaska", "04": "arizona", "05": "arkansas", "06": "california",
    "08": "colorado", "09": "connecticut", "10": "delaware", "11": "district of columbia",
    "12": "florida", "13": "georgia", "15": "hawaii", "16": "idaho", "17": "illinois",
    "18": "indiana", "19": "iowa", "20": "kansas", "21": "kentucky", "22": "louisiana",
    "23": "maine", "24": "maryland", "25": "massachusetts", "26": "michigan", "27": "minnesota",
    "28": "mississippi", "29": "missouri", "30": "montana", "31": "nebraska", "32": "nevada",
    "33": "new hampshire", "34": "new jersey", "35": "new mexico", "36": "new york",
    "37": "north carolina", "38": "north dakota", "39": "ohio", "40": "oklahoma",
    "41": "oregon", "42": "pennsylvania", "44": "rhode island", "45": "south carolina",
    "46": "south dakota", "47": "tennessee", "48": "texas", "49": "utah", "50": "vermont",
    "51": "virginia", "53": "washington", "54": "west virginia", "55": "wisconsin", "56": "wyoming"
}

# Add lowercase state name and county name columns
counties_gdf["state"] = counties_gdf["STATEFP"].map(state_fips_to_name)
counties_gdf["county"] = counties_gdf["NAME"].str.lower()

# Load summarized COVID data
start = time.time()
cases_df = pd.read_csv("Data/total_cases_by_county.csv")
cases_df["county"] = cases_df["county"].str.lower()
cases_df["state"] = cases_df["state"].str.lower()
end = time.time()
print(f"COVID data load time: {end - start:.2f} seconds")

# Merge cases with geometries
start = time.time()
merged = counties_gdf.merge(cases_df, on=["state", "county"], how="left")
end = time.time()
print(f"Merging COVID cases: {end - start:.2f} seconds")
print(merged["total_cases"].describe())

# Add a log-transformed column (add 1 to avoid log(0))
""" merged["log_total_cases"] = np.log1p(merged["total_cases"])
 """
# Plot using the log scale


#Load hospital dataset
start = time.time()
hospitals_df = pd.read_csv("Data/Hospitals_gdb_771050112109123987.csv")

# Normalize county and state names to match
hospitals_df["county"] = hospitals_df["COUNTY"].str.lower()
hospitals_df["state"] = hospitals_df["STATE"].str.lower()

# Count number of hospitals per (state, county)
hospital_counts = hospitals_df.groupby(["state", "county"]).size().reset_index(name="hospital_count")

# Merge hospital counts with COVID case data
merged = merged.merge(hospital_counts, on=["state", "county"], how="left")


# Replace 0 or NaN hospital counts with 1 (to avoid division by 0)
merged["hospital_count"].fillna(1, inplace=True)
end = time.time()
print(f"Hospital data process & merge time: {end - start:.2f} seconds")

# Calculate cases per hospital
merged["cases_per_hospital"] = merged["total_cases"] / merged["hospital_count"]
merged["log_cases_per_hospital"] = np.log1p(merged["cases_per_hospital"])

# Report memory usage
print(f"Memory usage of final GeoDataFrame: {merged.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Plot: Cases per hospital
start = time.time()
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
merged.plot(
    column='log_cases_per_hospital',
    ax=ax,
    legend=True,
    cmap='OrRd',
    linewidth=0.2,
    edgecolor='0.7',
    missing_kwds={"color": "lightgrey", "label": "No data"}
)
ax.set_title("COVID-19 Cases per Hospital by US County", fontsize=16)
ax.axis("off")

plt.tight_layout()

end = time.time()
print(f"Plotting time: {end - start:.2f} seconds")

end_total = time.time()
print(f"Total run time: {end_total - start_total:.2f} seconds")

plt.show()


