import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load shapefile with county geometries
counties_gdf = gpd.read_file("Data/cb_2018_us_county_20m.shp")
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
cases_df = pd.read_csv("Data/total_cases_by_county.csv")
cases_df["county"] = cases_df["county"].str.lower()
cases_df["state"] = cases_df["state"].str.lower()

# Merge cases with geometries
merged = counties_gdf.merge(cases_df, on=["state", "county"], how="left")
print(merged["total_cases"].describe())

# Add a log-transformed column (add 1 to avoid log(0))
""" merged["log_total_cases"] = np.log1p(merged["total_cases"])
 """
# Plot using the log scale


# Plot
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
merged.plot(
    column='total_cases',
    ax=ax,
    legend=True,
    cmap='Reds',
    linewidth=0.2,
    edgecolor='0.7',
    vmin=0,
    vmax=400000,  # tweak this based on your data
    missing_kwds={"color": "lightgrey", "label": "No data"}
)
""" merged.plot(
    column='log_total_cases',
    ax=ax,
    legend=True,
    cmap='Reds',
    linewidth=0.2,
    edgecolor='0.7',
    missing_kwds={"color": "lightgrey", "label": "No data"}
) """
ax.set_title("COVID-19 Total Cases by US County", fontsize=16)
ax.axis("off")

plt.tight_layout()
plt.show()
