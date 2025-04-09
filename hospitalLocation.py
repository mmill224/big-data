import pandas as pd
import matplotlib.pyplot as plt
# Load the hospital data
hospital_df = pd.read_csv("Data/Hospitals_gdb_771050112109123987.csv")

# Preview the data
print(hospital_df.columns)
print(hospital_df.head())

def clean_hospital_location_data(df):
    # Standardize column names just in case
    df.columns = [col.strip() for col in df.columns]

    # Try finding common coordinate column names
    lat_col = [col for col in df.columns if "lat" in col.lower()]
    lon_col = [col for col in df.columns if "lon" in col.lower() or "lng" in col.lower()]

    if not lat_col or not lon_col:
        raise ValueError("Could not find latitude or longitude columns.")

    lat_col = lat_col[0]
    lon_col = lon_col[0]

    df = df[[lat_col, lon_col]].copy()
    df = df.rename(columns={lat_col: "Latitude", lon_col: "Longitude"})

    # Drop rows with missing or invalid coordinates
    df = df.dropna(subset=["Latitude", "Longitude"])
    df = df[(df["Latitude"] != 0) & (df["Longitude"] != 0)]

    return df

def plot_hospitals(df):
    plt.figure(figsize=(10, 6))
    plt.scatter(df["LONGITUDE"], df["LATITUDE"], alpha=0.6, c='blue', edgecolors='k')
    plt.title("Hospital Locations")
    plt.xlabel("LONGITUDE")
    plt.ylabel("LATITUDE")
    plt.grid(True)
    plt.savefig("hospital_locations.png")
    plt.show()

if __name__ == "__main__":
    clean_hospital_location_data(hospital_df)
    plot_hospitals(hospital_df)

