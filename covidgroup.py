import pandas as pd

# Load the dataset
df = pd.read_csv("Data/us_counties_covid19_daily.csv")

# Ensure the 'date' column is datetime format
df["date"] = pd.to_datetime(df["date"])

# Sort by date so the last entry per group is the latest
df = df.sort_values("date")

# Group by county/state and get the last entry (latest date) for each
latest_cases_by_county = df.groupby(["state", "county", "fips"]).tail(1)

# Rename columns for clarity
latest_cases_by_county = latest_cases_by_county.rename(columns={"cases": "total_cases", "fips": "COUNTYFIPS"})

# Sort by total_cases descending
latest_cases_by_county = latest_cases_by_county.sort_values(by="total_cases", ascending=False)

# Save to CSV
latest_cases_by_county.to_csv("Data/total_cases_by_county.csv", index=False)

# Check
print("✅ Done! Output saved to 'total_cases_by_county.csv'")
