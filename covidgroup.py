import pandas as pd


df = pd.read_csv("Data/us_counties_covid19_daily.csv")

# Group by county name and sum up total cases
# If there's a "state" column too, you can group by ["county", "state"] for clarity
total_cases_by_county = df.groupby(["state", "county", "fips"])["cases"].sum().reset_index()

#Rename the column for clarity
total_cases_by_county.rename(columns={"cases": "total_cases", "fips": "COUNTYFIPS"}, inplace=True)

#Sort descending by case count
total_cases_by_county = total_cases_by_county.sort_values(by="total_cases", ascending=False)

#Save to new .csv
total_cases_by_county.to_csv("Data/total_cases_by_county.csv", index=False)

#To check if executed correctly
print("✅ Done! Output saved to 'total_cases_by_county.csv'")
