# COVID Florida

## Alachua County Temporal Analysis

### Import Standard Libraries
import os # Inlcuded in every script DC!
import pandas as pd # Incldued in every code script for DC!
import numpy as np # Incldued in every code script for DC!
import scipy as st # Incldued in every code script for DC!

### Set working directory to subproject folder
os.chdir("C:/Users/drewc/GitHub/covid-florida/_mar241012") # Set wd to project repository

#################### Break ####################

# Section A: Day and Week Temporal Analysis
print("Section Start") # Print result

## Step 1: Import Libraries and Data

### Import Libraries Specific for this Analysis
import matplotlib.pyplot as plt

### Import Data
df_case = pd.read_csv("_data/cases_stage.csv", encoding = "ISO-8859-1") # Import dataset saved as csv in _data folder

### Verify CMS
df_case.info() # Get class, memory, and column info: names, data types, obs.
df_case.head() # Print first 5 observations

## Step 2: Prepare Data for Analysis

### Create Daily Count of New Cases

df_filter = df_case.filter(["Case", "County", "DateCounted"]) # Keep only selected columns

df_group = df_filter.groupby(["County", "DateCounted"], as_index = False).count() # Group data By Columns and Sum

df_rename = df_group.rename(columns = {"DateCounted": "Date", "Case": "Cases"}) # Rename column

df_add = pd.DataFrame([["Alachua", "2020-03-06 00:00:00", 0]], columns = ["County", "Date", "Cases"])

df_rename["Date"] = df_rename["Date"].astype("datetime64") # Change data type of column in data frame

df_sort = df_rename.sort_values(by = ["Date"], ascending = True) # Sort Columns by Value

df_wide = df_sort.pivot_table(index = "County", columns = "Date", values = "Cases") # Pivot from Long to Wide Format

### Create Wide Dataset for All Days and Counties

df_flco = pd.read_csv("_data/fl_counties.csv", encoding = "ISO-8859-1") # Import dataset saved as csv in _data folder

df_join = pd.merge(df_wide, df_flco, on = "County", how = "outer") # Join by column while keeping only items that exist in both, select outer or left for other options

df_join.loc[:, df_join.columns != "County"] = df_join.loc[:, df_join.columns != "County"].fillna(0).astype(np.float64) # Remove NA and change to int64 zeros

df_join.loc["Total"] = df_join.sum()

df_join.loc["Total", "County"] = "Florida"

df_index = df_join.reset_index(drop = True)

cols_to_order = ["County"]
new_columns = cols_to_order + (df_index.columns.drop(cols_to_order).tolist())
df_order = df_index[new_columns]

l_days = ["County"]
for x in range(len(df_order.columns) - 1): l_days.append("Day" + str(x))
df_order.columns = l_days

df_daily = df_order

df_daily.to_csv(r"_data/daily_raw.csv") # Clean in excel and select variable

### Connect Wide Format with Week and Day Count

df_long = pd.melt(df_daily, id_vars = ["County"])

df_rename = df_long.rename(columns = {"variable": "Day", "value": "Cases"}) # Rename column

df_rename["Day"] = df_rename["Day"].str.strip("Day")
df_rename["Day"] = df_rename["Day"].astype("int64")

df_dywk = pd.read_csv("_data/weeks_days.csv", encoding = "ISO-8859-1") # Import dataset saved as csv in _data folder
df_join = pd.merge(df_rename, df_dywk, on = "Day", how = "left") # Join by column while keeping only items that exist in both, select outer or left for other options

df_flpop = pd.read_csv("_data/fl_population.csv", encoding = "ISO-8859-1") # Import dataset saved as csv in _data folder
df_join2 = pd.merge(df_join, df_flpop, on = "County", how = "left") # Join by column while keeping only items that exist in both, select outer or left for other options

df_join2["Rate"] = df_join2["Cases"]/df_join2["Population"]*100000

## Retrospective by Week and County

df_filter = df_join2.filter(["Cases", "Rate", "County", "Week"]) # Keep only selected columns

df_group = df_filter.groupby(["County", "Week"], as_index = False).sum() # Group data By Columns and Sum

df_sort = df_group.sort_values(by = ["County", "Week"], ascending = True) # Sort Columns by Value

df_sort["Change"] = df_sort["Cases"] - df_sort["Cases"].shift(1)

df_sort["Double"] = np.where(df_sort["Cases"] >= df_sort["Cases"].shift(1), "Yes", "No")

df_sort["Double"][df_sort["Week"] == 1] = "No"
df_sort["Double"][df_sort["Cases"] == 0] = "No"
df_sort["Change"][df_sort["Week"] == 1] = 0

df_sort["Increase"] = df_sort["Cases"] / df_sort["Change"] * 100

df_week = df_sort

df_week.to_csv(r"_data/weekly_raw.csv") # Clean in excel and select variable

## Retrospective by Day for Alachua County

df_sort = df_join2.sort_values(by = ["County", "Day"], ascending = True) # Sort Columns by Value

df_sort["Change"] = df_sort["Cases"] - df_sort["Cases"].shift(1)
df_sort["Double"] = np.where(df_sort["Cases"] >= df_sort["Cases"].shift(1), "Yes", "No")

df_sort["Double"][df_sort["Day"] == 0] = "No"
df_sort["Double"][df_sort["Cases"] == 0] = "No"
df_sort["Change"][df_sort["Day"] == 0] = 0

df_day = df_sort
df_day.to_csv(r"_data/day_raw.csv") # Clean in excel and select variable

df_br = df_day[df_day["County"].str.contains("Broward")]
df_br["Total"] = df_br["Rate"].cumsum()

df_al = df_day[df_day["County"].str.contains("Alachua")]
df_al["Total"] = df_al["Rate"].cumsum()
df_al.to_csv(r"_data/alachua_raw.csv") # Clean in excel and select variable

df_fl = df_day[df_day["County"].str.contains("Florida")]
df_fl["Total"] = df_fl["Rate"].cumsum()
df_fl.to_csv(r"_data/florida_raw.csv") # Clean in excel and select variable

## Step 4: Create Visuals and Outputs

## Create Barplot for Alachua County
plt.figure()
x = np.arange(len(df_al.County))
plt.bar((x - 0.2), df_al.Rate, color = 'g', width = 0.4)
plt.bar((x + 0.2), df_br.Rate, color = 'b', width = 0.4)
plt.xticks((x), df_al["Day"], rotation = 90)
plt.ylabel("Newly Confirmed Cases per 100,000")
plt.xlabel("Day Number (Day 0 is March 2, 2020)")
plt.legend(["Alachua", "Broward"])
plt.title("Florida DOH Confirmed COVID-19 Cases (As of March 22, 2020 6:21PM)")
plt.savefig("_fig/new_daily.jpeg", bbox_inches = "tight")

### Create Multiple Line Plot for Percent Change
plt.figure()
x = np.arange(len(df_al.County))
plt.plot(df_al.Day, df_al.Total, color = "g")
plt.plot(df_br.Day, df_br.Total, color = "b")
plt.plot(df_fl.Day, df_fl.Total, color = "r")
plt.ylabel("Total Confirmed Cases per 100,000")
plt.xlabel("Day Number (Day 0 is March 2, 2020)")
plt.xticks((x), df_al["Day"], rotation = 90)
plt.legend(["Alachua", "Broward", "Florida"])
plt.title("Florida DOH Confirmed COVID-19 Cases (As of March 22, 2020 6:21PM)")
plt.savefig("_fig/total_daily.jpeg", bbox_inches = "tight")

## Verify
plt.show() # Show created plots

# End Section
print("THE END") # Print result