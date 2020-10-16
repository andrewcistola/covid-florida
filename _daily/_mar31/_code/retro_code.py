# COVID Florida

## Alachua County Temporal Analysis

### Import Standard Libraries
import os # Inlcuded in every script DC!
import pandas as pd # Incldued in every code script for DC!
import numpy as np # Incldued in every code script for DC!
import scipy as st # Incldued in every code script for DC!

### Set working directory to subproject folder
os.chdir("C:/Users/drewc/GitHub/covid-florida/_mar31") # Set wd to project repository

#################### Break ####################

# Section A: Create Table of Confirmed Cases by County and Day Count
print("Section A: Start") # Print result

## Step 1: Import Libraries and Data

### Import Data
df_case = pd.read_csv("_data/cases_stage.csv", encoding = "ISO-8859-1") # Import dataset saved as csv in _data folder

### Verify CMS
df_case.info() # Get class, memory, and column info: names, data types, obs.
df_case.head() # Print first 5 observations

## Step 2: Prepare Data for Analysis

### Create Daily Count of New Cases
df_filter = df_case.filter(["CaseID", "County", "DateCaseCounted"]) # Keep only selected columns
df_group = df_filter.groupby(["County", "DateCaseCounted"], as_index = False).count() # Group data By Columns and Sum
df_rename = df_group.rename(columns = {"DateCaseCounted": "Date", "CaseID": "Cases"}) # Rename column
df_add = pd.DataFrame([["Alachua", "2020-03-06 00:00:00", 0]], columns = ["County", "Date", "Cases"]) # Add mArch 6 date where no cases were reported statewide
df_stack = pd.concat([df_add, df_rename]) # Combine rows with same columns
df_stack["Date"] = df_stack["Date"].astype("datetime64") # Change date type of column in data frame
df_sort = df_stack.sort_values(by = ["Date"], ascending = True) # Sort Columns by Value

### Create Wide Dataset for All Dates and Counties
df_wide = df_sort.pivot_table(index = "County", columns = "Date", values = "Cases") # Pivot from Long to Wide Format
df_flco = pd.read_csv("_data/fl_counties.csv", encoding = "ISO-8859-1") # Import dataset with county names saved as csv in _data folder
df_join = pd.merge(df_wide, df_flco, on = "County", how = "outer") # Join by column and add counties without confirmed cases
df_join.loc[:, df_join.columns != "County"] = df_join.loc[:, df_join.columns != "County"].fillna(0).astype(np.float64) # Remove NA and change to int64 zeros
df_join.loc["Total"] = df_join.sum() # Add row for column sum anmed "Total"
df_join.loc["Total", "County"] = "Florida" # Rename total Row as Florida
df_index = df_join.reset_index(drop = True) # Reset index as County variable

### Create data frame with dates in order 
cols_to_order = ["County"] # Create object of non date formatted columns
new_columns = cols_to_order + (df_index.columns.drop(cols_to_order).tolist()) # Create object of columns ordered by date following non date columns
df_order = df_index[new_columns] # Create new data frame with columns in order by date

### Rename dates as day count from first confirmed case
l_days = ["County"] # Save "County" as a list with one value
for x in range(len(df_order.columns) - 1): l_days.append("Day" + str(x)) 

#### Hit Enter in Terminal Manually ####

df_order.columns = l_days # Add list of County and Days as Column names for ordered data frame
df_daily = df_order.set_index("County") # Reset index as County variable and rename to daily

### Verify
df_daily.to_csv(r"_data/daily_raw.csv") # Export to csv for an easy to read table

# End Section
print("THE END") # Print result

#################### Break ####################

# Section B: Create Table of Confirmed Cases by County and Day Count
print("Section B: Start") # Print result

## Step 1: Import Libraries and Data

### Import Libraries
import math as mt # Basic Math library
import matplotlib.pyplot as plt # Standard graphing library

### Import Data
df_daily = pd.read_csv("_data/daily_raw.csv", encoding = "ISO-8859-1") # Import dataset saved as csv in _data folder

### Verify CMS
df_daily.info() # Get class, memory, and column info: names, data types, obs.
df_daily.head() # Print first 5 observations

## Step 2: Prepare Data for Analysis

### Connect Wide Format with Week and Day Count
df_long = pd.melt(df_daily, id_vars = ["County"]) # Convert daily table from Long to Wide with columns for County and Day
df_rename = df_long.rename(columns = {"variable": "Day", "value": "Cases"}) # Rename columns
df_rename["Day"] = df_rename["Day"].str.strip("Day") # Remove Day String value
df_rename["Day"] = df_rename["Day"].astype("int64") # Convert Day value to integer
df_rename["Cases"] = df_rename["Cases"].astype("int64") # Convert Day value to integer

### Calculate Population Adjusted Incidence Rate (per 100k)
df_flpop = pd.read_csv("_data/fl_population.csv", encoding = "ISO-8859-1") # Import dataset saved as csv in _data folder of 2018 popualtion estimate by County from FlCHarts
df_join = pd.merge(df_rename, df_flpop, on = "County", how = "inner") # Join by column assign population count
df_sort = df_join.sort_values(by = ["County", "Day"], ascending = True) # Sort Columns by Value
df_sort["Rate"] = df_sort["Cases"]/df_sort["Population"]*100000 # Calculate incidence rate dividing daily new cases by population count and multiplying by 100,000

### Calculate Logarithm of Counts
df_sort["Log"] = np.log(df_sort["Cases"]).replace(-np.inf, np.nan).replace(np.nan, 0)

### Verify
df_day = df_sort # Rename

### Create Top 10 County Specific Data
df_fl = df_day[df_day["County"].str.contains("Florida")]
df_al = df_day[df_day["County"].str.contains("Alachua")]
df_br = df_day[df_day["County"].str.contains("Broward")]
df_da = df_day[df_day["County"].str.contains("Dade")]
df_pb = df_day[df_day["County"].str.contains("Palm Beach")]
df_hb = df_day[df_day["County"].str.contains("Hillsborough")]
df_or = df_day[df_day["County"].str.contains("Orange")]
df_pn = df_day[df_day["County"].str.contains("Pinellas")]
df_dv = df_day[df_day["County"].str.contains("Duval")]
df_ln = df_day[df_day["County"].str.contains("Leon")]

### Create Cumulative Totals for Counts
df_fl["TotalCount"] = df_fl["Cases"].cumsum()
df_al["TotalCount"] = df_al["Cases"].cumsum()
df_br["TotalCount"] = df_br["Cases"].cumsum()
df_da["TotalCount"] = df_da["Cases"].cumsum()
df_pb["TotalCount"] = df_pb["Cases"].cumsum()
df_hb["TotalCount"] = df_hb["Cases"].cumsum()
df_or["TotalCount"] = df_or["Cases"].cumsum()
df_pn["TotalCount"] = df_pn["Cases"].cumsum()
df_dv["TotalCount"] = df_dv["Cases"].cumsum()
df_ln["TotalCount"] = df_ln["Cases"].cumsum()

### Create Cumulative Totals for Rates
df_fl["TotalRate"] = df_fl["Rate"].cumsum()
df_al["TotalRate"] = df_al["Rate"].cumsum()
df_br["TotalRate"] = df_br["Rate"].cumsum()
df_da["TotalRate"] = df_da["Rate"].cumsum()
df_pb["TotalRate"] = df_pb["Rate"].cumsum()
df_hb["TotalRate"] = df_hb["Rate"].cumsum()
df_or["TotalRate"] = df_or["Rate"].cumsum()
df_pn["TotalRate"] = df_pn["Rate"].cumsum()
df_dv["TotalRate"] = df_dv["Rate"].cumsum()
df_ln["TotalRate"] = df_ln["Rate"].cumsum()

### Create Cumulative Totals for Rates
df_fl["TotalLog"] = df_fl["Log"].cumsum()
df_al["TotalLog"] = df_al["Log"].cumsum()
df_br["TotalLog"] = df_br["Log"].cumsum()
df_da["TotalLog"] = df_da["Log"].cumsum()
df_pb["TotalLog"] = df_pb["Log"].cumsum()
df_hb["TotalLog"] = df_hb["Log"].cumsum()
df_or["TotalLog"] = df_or["Log"].cumsum()
df_pn["TotalLog"] = df_pn["Log"].cumsum()
df_dv["TotalLog"] = df_dv["Log"].cumsum()
df_ln["TotalLog"] = df_ln["Log"].cumsum()

### Create Doubling Time for Counts
df_fl["Double"] = mt.log(2) / df_fl["Cases"].pct_change()
df_al["Double"] = mt.log(2) / df_al["Cases"].pct_change()
df_br["Double"] = mt.log(2) / df_br["Cases"].pct_change()
df_da["Double"] = mt.log(2) / df_da["Cases"].pct_change()
df_pb["Double"] = mt.log(2) / df_pb["Cases"].pct_change()
df_hb["Double"] = mt.log(2) / df_hb["Cases"].pct_change()
df_or["Double"] = mt.log(2) / df_or["Cases"].pct_change()
df_pn["Double"] = mt.log(2) / df_pn["Cases"].pct_change()
df_dv["Double"] = mt.log(2) / df_dv["Cases"].pct_change()
df_ln["Double"] = mt.log(2) / df_ln["Cases"].pct_change()

### Verify
df_fl.to_csv(r"_data/florida_raw.csv") # Clean in excel and select variable
df_al.to_csv(r"_data/alachua_raw.csv") # Clean in excel and select variable

## Step 4: Create Visuals and Outputs

## Alachua County

### Create Barplot for New Cases by Count
plt.figure()
x = np.arange(len(df_al.Day))
plt.bar((x), df_al.Cases, color = 'g', width = 0.4)
plt.xticks((x), df_al["Day"], rotation = 90)
plt.ylabel("Newly Confirmed Case Count")
plt.xlabel("Day Number (Day 0 is March 2, 2020)")
plt.legend(["Alachua"])
plt.title("Florida DOH Alachua Confirmed COVID-19 Cases (As of March 22, 2020 10:18AM)")
plt.savefig("_fig/alachua_daily_count.jpeg", bbox_inches = "tight")

### Create Barplot for New Cases by Rate
plt.figure()
x = np.arange(len(df_al.Day))
plt.bar((x), df_al.Rate, color = 'g', width = 0.4)
plt.xticks((x), df_al["Day"], rotation = 90)
plt.ylabel("Newly Confirmed Case Incidence Rate (per 100k)")
plt.xlabel("Day Number (Day 0 is March 2, 2020)")
plt.legend(["Alachua"])
plt.title("Florida DOH Alachua County Confirmed COVID-19 Cases (As of March 22, 2020 10:18AM)")
plt.savefig("_fig/alachua_daily_rate.jpeg", bbox_inches = "tight")

## Florida 

### Create Barplot for New Cases by Count
plt.figure()
x = np.arange(len(df_fl.Day))
plt.bar((x), df_fl.Cases, color = 'b', width = 0.4)
plt.xticks((x), df_fl["Day"], rotation = 90)
plt.ylabel("Newly Confirmed Case Count")
plt.xlabel("Day Number (Day 0 is March 2, 2020)")
plt.legend(["Florida"])
plt.title("Florida DOH Statewide Confirmed COVID-19 Cases (As of March 24, 2020 10:18AM)")
plt.savefig("_fig/florida_daily_count.jpeg", bbox_inches = "tight")

### Create Barplot for New Cases by Rate
plt.figure()
x = np.arange(len(df_fl.Day))
plt.bar((x), df_fl.Rate, color = 'b', width = 0.4)
plt.xticks((x), df_fl["Day"], rotation = 90)
plt.ylabel("Newly Confirmed Case Incidence Rate (per 100k)")
plt.xlabel("Day Number (Day 0 is March 2, 2020)")
plt.legend(["Florida"])
plt.title("Florida DOH Statewide Confirmed COVID-19 Cases (As of March 24, 2020 10:18AM)")
plt.savefig("_fig/florida_daily_rate.jpeg", bbox_inches = "tight")

### Create Multiple Line Plot for Total by Count
plt.figure()
x = np.arange(len(df_al.Day))
plt.plot(df_al.Day, df_al.TotalCount, color = "g")
plt.plot(df_br.Day, df_br.TotalCount, color = "b")
plt.plot(df_da.Day, df_da.TotalCount, color = "r")
plt.plot(df_dv.Day, df_dv.TotalCount, color = "0.75")
plt.plot(df_hb.Day, df_hb.TotalCount, color = "m")
plt.plot(df_ln.Day, df_ln.TotalCount, color = "0.25")
plt.plot(df_or.Day, df_or.TotalCount, color = "y")
plt.plot(df_pb.Day, df_pb.TotalCount, color = "c")
plt.plot(df_pn.Day, df_pn.TotalCount, color = "k")
plt.ylabel("Total Confirmed Case Count")
plt.xlabel("Day Number (Day 0 is March 2, 2020)")
plt.xticks((x), df_al["Day"], rotation = 90)
plt.legend(["Florida", "Alachua", "Broward", "Dade", "Duval", "Hillsborough", "Leon", "Orange", "Palm Beach", "Pinellas"])
plt.title("Florida DOH Confirmed COVID-19 Cases by County (As of March 24, 2020 10:18AM)")
plt.savefig("_fig/county_total_count.jpeg", bbox_inches = "tight")

### Create Multiple Line Plot for Total by Rate
plt.figure()
x = np.arange(len(df_al.Day))
plt.plot(df_al.Day, df_al.TotalRate, color = "g")
plt.plot(df_br.Day, df_br.TotalRate, color = "b")
plt.plot(df_da.Day, df_da.TotalRate, color = "r")
plt.plot(df_dv.Day, df_dv.TotalRate, color = "0.75")
plt.plot(df_hb.Day, df_hb.TotalRate, color = "m")
plt.plot(df_ln.Day, df_ln.TotalRate, color = "0.25")
plt.plot(df_or.Day, df_or.TotalRate, color = "y")
plt.plot(df_pb.Day, df_pb.TotalRate, color = "c")
plt.plot(df_pn.Day, df_pn.TotalRate, color = "k")
plt.ylabel("Total Confirmed Case Incidence Rate (per 100k)")
plt.xlabel("Day Number (Day 0 is March 2, 2020)")
plt.xticks((x), df_al["Day"], rotation = 90)
plt.legend(["Florida", "Alachua", "Broward", "Dade", "Duval", "Hillsborough", "Leon", "Orange", "Palm Beach", "Pinellas"])
plt.title("Florida DOH Confirmed COVID-19 Cases by County (As of March 24, 2020 10:18AM)")
plt.savefig("_fig/county_total_rate.jpeg", bbox_inches = "tight")

### Create Multiple Line Plot for Total by Rate
plt.figure()
x = np.arange(len(df_al.Day))
plt.plot(df_al.Day, df_fl.TotalRate, color = "b")
plt.ylabel("Total Confirmed Case Incidence Rate (per 100k)")
plt.xlabel("Day Number (Day 0 is March 2, 2020)")
plt.xticks((x), df_al["Day"], rotation = 90)
plt.legend(["Florida", "Alachua", "Broward", "Dade", "Duval", "Hillsborough", "Leon", "Orange", "Palm Beach", "Pinellas"])
plt.title("Florida DOH Confirmed COVID-19 Cases by County (As of March 24, 2020 10:18AM)")
plt.savefig("_fig/county_total_rate.jpeg", bbox_inches = "tight")

## Verify
plt.show() # Show created plots

# End Section
print("THE END") # Print result