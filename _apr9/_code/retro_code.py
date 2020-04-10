# COVID Florida

## Alachua County Temporal Analysis

### Import Standard Libraries
import os # Inlcuded in every script DC!
import pandas as pd # Incldued in every code script for DC!
import numpy as np # Incldued in every code script for DC!
import scipy as st # Incldued in every code script for DC!

### Set working directory to subproject folder
os.chdir("C:/Users/drewc/GitHub/covid-florida") # Set wd to project repository

#################### Break ####################

# Section A: Create Table of Confirmed Cases by County and Day Count
print("Section A: Start") # Print result

## Step 1: Import Libraries and Data

### Import Data
df_case = pd.read_csv("_data/features.csv", encoding = "ISO-8859-1") # Import dataset saved as csv in _data folder

### Verify CMS
df_case.info() # Get class, memory, and column info: names, data types, obs.
df_case.head() # Print first 5 observations

## Step 2: Prepare Data for Analysis

### Create Daily Count of New Cases
df_rename =  df_case.rename(columns = {"ObjectID": "ID", "EventDate": "Date"}) # Rename column
df_group = df_rename.groupby(["County", "Date"], as_index = False).count() # Group data By Columns and Sum
df_add = pd.DataFrame([["Alachua", "3/6/20", 0]], columns = ["County", "Date", "ID"]) # Add mArch 6 date where no cases were reported statewide
df_stack = pd.concat([df_add, df_group]) # Combine rows with same columns
df_stack["Date"] = df_stack["Date"].astype("datetime64") # Change date type of column in data frame
df_sort = df_stack.sort_values(by = ["Date"], ascending = True) # Sort Columns by Value
df_rename =  df_sort.rename(columns = {"ID": "Cases"}) # Rename column

### Create Wide Dataset for All Dates and Counties
df_wide = df_rename.pivot_table(index = "County", columns = "Date", values = "Cases") # Pivot from Long to Wide Format
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

### Remove Day 34 in Excel (Incomplete at time of Report) and save as staged

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
df_daily = pd.read_csv("_data/daily_stage.csv", encoding = "ISO-8859-1") # Import dataset saved as csv in _data folder

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

### Verify
df_fl.to_csv(r"_data/florida_raw.csv") # Clean in excel and select variable
df_al.to_csv(r"_data/alachua_raw.csv") # Clean in excel and select variable

df_fl = pd.read_csv("_apr8/_data/florida_raw.csv", encoding = "ISO-8859-1") # Import dataset saved as csv in _data folder
df_al = pd.read_csv("_apr8/_data/alachua_raw.csv", encoding = "ISO-8859-1") # Import dataset saved as csv in _data folder

## Step 4: Create Visuals and Outputs

## Alachua County

### Create Barplot for New Cases by Count
plt.figure()
x = np.arange(len(df_al.Day))
plt.bar((x), df_al.Cases, color = 'xkcd:neon green', width = 0.4)
plt.xticks((x), df_al["Day"], rotation = 90)
plt.ylabel("Daily Cases Rate per 100k")
plt.xlabel("Day Number")
plt.legend(["Alachua"])
plt.title("Florida DOH Confirmed COVID-19 Cases by Day")
plt.savefig("_apr8/_fig/alachua_daily_count.jpeg", bbox_inches = "tight")

### Create Barplot for New Cases by Rate
plt.figure()
x = np.arange(len(df_al.Day))
plt.bar((x), df_al.Rate, color = 'xkcd:neon green', width = 0.4)
plt.xticks((x), df_al["Day"], rotation = 90)
plt.ylabel("Daily Cases Rate per 100k")
plt.xlabel("Day Number")
plt.legend(["Alachua"])
plt.title("Florida DOH County Confirmed COVID-19 Cases by Day")
plt.savefig("_apr8/_fig/alachua_daily_rate.jpeg", bbox_inches = "tight")

## Florida 

### Create Barplot for New Cases by Count
plt.figure()
x = np.arange(len(df_fl.Day))
plt.bar((x), df_fl.Cases, color = 'xkcd:neon blue', width = 0.4)
plt.xticks((x), df_fl["Day"], rotation = 90)
plt.ylabel("Daily Cases")
plt.xlabel("Day Number")
plt.legend(["Florida"])
plt.title("Florida DOH Confirmed COVID-19 Cases by Day")
plt.savefig("_apr8/_fig/florida_daily_count.jpeg", bbox_inches = "tight")

### Create Barplot for New Cases by Rate
plt.figure()
x = np.arange(len(df_fl.Day))
plt.bar((x), df_fl.Rate, color = 'xkcd:neon blue', width = 0.4)
plt.xticks((x), df_fl["Day"], rotation = 90)
plt.ylabel("Daily Cases Rate per 100k")
plt.xlabel("Day Number")
plt.legend(["Florida"])
plt.title("Florida DOH Confirmed COVID-19 Cases by Day")
plt.savefig("_apr8/_fig/florida_daily_rate.jpeg", bbox_inches = "tight")

## Multi-County

### Create Multiple Line Plot for Total by Count
plt.figure()
x = np.arange(len(df_al.Day))
plt.plot(df_al.Day, df_al.TotalCount, color = "xkcd:neon green")
plt.plot(df_br.Day, df_br.TotalCount, color = "r")
plt.plot(df_da.Day, df_da.TotalCount, color = "y")
plt.plot(df_hb.Day, df_hb.TotalCount, color = "m")
plt.plot(df_or.Day, df_or.TotalCount, color = "g")
plt.plot(df_pb.Day, df_pb.TotalCount, color = "tab:orange")
plt.plot(df_pn.Day, df_pn.TotalCount, color = "tab:purple")
plt.ylabel("Cumulative Cases")
plt.xlabel("Day Number")
plt.xticks((x), df_al["Day"], rotation = 90)
plt.legend(["Alachua", "Broward", "Dade", "Hillsborough", "Orange", "Palm Beach", "Pinellas"])
plt.title("Florida DOH Confirmed COVID-19 Cases by Day")
plt.savefig("_apr8/_fig/county_total_count.jpeg", bbox_inches = "tight")

### Create Multiple Line Plot for Total by Count
plt.figure()
x = np.arange(len(df_al.Day))
plt.plot(df_al.Day, df_al.TotalRate, color = "xkcd:neon green")
plt.plot(df_br.Day, df_br.TotalRate, color = "r")
plt.plot(df_da.Day, df_da.TotalRate, color = "y")
plt.plot(df_hb.Day, df_hb.TotalRate, color = "m")
plt.plot(df_or.Day, df_or.TotalRate, color = "g")
plt.plot(df_pb.Day, df_pb.TotalRate, color = "tab:orange")
plt.plot(df_pn.Day, df_pn.TotalRate, color = "tab:purple")
plt.ylabel("Cumulative Cases Rate per 100k")
plt.xlabel("Day Number")
plt.xticks((x), df_al["Day"], rotation = 90)
plt.legend(["Alachua", "Broward", "Dade", "Hillsborough", "Orange", "Palm Beach", "Pinellas"])
plt.title("Florida DOH Confirmed COVID-19 Cases by Day")
plt.savefig("_apr8/_fig/county_total_rate.jpeg", bbox_inches = "tight")

## Verify
plt.show() # Show created plots

# End Section
print("THE END") # Print result

#################### Break ####################

# Section C: Compare with NYC and Global Data
print("Section C: Start") # Print result

## Step 1: Import Libraries and Data

### Import Data
df_nyc = pd.read_csv("nychealth_coronavirus-data/case-hosp-death.csv", encoding = "ISO-8859-1") # Import dataset saved as csv in _data folder
df_jhu = pd.read_csv("CSSEGISandData_COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv", encoding = "ISO-8859-1") # Import dataset saved as csv in _data folder

## Step 2: Prepare Data for Analysis

### Manipulate NYC Data to match DOH
df_nyc.insert(0, "Day", range(0, len(df_nyc))) # Create column of ascending values the length of the dataframe
df_rename = df_nyc.rename(columns = {"NEW_COVID_CASE_COUNT": "Cases"}) # Rename columns
df_filter = df_rename.filter(["Day", "Cases"]) # Keep only selected columns

### Create NYC 100 Day Data 
df_filter["TotalCount"] = df_filter["Cases"].cumsum() # Calculate cumulative total of column
df_nyc100 = df_filter[df_filter.TotalCount > 100] # Subset data frame by Total COunt of more than 100
df_nyc100.insert(0, "Day100", range(0, len(df_nyc100))) # Create column of ascending values the length of the dataframe

### Manipulate JHU Data for South Korea
df_sk = df_jhu[df_jhu["Country/Region"].str.contains("Korea, South")] # Subset by string value
df_drop = df_sk.drop(columns = ["Province/State", "Lat", "Long"]) # Keep only selected columns
df_rename = df_drop.rename(columns = {"Country/Region": "County"}) # Rename columns
df_long = pd.melt(df_rename, id_vars = ["County"]) # Convert daily table from Long to Wide with columns for County and Day
df_rename = df_long.rename(columns = {"variable": "Date", "value": "TotalCount"}) # Rename columns
df_rename.insert(0, "Day", range(0, len(df_rename))) # Create column of ascending values the length of the dataframe

### Create SK 100 Day Data
df_sk100 = df_rename[df_rename.TotalCount > 100] # Subset data frame by Total COunt of more than 100
df_sk100.insert(0, "Day100", range(0, len(df_sk100))) # Create column of ascending values the length of the dataframe

### Manipulate JHU Data for Italy
df_it = df_jhu[df_jhu["Country/Region"].str.contains("Italy")] # Subset by string value
df_drop = df_it.drop(columns = ["Province/State", "Lat", "Long"]) # Keep only selected columns
df_rename = df_drop.rename(columns = {"Country/Region": "County"}) # Rename columns
df_long = pd.melt(df_rename, id_vars = ["County"]) # Convert daily table from Long to Wide with columns for County and Day
df_rename = df_long.rename(columns = {"variable": "Date", "value": "TotalCount"}) # Rename columns
df_rename.insert(0, "Day", range(0, len(df_rename))) # Create column of ascending values the length of the dataframe

### Create It 100 Day Data
df_it100 = df_rename[df_rename.TotalCount > 100] # Subset data frame by Total COunt of more than 100
df_it100.insert(0, "Day100", range(0, len(df_it100))) # Create column of ascending values the length of the dataframe

### Create Florida 100 Day Data
df_fl100 = df_fl[df_fl.TotalCount > 100] # Subset data frame by Total COunt of more than 100
df_fl100.insert(0, "Day100", range(0, len(df_fl100))) # Create column of ascending values the length of the dataframe

### Create 100 Case Start Date
df_sk100["OneDay"] = 100*2.7183818**(0.6931471/1*df_sk100["Day100"])
df_sk100["TwoDay"] = 100*2.7183818**(0.6931471/2*df_sk100["Day100"])
df_sk100["ThreeDay"] = 100*2.7183818**(0.6931471/3*df_sk100["Day100"])
df_sk100["SevenDay"] = 100*2.7183818**(0.6931471/7*df_sk100["Day100"])
df_sk100["ThirtyDay"] = 100*2.7183818**(0.6931471/30*df_sk100["Day100"])

### Create Multiple Line Plot for Total by Log Post 100 Days
plt.figure(figsize = (16, 8))
plt.plot(df_fl100.Day100, df_fl100.TotalCount, color = "xkcd:neon blue")
plt.plot(df_nyc100.Day100, df_nyc100.TotalCount, color = "tab:olive")
plt.plot(df_it100.Day100, df_it100.TotalCount, color = "tab:brown")
plt.plot(df_sk100.Day100, df_sk100.TotalCount, color = "tab:pink")
plt.plot(df_sk100.Day100, df_sk100.OneDay, color = "0.75")
plt.plot(df_sk100.Day100, df_sk100.TwoDay, color = "0.75")
plt.plot(df_sk100.Day100, df_sk100.ThreeDay, color = "0.75")
plt.plot(df_sk100.Day100, df_sk100.SevenDay, color = "0.75")
plt.plot(df_sk100.Day100, df_sk100.ThirtyDay, color = "0.75")
plt.yscale("log")
plt.ylim(100, 1000000)
plt.yticks(np.arange(100, 1000000, step = 10000), np.arange(0, 1000000, step = 10000))
plt.ylabel("Total Confirmed Case Count")
plt.xlabel("Day Number after first 100 Cases")
plt.xticks(np.arange(len(df_sk100.Day100), step = 5), np.arange(len(df_sk100.Day100), step = 5), rotation = 90)
plt.legend(["Florida", "New York City", "Italy", "South Korea", "Double 2 Days", "Double 7 Days"])
plt.title("Florida DOH Confirmed COVID-19 Cases")
plt.savefig("_apr8/_fig/global_fl_total_100_log.jpeg")

## Verify
plt.show() # Show created plots

# End Section
print("THE END") # Print result