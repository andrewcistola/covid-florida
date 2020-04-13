# COVID Florida

## Alachua County Temporal Analysis

### Import Standard Libraries
import os # Inlcuded in every script DC!
import pandas as pd # Incldued in every code script for DC!
import numpy as np # Incldued in every code script for DC!
import scipy as st # Incldued in every code script for DC!

### Set working directory to subproject folder
os.chdir("C:/Users/drewc/GitHub/covid-florida/_apr9") # Set wd to project repository

#################### Break ####################

# Section A: Create Table of Confirmed Cases by County and Day Count
print("Section A: Start") # Print result

## Step 1: Import Libraries and Data

### Import Data
df_case = pd.read_csv("_data/cases_raw.csv", encoding = "ISO-8859-1") # Import dataset saved as csv in _data folder

### Verify CMS
df_case.info() # Get class, memory, and column info: names, data types, obs.
df_case.head() # Print first 5 observations

## Step 2: Prepare Data for Analysis

### Create Daily Count of New Cases
df_rename =  df_case.rename(columns = {"ObjectId": "ID", "Case_": "Date"}) # Rename column
df_drop = df_rename.filter(["County", "Date", "ID"]) # Keep only selected columns
df_group = df_drop.groupby(["County", "Date"], as_index = False).count() # Group data By Columns and Sum
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
df_daily = df_order # Reset index as County variable and rename to daily

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

### Create Cumulative Totals for Counts
df_fl["TotalCount"] = df_fl["Cases"].cumsum()
df_al["TotalCount"] = df_al["Cases"].cumsum()

### Create Cumulative Totals for Rates
df_fl["TotalRate"] = df_fl["Rate"].cumsum()
df_al["TotalRate"] = df_al["Rate"].cumsum()

### Verify
df_fl.to_csv(r"_data/florida_raw.csv") # Clean in excel and select variable
df_al.to_csv(r"_data/alachua_raw.csv") # Clean in excel and select variable

## Step 4: Create Visuals and Outputs

## Alachua County

### Create Barplot for New Cases by Count
plt.figure()
x = np.arange(len(df_al.Day))
plt.bar((x), df_al.Cases, color = 'xkcd:neon green', width = 0.4)
plt.xticks((x), df_al["Day"], rotation = 90)
plt.ylabel("Daily Cases")
plt.xlabel("Day Number")
plt.legend(["Alachua"])
plt.title("Florida DOH Confirmed COVID-19 Cases by Day")
plt.savefig("_fig/alachua_daily_count.jpeg", bbox_inches = "tight")

### Create Barplot for New Cases by Rate
plt.figure()
x = np.arange(len(df_al.Day))
plt.bar((x), df_al.Rate, color = 'xkcd:neon green', width = 0.4)
plt.xticks((x), df_al["Day"], rotation = 90)
plt.ylabel("Daily Cases Rate per 100k")
plt.xlabel("Day Number")
plt.legend(["Alachua"])
plt.title("Florida DOH County Confirmed COVID-19 Cases by Day")
plt.savefig("_fig/alachua_daily_rate.jpeg", bbox_inches = "tight")

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
plt.savefig("_fig/florida_daily_count.jpeg", bbox_inches = "tight")

### Create Barplot for New Cases by Rate
plt.figure()
x = np.arange(len(df_fl.Day))
plt.bar((x), df_fl.Rate, color = 'xkcd:neon blue', width = 0.4)
plt.xticks((x), df_fl["Day"], rotation = 90)
plt.ylabel("Daily Cases Rate per 100k")
plt.xlabel("Day Number")
plt.legend(["Florida"])
plt.title("Florida DOH Confirmed COVID-19 Cases by Day")
plt.savefig("_fig/florida_daily_rate.jpeg", bbox_inches = "tight")

## Verify
plt.show() # Show created plots

# End Section
print("THE END") # Print result

#################### Break ####################

# Section C: Create Table of Confirmed Cases by Top FL Counties
print("Section C: Start") # Print result

## Step 1: Import Libraries and Data

### Verify FL and Alachua 
df_fl.info() # Get class, memory, and column info: names, data types, obs.
df_fl.head() # Print first 5 observations
df_al.info() # Get class, memory, and column info: names, data types, obs.
df_al.head() # Print first 5 observations

### Create Top County Specific Data
df_br = df_day[df_day["County"].str.contains("Broward")]
df_da = df_day[df_day["County"].str.contains("Dade")]
df_pb = df_day[df_day["County"].str.contains("Palm Beach")]
df_hb = df_day[df_day["County"].str.contains("Hillsborough")]
df_or = df_day[df_day["County"].str.contains("Orange")]
df_pn = df_day[df_day["County"].str.contains("Pinellas")]
df_dv = df_day[df_day["County"].str.contains("Duval")]

### Create Cumulative Totals for Counts
df_br["TotalCount"] = df_br["Cases"].cumsum()
df_da["TotalCount"] = df_da["Cases"].cumsum()
df_pb["TotalCount"] = df_pb["Cases"].cumsum()
df_hb["TotalCount"] = df_hb["Cases"].cumsum()
df_or["TotalCount"] = df_or["Cases"].cumsum()
df_pn["TotalCount"] = df_pn["Cases"].cumsum()
df_dv["TotalCount"] = df_dv["Cases"].cumsum()

### Create Cumulative Totals for Rates
df_br["TotalRate"] = df_br["Rate"].cumsum()
df_da["TotalRate"] = df_da["Rate"].cumsum()
df_pb["TotalRate"] = df_pb["Rate"].cumsum()
df_hb["TotalRate"] = df_hb["Rate"].cumsum()
df_or["TotalRate"] = df_or["Rate"].cumsum()
df_pn["TotalRate"] = df_pn["Rate"].cumsum()
df_dv["TotalRate"] = df_dv["Rate"].cumsum()

### Create County 25 Day Data
df_al25 = df_al[df_al.TotalCount > 25] # Subset data frame by Total COunt of more than 100
df_al25.insert(0, "Day25", range(0, len(df_al25))) # Create column of ascending values the length of the dataframe
df_br25 = df_br[df_br.TotalCount > 25] # Subset data frame by Total COunt of more than 100
df_br25.insert(0, "Day25", range(0, len(df_br25))) # Create column of ascending values the length of the dataframe
df_da25 = df_da[df_da.TotalCount > 25] # Subset data frame by Total COunt of more than 100
df_da25.insert(0, "Day25", range(0, len(df_da25))) # Create column of ascending values the length of the dataframe
df_pb25 = df_pb[df_pb.TotalCount > 25] # Subset data frame by Total COunt of more than 100
df_pb25.insert(0, "Day25", range(0, len(df_pb25))) # Create column of ascending values the length of the dataframe
df_hb25 = df_hb[df_hb.TotalCount > 25] # Subset data frame by Total COunt of more than 100
df_hb25.insert(0, "Day25", range(0, len(df_hb25))) # Create column of ascending values the length of the dataframe
df_or25 = df_or[df_or.TotalCount > 25] # Subset data frame by Total COunt of more than 100
df_or25.insert(0, "Day25", range(0, len(df_or25))) # Create column of ascending values the length of the dataframe
df_pn25 = df_pn[df_pn.TotalCount > 25] # Subset data frame by Total COunt of more than 100
df_pn25.insert(0, "Day25", range(0, len(df_pn25))) # Create column of ascending values the length of the dataframe
df_dv25 = df_dv[df_dv.TotalCount > 25] # Subset data frame by Total COunt of more than 100
df_dv25.insert(0, "Day25", range(0, len(df_dv25))) # Create column of ascending values the length of the dataframe

### Create 25 Case Start Date
df_al25["OneDay"] = 25*2.7183818**(0.6931471/1*df_al25["Day25"])
df_al25["TwoDay"] = 25*2.7183818**(0.6931471/2*df_al25["Day25"])
df_al25["ThreeDay"] = 25*2.7183818**(0.6931471/3*df_al25["Day25"])
df_al25["SevenDay"] = 25*2.7183818**(0.6931471/7*df_al25["Day25"])
df_al25["ThirtyDay"] = 25*2.7183818**(0.6931471/30*df_al25["Day25"])

### Create 1 per 100k Data
df_al1k = df_al[df_al.TotalRate > 1] # Subset data frame by Total COunt of more than 100
df_al1k.insert(0, "Day1k", range(0, len(df_al1k))) # Create column of ascending values the length of the dataframe
df_br1k = df_br[df_br.TotalRate > 1] # Subset data frame by Total COunt of more than 100
df_br1k.insert(0, "Day1k", range(0, len(df_br1k))) # Create column of ascending values the length of the dataframe
df_da1k = df_da[df_da.TotalRate > 1] # Subset data frame by Total COunt of more than 100
df_da1k.insert(0, "Day1k", range(0, len(df_da1k))) # Create column of ascending values the length of the dataframe
df_pb1k = df_pb[df_pb.TotalRate > 1] # Subset data frame by Total COunt of more than 100
df_pb1k.insert(0, "Day1k", range(0, len(df_pb1k))) # Create column of ascending values the length of the dataframe
df_hb1k = df_hb[df_hb.TotalRate > 1] # Subset data frame by Total COunt of more than 100
df_hb1k.insert(0, "Day1k", range(0, len(df_hb1k))) # Create column of ascending values the length of the dataframe
df_or1k = df_or[df_or.TotalRate > 1] # Subset data frame by Total COunt of more than 100
df_or1k.insert(0, "Day1k", range(0, len(df_or1k))) # Create column of ascending values the length of the dataframe
df_pn1k = df_pn[df_pn.TotalRate > 1] # Subset data frame by Total COunt of more than 100
df_pn1k.insert(0, "Day1k", range(0, len(df_pn1k))) # Create column of ascending values the length of the dataframe
df_dv1k = df_dv[df_dv.TotalRate > 1] # Subset data frame by Total COunt of more than 100
df_dv1k.insert(0, "Day1k", range(0, len(df_dv1k))) # Create column of ascending values the length of the dataframe

### Create 25 Case Start Date
df_al1k["OneDay"] = 1*2.7183818**(0.6931471/1*df_al1k["Day1k"])
df_al1k["TwoDay"] = 1*2.7183818**(0.6931471/2*df_al1k["Day1k"])
df_al1k["ThreeDay"] = 1*2.7183818**(0.6931471/3*df_al1k["Day1k"])
df_al1k["SevenDay"] = 1*2.7183818**(0.6931471/7*df_al1k["Day1k"])
df_al1k["ThirtyDay"] = 1*2.7183818**(0.6931471/30*df_al1k["Day1k"])

## Multi-County

### Create Multiple Line Plot for Total by Count
plt.figure(figsize = (16, 8))
plt.plot(df_al25.Day25, df_al25.TotalCount, color = "xkcd:neon green")
plt.plot(df_br25.Day25, df_br25.TotalCount, color = "r")
plt.plot(df_da25.Day25, df_da25.TotalCount, color = "y")
plt.plot(df_hb25.Day25, df_hb25.TotalCount, color = "m")
plt.plot(df_or25.Day25, df_or25.TotalCount, color = "g")
plt.plot(df_pb25.Day25, df_pb25.TotalCount, color = "tab:orange")
plt.plot(df_pn25.Day25, df_pn25.TotalCount, color = "tab:purple")
plt.plot(df_dv25.Day25, df_dv25.TotalCount, color = "tab:blue")
plt.plot(df_al25.Day25, df_al25.OneDay, color = "0.75")
plt.plot(df_al25.Day25, df_al25.TwoDay, color = "0.75")
plt.plot(df_al25.Day25, df_al25.ThreeDay, color = "0.75")
plt.plot(df_al25.Day25, df_al25.SevenDay, color = "0.75")
plt.plot(df_al25.Day25, df_al25.ThirtyDay, color = "0.75")
plt.ylim(25, 7000)
plt.ylabel("Cumulative Cases")
plt.yscale("log")
plt.xlabel("Day Number")
plt.xticks(np.arange(len(df_al25.Day25), step = 5), np.arange(len(df_al25.Day25), step = 5), rotation = 90)
plt.legend(["Alachua", "Broward", "Dade", "Hillsborough", "Orange", "Palm Beach", "Pinellas", "Duval"])
plt.title("Florida DOH Confirmed COVID-19 Cases by Day")
plt.savefig("_fig/county_total_count.jpeg")

### Create Multiple Line Plot for Total by Count
plt.figure(figsize = (16, 8))
plt.plot(df_al1k.Day1k, df_al1k.TotalRate, color = "xkcd:neon green")
plt.plot(df_br1k.Day1k, df_br1k.TotalRate, color = "r")
plt.plot(df_da1k.Day1k, df_da1k.TotalRate, color = "y")
plt.plot(df_hb1k.Day1k, df_hb1k.TotalRate, color = "m")
plt.plot(df_or1k.Day1k, df_or1k.TotalRate, color = "g")
plt.plot(df_pb1k.Day1k, df_pb1k.TotalRate, color = "tab:orange")
plt.plot(df_pn1k.Day1k, df_pn1k.TotalRate, color = "tab:purple")
plt.plot(df_dv1k.Day1k, df_dv1k.TotalRate, color = "tab:blue")
plt.plot(df_al1k.Day1k, df_al1k.OneDay, color = "0.75")
plt.plot(df_al1k.Day1k, df_al1k.TwoDay, color = "0.75")
plt.plot(df_al1k.Day1k, df_al1k.ThreeDay, color = "0.75")
plt.plot(df_al1k.Day1k, df_al1k.SevenDay, color = "0.75")
plt.plot(df_al1k.Day1k, df_al1k.ThirtyDay, color = "0.75")
plt.ylim(1, 350)
plt.ylabel("Cumulative Cases Rate per 100k")
plt.yscale("log")
plt.xlabel("Day Number")
plt.xticks(np.arange(len(df_al1k.Day1k), step = 5), np.arange(len(df_al1k.Day1k), step = 5), rotation = 90)
plt.legend(["Alachua", "Broward", "Dade", "Hillsborough", "Orange", "Palm Beach", "Pinellas", "Duval"])
plt.title("Florida DOH Confirmed COVID-19 Cases by Day")
plt.savefig("_fig/county_total_rate.jpeg")

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