# My Library

## Temporal Data Analysis

### Import Standard Libraries
import os # Inlcuded in every script DC!
import pandas as pd # Incldued in every code script for DC!
import numpy as np # Incldued in every code script for DC!
import scipy as st # Incldued in every code script for DC!

### Set working directory to subproject folder
os.chdir("C:/Users/drewc/GitHub/mylibrary/templates") # Set wd to project repository

#################### Break ####################

# Section A: Analysis Title
print("Section Start") # Print result

## Step 1: Import Libraries and Data

### Import Libraries Specific for this Analysis
import matplotlib.pyplot as plt

### Import CMS Data
df_cms = pd.read_csv("_data/health_mspb_hospital_stage.csv", encoding = "ISO-8859-1") # Import dataset saved as csv in _data folder

### Import State Shape File
gdf_state = gp.read_file("_data/health_maps_state_stage.shp")

### Verify CMS
df_cms.info() # Get class, memory, and column info: names, data types, obs.
df_cms.head() # Print first 5 observations

## Step 2: Prepare Data for Analysis

### Select only State and Measure
df_filter = df_cms.filter(["State", "Score"]) # Keep only selected columns

### Group by State
df_group = df_filter.groupby(["State"], as_index = False).mean() # Group data By Columns and Sum

### Rename Score as MSPB
df_rename = df_group.rename(columns = {"Score": "MSPB"}) # Rename column

### Drop NA values
df_na = df_rename.dropna() # Drop all rows with NA values

### Rename Dataframe
df_mspb = df_na # Rename sorted dataframe as MSPB for clarity

### Verify MSPB
df_mspb.info() # Get class, memory, and column info: names, data types, obs.
df_mspb.head() # Print first 5 observations

## Step 3: Conduct Analysis and Tests

### Conduct ChiSq in SciPy
obs = df_chsq["Observed"]
exp = df_chsq["Expected"]
st.chisquare(obs, exp) # ChiSq with obs = observed and exp = observed

## Step 4: Create Visuals and Outputs

## Create Side by Side barplot
plt.figure()
plt.bar((1 - 0.2), df_chsq.loc[0, "Expected"], color = 'b', width = 0.4)
plt.bar((1 + 0.2), df_chsq.loc[0, "Observed"], color = 'r', width = 0.4)
plt.bar((2 - 0.2), df_chsq.loc[1, "Expected"], color = 'b', width = 0.4)
plt.bar((2 + 0.2), df_chsq.loc[1, "Observed"], color = 'r', width = 0.4)
plt.bar((3 - 0.2), df_chsq.loc[2, "Expected"], color = 'b', width = 0.4)
plt.bar((3 + 0.2), df_chsq.loc[2, "Observed"], color = 'r', width = 0.4)
plt.bar((4 - 0.2), df_chsq.loc[3, "Expected"], color = 'b', width = 0.4)
plt.bar((4 + 0.2), df_chsq.loc[3, "Observed"], color = 'r', width = 0.4)
plt.bar((5 - 0.2), df_chsq.loc[4, "Expected"], color = 'b', width = 0.4)
plt.bar((5 + 0.2), df_chsq.loc[4, "Observed"], color = 'r', width = 0.4)
plt.bar((6 - 0.2), df_chsq.loc[5, "Expected"], color = 'b', width = 0.4)
plt.bar((6 + 0.2), df_chsq.loc[5, "Observed"], color = 'r', width = 0.4)
plt.bar((7 - 0.2), df_chsq.loc[6, "Expected"], color = 'b', width = 0.4)
plt.bar((7 + 0.2), df_chsq.loc[6, "Observed"], color = 'r', width = 0.4)
plt.xticks((1, 2, 3, 4, 5, 6, 7), df_chsq["Ownership"], rotation = 90)
plt.legend(["Expected", "Observed"])
plt.title("Expected and Observed Counts of VBP Penalties over 1 Percent by Hospital Type 2019")
plt.savefig("_fig/health_penalty_hospital_bar.jpeg", bbox_inches = "tight")

## Verify
plt.show() # Show created plots

# End Section
print("THE END") # Print result