# Information
label = "sdoh_mrfp"
path = "_sdoh/_code/"
version = "_alpha"
title = "SDOH and COVID-19 by Zip Code in Florida"
author = "DrewC!"

## Setup Workspace

### Import python libraries
import os # Operating system navigation
from datetime import datetime
from datetime import date

### Import data science libraries
import pandas as pd # Widely used data manipulation library with R/Excel like tables named 'data frames'
import numpy as np # Widely used matrix library for numerical processes
import statsmodels.api as sm # Statistics package best for regression models
import scipy.stats as st # Statistics package best for t-test, ChiSq, correlation

### Import scikit-learn libraries
from sklearn.preprocessing import StandardScaler # Standard scaling for easier use of machine learning algorithms
from sklearn.impute import SimpleImputer # Univariate imputation for missing data
from sklearn.cluster import KMeans # clusters data by trying to separate samples in n groups of equal variance
from sklearn.decomposition import PCA # Principal compnents analysis from sklearn
from sklearn.ensemble import RandomForestRegressor # Random Forest regression component
from sklearn.ensemble import RandomForestClassifier # Random Forest classification component
from sklearn.feature_selection import RFECV # Recursive Feature elimination with cross validation
from sklearn.linear_model import LinearRegression # Used for machine learning with quantitative outcome
from sklearn.linear_model import LogisticRegression # Used for machine learning with quantitative outcome
from sklearn.metrics import roc_curve # Reciever operator curve
from sklearn.metrics import auc # Area under the curve 

### Import keras libraries
from keras.models import Sequential # Uses a simple method for building layers in MLPs
from keras.models import Model # Uses a more complex method for building layers in deeper networks
from keras.layers import Dense # Used for creating dense fully connected layers
from keras.layers import Input # Used for designating input layers

### Import Visualization Libraries
import matplotlib.pyplot as plt # Comprehensive graphing package in python
import geopandas as gp # Simple mapping library for csv shape files with pandas like syntax for creating plots using matplotlib 

### Set Directory
os.chdir("/home/drewc/GitHub/covid-florida/") # Set wd to project repository

### Set Timestamps
day = str(date.today())
stamp = str(datetime.now())

### Append to Text File
text_file = open(path + label + version + "_" + day + ".txt", "w") # Open text file and name with subproject, content, and result suffix
text_file.write("####################" + "\n\n")
text_file.write(title + "\n") # Line of text with space after
text_file.write(version + "\n") # Line of text with space after
text_file.write(author + "\n") # Line of text with space after
text_file.write(stamp + "\n") # Line of text with space after
text_file.write("\n" + "####################" + "\n\n")
text_file.close() # Close file

# Step 1: Raw Data Processing
s1 = "Step 1: Raw Data Processing and Feature Engineering"
acs = "2018 American Community Survey 5-Year Percent Estimates by Zip Code"
doh = "Florida Department of Health Positve COVID-19 Cases by Zip Code"
rate = "Cumulative total of positive COVID-19 cases per 100,000 people"

## Zip Code Total Cases Population Rate
df_c19 = pd.read_csv("_sdoh/_raw/DOH_ZCTA/July 20 2020/FL_DOH_COVID_20July20_stage.csv") # Import dataset saved as csv in _data folder
df_c19["ZIP"] = df_c19["ZIP"].astype("str") # Change data type of column in data frame
df_c19["ZCTA"] = "ZCTA" + df_c19["ZIP"] # Add character string to each row in column
df_c19 = df_c19.drop(columns = ["COUNTYNAME", "ZIP"]) # Drop Unwanted Columns
df_c19 = df_c19.groupby(["ZCTA"], as_index = False).sum() # Group data by columns and sum
df_pop = pd.read_csv("_sdoh/_raw/ACS_DP5Y2018/ACS_DP5Y2018_ZCTA_full.csv") # Import dataset saved as csv in _data folder
df_pop = df_pop.filter(["DP05_0001PE", "ZCTA"]) # Keep only selected columns
df_c19 = pd.merge(df_c19, df_pop, on = "ZCTA", how = "inner") # Join by column while keeping only items that exist in both, select outer or left for other options
df_c19 = df_c19[df_c19['C19_CASES'] > 5] # Susbet numeric column by condition
df_c19 = df_c19[df_c19['DP05_0001PE'] > 500] # Susbet numeric column by condition
df_c19["CASE_RATE"] = df_c19["C19_CASES"] / df_c19["DP05_0001PE"] * 100000 # Calculate population rate per 100,000
df_c19 = df_c19.filter(["CASE_RATE", "ZCTA"]) # Keep only selected columns
df_c19 = df_c19.replace([np.inf, -np.inf], np.nan) # Replace infitite values with na
df_c19 = df_c19.dropna() # Drop all rows with NA values
df_c19 = df_c19.sort_values(by = ["CASE_RATE"], ascending = False) # Sort Columns by Value
df_c19.info() # Get class, memory, and column info: names, data types, obs.

## SDOH from the American Community Survey
df_acs = pd.read_csv("_sdoh/_raw/ACS_DP5Y2018/ACS_DP5Y2018_ZCTA_full.csv") # Import dataset saved as csv in _data folder
df_acs = pd.merge(df_c19, df_acs, on = "ZCTA", how = "inner") # Join by column while keeping only items that exist in both, select outer or left for other options
df_acs = df_acs.drop(columns = ["ZCTA", "FIPS", "ST", "CASE_RATE"]) # Drop Unwanted Columns
df_acs = df_acs.select_dtypes(exclude = ['int64']) # Drop all data types of certain column
df_acs = df_acs.replace([np.inf, -np.inf], np.nan) # Replace infitite values with na
df_acs = df_acs.dropna(axis = 1, thresh = 0.75*len(df_acs)) # Drop features less than 75% non-NA count for all columns
df_acs = pd.DataFrame(SimpleImputer(strategy = "median").fit_transform(df_acs), columns = df_acs.columns) # Impute missing data
df_acs = pd.DataFrame(StandardScaler().fit_transform(df_acs.values), columns = df_acs.columns) # Standard scale values by converting the normalized features into a tabular format with the help of DataFrame.
df_acs = df_acs.reset_index() # Reset Index
df_acs.info() # Get class, memory, and column info: names, data types, obs.

## Combine Cleaned ACS and Case Rate for FL Zip Codes
df_zcta = pd.read_csv("_sdoh/_raw/ACS_DP5Y2018/ACS_DP5Y2018_ZCTA_full.csv") # Import dataset saved as csv in _data folder
df_zcta = df_zcta.filter(["ZCTA", "ST"]) # Keep only selected columns
df_zcta = pd.merge(df_c19, df_zcta, on = "ZCTA", how = "inner") # Join by column while keeping only items that exist in both, select outer or left for other options
df_zcta = df_zcta.reset_index() # Reset Index
df_zcta = df_zcta.filter(["index", "ZCTA", "CASE_RATE"]) # Keep only selected columns
df_zcta = pd.merge(df_zcta, df_acs, on = "index", how = "inner") # Join by column while keeping only items that exist in both, select outer or left for other options
df_zcta = df_zcta.drop(columns = ["index"]) # Drop Unwanted Columns
df_zcta = df_zcta.set_index("ZCTA") # Set column as index
df_zcta.info() # Get class, memory, and column info: names, data types, obs.

## Standardize inputs and targets
df_xy = df_zcta # Rename data frame
df_xy["quant"] = df_xy["CASE_RATE"] # Rename as standard outcome
df_xy = df_xy.drop(columns = ["CASE_RATE"]) # Drop Unwanted Columns
df_xy["train"] = np.where(df_xy["quant"] >= np.percentile(df_xy["quant"], 95), 1, 0) # Create New Column Based on Conditions, Input 1 for each value over 95 percentile
df_xy["test"] = np.where(df_xy["quant"] >= np.percentile(df_xy["quant"], 80), 1, 0) # Create New Column Based on Conditions, Input 1 for each value over 80 percentile
df_X = df_xy.drop(columns = ["quant", "train", "test"]) # Drop Unwanted Columns
df_Y = df_xy.filter(["quant", "train", "test"]) # Keep only selected columns
df_xy.info() # Get class, memory, and column info: names, data types, obs.

### Create Choropleth Map
gdf_shape = gp.read_file("_sdoh/_raw/cb_2018_us_zcta510_500k/cb_2018_us_zcta510_500k.shp") # Import shape files from folder with all other files downloaded
gdf_c19 = pd.merge(gdf_shape, df_c19, on = "State", how = "inner") # Geojoins can use pandas merge as long as geo data is first passed in function
map = gdf_c19.plot(column = "CASE_RATE", cmap = "Blues", figsize = (16, 10), scheme = "equal_interval", k = 9, legend = True)
map.set_axis_off()
map.set_title(title, fontdict = {'fontsize': 20}, loc = "left")
map.get_legend().set_bbox_to_anchor((.6, .4))
plt.savefig(path + label + version + "_" + day + ".jpeg", dpi = 1000) 

### Append to Text File
text_file = open(path + label + version + "_" + day + ".txt", "a") # Open text file and name with subproject, content, and result suffix
text_file.write(s1 + "\n\n") # Line of text with space after
text_file.write(doh + "\n\n") # Add two lines of blank text at end of every section text
text_file.write("Target labels: (quant, train, test) = (case rate per 100k, 95 percentile, top quintile)" + "\n")
text_file.write("Target processing: Pop > 500, Cases > 5, Only Non-NA" + "\n\n") # Add two lines of blank text at end of every section text
text_file.write(str(df_Y.describe())  + "\n\n") # Add two lines of blank text at end of every section text
text_file.write(str(df_c19.head(10))  + "\n\n") # Add two lines of blank text at end of every section text
text_file.write(acs + "\n\n") # Add two lines of blank text at end of every section text
text_file.write("Features and observations: (Rows, Columns) = " + str(df_acs.shape) + "\n") # Add two lines of blank text at end of every section text
text_file.write("Feature processing: 75% nonNA, Median Imputed NA, Standard Scaled" + "\n\n") # Add two lines of blank text at end of every section text
text_file.write("####################" + "\n\n")
text_file.close() # Close file

# Step 2: Identify Predictors with Open Box Models
s2 = "Step 2: Identify Predictors with Open Models"
m1 = "Principal Component Analysis"
m2 = "Random Forests"
m3 = "Recursive feature Elimination"

## Principal Component Analysis
degree = len(df_X.columns) - 1  # Save number of features -1 to get degrees of freedom
pca = PCA(n_components = degree) # Pass the number of components to make PCA model based on degrees of freedom
pca.fit(df_X) # Fit initial PCA model
df_comp = pd.DataFrame(pca.explained_variance_) # Print explained variance of components
df_comp = df_comp[(df_comp[0] > 1)] # Save eigenvalues above 1 to identify components
components = len(df_comp.index) - 3 # Save count of components for Variable reduction
pca = PCA(n_components = components) # you will pass the number of components to make PCA model
pca.fit_transform(df_X) # finally call fit_transform on the aggregate data to create PCA results object
df_pc = pd.DataFrame(pca.components_, columns = df_X.columns) # Export eigenvectors to data frame with column names from original data
df_pc["Variance"] = pca.explained_variance_ratio_ # Save eigenvalues as their own column
df_pc = df_pc[df_pc["Variance"] > df_pc["Variance"].mean()] # Susbet by eigenvalues with above average exlained variance ratio
df_pc = df_pc.abs() # Get absolute value of eigenvalues
df_pc = df_pc.drop(columns = ["Variance"]) # Drop outcomes and targets
df_p = pd.DataFrame(df_pc.max(), columns = ["MaxEV"]) # select maximum eigenvector for each feature
df_p = df_p[df_p.MaxEV > df_p.MaxEV.mean()] # Susbet by above average max eigenvalues 
df_p = df_p.reset_index() # Add a new index of ascending values, existing index consisting of feature labels becomes column named "index"
df_pca = df_p.rename(columns = {"index": "Features"}) # Rename former index as features
df_pca = df_pca.sort_values(by = ["MaxEV"], ascending = False) # Sort Columns by Value
df_pca.info() # Get class, memory, and column info: names, data types, obs.

### Random Forest Regressor
forest = RandomForestRegressor(n_estimators = 1000, max_depth = 10) #Use default values except for number of trees. For a further explanation see readme included in repository. 
forest.fit(df_X, df_Y["quant"]) # Fit Forest model, This will take time
rf = forest.feature_importances_ # Output importances of features
l_rf = list(zip(df_X, rf)) # Create list of variables alongside importance scores 
df_rf = pd.DataFrame(l_rf, columns = ["Features", "Gini"]) # Create data frame of importances with variables and gini column names
df_rf = df_rf[(df_rf["Gini"] > df_rf["Gini"].mean())] # Subset by Gini values higher than mean
df_rf = df_rf.sort_values(by = ["Gini"], ascending = False) # Sort Columns by Value
df_rf.info() # Get class, memory, and column info: names, data types, obs.

### Fracture: Join RF and PCA 
df_fr = pd.merge(df_pca, df_rf, on = "Features", how = "inner") # Join by column while keeping only items that exist in both, select outer or left for other options
fracture = df_fr["Features"].tolist() # Save features from data frame
df_fr.info() # Get class, memory, and column info: names, data types, obs.

### Recursive Feature Elimination
recursive = RFECV(estimator = LinearRegression(), min_features_to_select = 5) # define selection parameters, in this case all features are selected. See Readme for more ifo
recursive.fit(df_X[fracture], df_Y["quant"]) # This will take time
rfe = recursive.support_ # Save Boolean values as numpy array
l_rfe = list(zip(df_X[fracture], rfe)) # Create list of variables alongside RFE value 
df_rfe = pd.DataFrame(l_rfe, columns = ["Features", "RFE"]) # Create data frame of importances with variables and gini column names
df_rfe = df_rfe.sort_values(by = ["RFE"], ascending = True) # Sort Columns by Value
df_rfe = df_rfe[df_rfe["RFE"] == True] # Select Variables that were True
df_rfe.info() # Get class, memory, and column info: names, data types, obs.

### FractureProof: Join RFE with Fracture
df_fp = pd.merge(df_fr, df_rfe, on = "Features", how = "inner") # Join by column while keeping only items that exist in both, select outer or left for other options
fractureproof = df_fp["Features"].tolist() # Save chosen featres as list
df_fp.info() # Get class, memory, and column info: names, data types, obs.

### Initial Multiple Regression
X = df_X[fractureproof] # Subset by selected features
Y = df_Y["quant"] # Subset by selected features
mod = sm.OLS(Y, X) # Create linear model
res = mod.fit() # Fit model to create result
res.summary() # Print results of regression model

### Final Multiple Regression
mrfractureproof = ["DP02_0066PE",
                    "DP04_0047PE",
                    "DP03_0027PE",
                    "DP02_0115PE",
                    "DP02_0071PE",
                    "DP02_0012PE",
                    "DP03_0009PE",
                    "DP02_0064PE",
                    "DP05_0077PE"] # DP02_0071PE: With Disability, DP02_0012PE: Over 65, DP03_0009PE: Unemployment Rate, DP02_0064PE: With College Degree, DP05_0077PE: Non-Hispanic White
X = df_X[mrfractureproof] # Subset by selected features
Y = df_Y["quant"] # Subset by selected features
mod = sm.OLS(Y, X) # Create linear model
res = mod.fit() # Fit model to create result
res.summary() # Print results of regression model

### Add feature labels
df_label = pd.read_csv('_sdoh/_raw/ACS_DP5Y2018/ACS_DP5Y2018_ZCTA__PE_labels.csv') # Import dataset saved as csv in _data folder
df_label = df_label.rename(columns = {"Code": "Features"}) # Rename multiple columns in place
df_label = df_label.filter(["Features", "Label"]) # Keep only selected columns
df_label = df_label.set_index("Features") # Set column as index
df_label = df_label.transpose() # Switch rows and columns
df_label = df_label[mrfractureproof] # Save chosen featres as list
df_label = df_label.transpose() # Switch rows and columns
df_label = df_label.reset_index() # Reset index
l_label = list(zip(df_label["Features"], df_label["Label"])) # Create list of variables alongside RFE value 
df_label.info() # Get class, memory, and column info: names, data types, obs.

### Append to Text File
text_file = open(path + label + version + "_" + day + ".txt", "a") # Open text file and name with subproject, content, and result suffix
text_file.write(s2 + "\n\n") # Line of text with space after
text_file.write(acs + "\n") # Add two lines of blank text at end of every section text
text_file.write("Models:" + m1 + m2 + m3 + "\n\n") # Add two lines of blank text at end of every section text
text_file.write("Values: Eigenvectors, Gini Impurity, Boolean" + "\n") # Add two lines of blank text at end of every section text
text_file.write("Thresholds: Mean, Mean, Cross Validation" + "\n\n") # Add two lines of blank text at end of every section text
text_file.write(str(df_fp)  + "\n\n") # Add two lines of blank text at end of every section text
text_file.write(str(res.summary())  + "\n\n") # Add two lines of blank text at end of every section text
text_file.write(str(l_label)  + "\n\n") # Add two lines of blank text at end of every section text
text_file.write("####################" + "\n\n")
text_file.close() # Close file

# Step 3: Prediction Comparison with Closed Box Models
sub2 = "Step 3: Prediction Comparison with Closed Box Models"
m5 = "Multi-Layer Perceptron (with full ACS dataset)"
m6 = "Multi-Layer Perceptron (with selected features)"

### Build Network with keras Sequential API
# Prep Inputs
X = df_X
Y_train = df_X["train"]
Y_test = df_X["test"]
epochs = 100
input = df_X.shape[1] # Save number of columns as length minus quant, test, train and round to nearest integer
nodes = round(input / 2) # Number of input dimensions divided by two for nodes in each layer
network = Sequential()
# Dense Layers
network.add(Dense(nodes, activation = 'relu', kernel_initializer = 'random_normal', input_dim = input)) # First Hidden Layer
network.add(Dense(nodes, activation = 'relu', kernel_initializer = 'random_normal')) # Second Hidden Layer
# Activation Layer
network.add(Dense(1, activation = 'sigmoid', kernel_initializer = 'random_normal')) # Output Layer
# Compile
network.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) # Compile network with ADAM ("Adaptive moment estimation" or RMSProp + Momentum)
# Fit
network.fit(X, Y_train, batch_size = 10, epochs = epochs) # Fitting the data to the train outcome
# Predict
Y_pred = network.predict(X) # Predict values from testing model
# AUC Score
fpr, tpr, threshold = roc_curve((Y_train > 0), (Y_pred > 0.5)) # Create ROC outputs, true positive rate and false positive rate
auc_train = auc(fpr, tpr) # Plot ROC and get AUC score
fpr, tpr, threshold = roc_curve((Y_test > 0), (Y_pred > 0.5)) # Create ROC outputs, true positive rate and false positive rate
auc_test = auc(fpr, tpr) # Plot ROC and get AUC score
# Prep Outputs
full_epochs = epochs
full_train = auc_train
full_test = auc_test

# Prep Inputs
X = df_X["fractureproof"]
Y_train = df_X["train"]
Y_test = df_X["test"]
epochs = 100
input = df_X.shape[1] # Save number of columns as length minus quant, test, train and round to nearest integer
nodes = round(input / 2) # Number of input dimensions divided by two for nodes in each layer
network = Sequential()
# Dense Layers
network.add(Dense(nodes, activation = 'relu', kernel_initializer = 'random_normal', input_dim = input)) # First Hidden Layer
network.add(Dense(nodes, activation = 'relu', kernel_initializer = 'random_normal')) # Second Hidden Layer
# Activation Layer
network.add(Dense(1, activation = 'sigmoid', kernel_initializer = 'random_normal')) # Output Layer
# Compile
network.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) # Compile network with ADAM ("Adaptive moment estimation" or RMSProp + Momentum)
# Fit
network.fit(X, Y_train, batch_size = 10, epochs = epochs) # Fitting the data to the train outcome
# Predict
Y_pred = network.predict(X) # Predict values from testing model
# AUC Score
fpr, tpr, threshold = roc_curve((Y_train > 0), (Y_pred > 0.5)) # Create ROC outputs, true positive rate and false positive rate
auc_train = auc(fpr, tpr) # Plot ROC and get AUC score
fpr, tpr, threshold = roc_curve((Y_test > 0), (Y_pred > 0.5)) # Create ROC outputs, true positive rate and false positive rate
auc_test = auc(fpr, tpr) # Plot ROC and get AUC score
# Prep Outputs
sub_epochs = epochs
sub_train = auc_train
sub_test = auc_test

### Append to Text File
text_file = open(path + day + "_results" + label + ".txt", "a") # Open text file and name with subproject, content, and result suffix
text_file.write(s3 + "\n\n") # Line of text with space after
text_file.write("Layers: Dense, Dense, Activation" + "\n") # Add two lines of blank text at end of every section text
text_file.write("Functions: ReLU, ReLU, Sigmoid" + "\n") # Add two lines of blank text at end of every section text
text_file.write("Targets: (train, test), (95 percentile, 80 percentile)" + "\n")
text_file.write("Features: Selected = " + fractureproof + "\n\n")
text_file.write(m5 + "AUC Scores" + "\n") # Add two lines of blank text at end of every section text
text_file.write("  train = " + full_train + "\n") # Add two lines of blank text at end of every section text
text_file.write("  test = " + full_test + "\n") # Add two lines of blank text at end of every section text
text_file.write("  Epochs = " + full_epochs + "\n") # Add two lines of blank text at end of every section text
text_file.write(m6 + "AUC Scores" + "\n\n") # Add two lines of blank text at end of every section text
text_file.write("  train = " + sub_train + "\n") # Add two lines of blank text at end of every section text
text_file.write("  test = " + sub_test + "\n") # Add two lines of blank text at end of every section text
text_file.write("  Epochs = " + sub_epochs + "\n") # Add two lines of blank text at end of every section text
text_file.write("####################" + "\n\n")
text_file.close() # Close file

