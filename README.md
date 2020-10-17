# The COVID-19 Pandemic in Florida
Data and visualization from the Florida Department of Health on the health outcomes related to the spread of novel coronovirus-19 and the resulting disease SARS CoV-2

## About this Repository
This repository is a collection of open source resources developed to conduct retrospective analysis on confirmed cases of COVID-19 by the Florida Department of Health in 2020. 
The data used for these analyses come directly from the public reports taken from the Florida DOH website for open consumption. You can reference the documentation in the raw data folders for each day for notes and sources.
Each folder in the repository contains updated data through the date and time listed in the name as well as code and figures specific for that data release. 

## Subrepositories
This repository is setup in two different projects for different use purposes.

### Daily Tracking by County
Subrepository label: `_daily` <br>
This subrespotiory contains data and visuals for tracking daily cases and mortality from COVID-19 among Florida Counties. The latest update was April 13, 2020. 

### Social Determinants by Zip Code
Subrepository label: `_sdoh` <br>
This subrepository contains data and visuals for tracking cases, hospitalizations, and mortality among zip codes and counties for the purpose of identifying socio-economic factors associated with variation in outcomes. 

## Repository Structure
The repository uses the following file organization and naming convenstions.

### File Organization
`_code`: code files related to the project
<br>`_data`: staged data files related to the project
<br>`_fig`: graphs, images, and maps related to the project
<br>`_pubs`: presentations and manuscrips related to the project
<br>`_raw`: raw data downloads related to the project
<br>`_refs`: selected literature related to the project

### File Naming Structure:
`prefix_topic_suffix_version.ext`

#### Prefixes:
`eco_`: Ecological predictor data files
<br>`geo_`: Geographic data files
<br>`cnn_`: Convolutional neural network with autoencoder scripts used for unstructured modelings
<br>`dnn_`: Dense neural network with classitifcation scripts for structured modelings
<br>`ml_`: Machine learning scripts using structured or unstructured models for feature selection
<br>`stat_`: Statistical testing and traditional regression scripts for validation models
<br>`plot_`: Visualization scripts for 2D displays including maps and figures

#### Topics
(To be added)

#### Suffixes:
`dev_`: Development script for working in an IDE
<br>`book_`: Jupyter notebook 
<br>`stage_`: Data files that have been modified from raw source
<br>`result_`: Text scripts displaying results from a model
<br>`map_`: 2D geographic display
<br>`graph_`: 2D chart or graph representing numeric data

#### Versions:
`alpha_`: first version of the file
<br>`beta_` : second version of the file
<br>`gamma_` : third version of the file
<br>`delta_` : fourth version of the file
<br>etc...

## Other Repositories
Non-florida data is collected from the following repositories:

#### [Johns Hopkins University Center for Systems Science and Engineering: https://github.com/CSSEGISandData/COVID-19](https://github.com/CSSEGISandData/COVID-19)
#### [New York City Department of Health and Mental Hygiene: https://github.com/nychealth/coronavirus-data](https://github.com/nychealth/coronavirus-data)

## Disclaimer
While the author (Andrew Cistola) is a Florida DOH employee and a University of Florida PhD student, these are NOT official publications by the Florida DOH, the University of Florida, or any other agency. 
No information is included in this repository that is not available to any member of the public. 
All information in this repository is available for public review and dissemination but is not to be used for making medical decisions. 
All code and data inside this repository is available for open source use per the terms of the included license. 