Covid-Florida - April 5
Data Staging from Raw Files

#####

cases_stage.csv

Raw Data File: "covid-florida/_apr5/_raw/FLCharts COVID19 on 5 Apr 2020 250 pm/features.csv"

Staging Steps taken in Microsoft Excel: 
	Delete all columns except: "County", "Case_", "ObjectId"
	Rename column: Old = "Case_" New = "Date"; Old = "ObjectId" New = "ID"
	Set data type using "Home" / "Number" dropdown menu : Column = Date, Type = Short Date

Staged Data File: "covid-florida/_apr5/_data/cases_stage.csv"

#####

nyc_cases_stage.csv

Raw Data File: ""

Staging Steps taken in Microsoft Excel: 
	Delete all columns except: ""
	Rename column: Old = "" New = "Date"; Old = "" New = "ID"
	Set data type using "Home" / "Number" dropdown menu : Column = Date, Type = Short Date

Staged Data File: ""
	
#####

global_cases_stage.csv

Raw Data File: "C:\Users\drewc\GitHub\covid-florida\_apr5\_raw\JHUCSSE\csse_covid_19_data\csse_covid_19_time_series\time_series_covid19_confirmed_global.csv"

Staging Steps taken in Microsoft Excel: 
	Delete all rows except: "Italy", "Spain", "Korea, South"
	Delete Column: "Province/State"
	Rename column: Old = "Region/Country" New = "County"
	Set data type using "Home" / "Number" dropdown menu : Column = Date, Type = Short Date
	First day for each row with a case set to Day0
	Dates removed from rows
	"Korea, South" Changed to South Korea
	All Days changed to integer format

Staged Data File: "covid-florida/_apr5/_data/cases_stage.csv"