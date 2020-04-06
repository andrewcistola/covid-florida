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

	