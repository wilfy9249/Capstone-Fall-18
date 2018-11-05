#!/usr/bin/env python
# coding: utf-8

###------NYC Events Historic : DATA CLEANING-----###

#Import Packages
import pandas as pd
from data_utility import readCsvFile

## Read CSV
getEvents = readCsvFile('../Data/NYC_Permitted_Event_Information_Historical.csv')

#Drop Columns
getRemEvents = getEvents.drop(['Event Street Side','Street Closure Type', 'Community Board'], axis=1)

#Remove Nan Values from the dataset
dropEvent1 = getRemEvents.dropna(subset=['Event ID', 'Event Name'],how = "all")
dropEvent2 = dropEvent1.dropna(subset=['Start Date/Time', 'End Date/Time'],how = "all")
dropEvent3 = dropEvent2.dropna(subset=['Event Borough'],how = "all")
dropEvent4 = dropEvent3.dropna(subset=['Police Precinct'],how = "all")

#Copy of Cleaned Dataset
eventDf = dropEvent4.copy()

def getCleanedEventsData():
    return eventDf


