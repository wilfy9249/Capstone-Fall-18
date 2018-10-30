#!/usr/bin/env python
# coding: utf-8

###------NYC TOTAL POPULATION BY BOROUGH : DATA CLEANING-----###

#Import Packages
import pandas as pd
from data_utility import getCurrentDirectory, readCsvFile

#Get Current Working Directory
listDir = getCurrentDirectory()
listDir

## Read CSV
total_pop = readCsvFile('../Data/Population_by_Borough_NYC.csv')
total_pop.head()

#Drop unncessary columns of the population dataset
dropColPop = total_pop.copy()
dropColPop.drop(['Age Group','1950 - Boro share of NYC total','1960 - Boro share of NYC total',
                 '1970 - Boro share of NYC total','1980 - Boro share of NYC total',
                 '1990 - Boro share of NYC total','2000 - Boro share of NYC total',
                 '2010 - Boro share of NYC total','2020 - Boro share of NYC total',
                 '2030 - Boro share of NYC total','2040 - Boro share of NYC total'], axis=1, inplace=True)
dropColPop


def getMeanPopulation(borough, index):
    boroList = dropColPop['Borough'][1:6]
    meanPop = 0.0
    
    for boro in boroList:
        if boro.strip() == borough:
            numA = dropColPop['2010'][index]
            numB = dropColPop['2020'][index]
            meanPop = (numA + numB)/2
    return meanPop

