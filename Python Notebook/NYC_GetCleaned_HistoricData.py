#!/usr/bin/env python
# coding: utf-8

# Data exploration
import pandas as pd

# Numerical
import numpy as np

## Read CSV
crimes_original = pd.read_csv('../Data/NYPD_Complaint_Data_Cleaned.csv', low_memory=False)
crimes_original.isnull().sum()
crimes_original.describe(include = [np.number])
crimes_original.describe(include = [np.object])
crime_remCol = crimes_original.drop(['CMPLNT_NUM','KY_CD','JURIS_DESC','ADDR_PCT_CD','PD_CD','PARKS_NM','HADEVELOPT','X_COORD_CD','Y_COORD_CD'], axis=1)

#print(crime_remCol)

# Fill CMPLNT_TO_DT NaNs with CMPLNT_FR_DT values.
crime_remCol['CMPLNT_TO_DT'].fillna(crime_remCol['CMPLNT_FR_DT'], axis = 0, inplace = True)
crime_remCol['CMPLNT_FR_DT'].fillna(crime_remCol['CMPLNT_TO_DT'], axis = 0, inplace = True)

# Fill CMPLNT_TO_TM NaNs with CMPLNT_FR_TM values.
crime_remCol['CMPLNT_TO_TM'].fillna(crime_remCol['CMPLNT_FR_TM'], axis = 0, inplace = True)
crime_remCol['CMPLNT_FR_TM'].fillna(crime_remCol['CMPLNT_TO_TM'], axis = 0, inplace = True)

# All NaNs from 'PD_DESC' series are filled with copy of 'OFNS_DESC' values
crime_remCol['OFNS_DESC'] = np.where(crime_remCol['OFNS_DESC'].isnull(), crime_remCol['PD_DESC'], crime_remCol['OFNS_DESC'])
# And vice versa
crime_remCol['PD_DESC'] = np.where(crime_remCol['PD_DESC'].isnull(), crime_remCol['OFNS_DESC'], crime_remCol['PD_DESC'])

#Remove Nan for premises.
prem_typ_desc_copy = crime_remCol['PREM_TYP_DESC'].copy(deep = True)
prem_typ_desc_copy_rand = prem_typ_desc_copy.value_counts(normalize = True).sort_values(ascending = False)

# Fill PREM_TYP_DESC values NaN values with values from locations of other incidents.
crime_remCol['PREM_TYP_DESC'] = crime_remCol['PREM_TYP_DESC'].apply(lambda x: np.random.choice([x for x in crime_remCol.prem_typ_desc],
                          replace = True, p = prem_typ_desc_copy_rand ) if (x == np.nan) else x).astype(str)


#Fill the location of occurance desc with max count
crime_remCol['LOC_OF_OCCUR_DESC'].fillna(value ='INSIDE', axis = 0, inplace = True)
#crime_remCol.isnull().sum()

crime_filt1 = crime_remCol.dropna(subset=['CMPLNT_FR_DT', 'CMPLNT_TO_DT'],how = "all")
crime_filt2 = crime_filt1.dropna(subset=['CMPLNT_FR_TM', 'CMPLNT_TO_TM'],how = "all")
crime_filt3 = crime_filt2.dropna(subset=['CRM_ATPT_CPTD_CD'],how = "all")
crime_filt4 = crime_filt3.dropna(subset=['BORO_NM'],how = "all")
crime_filt5 = crime_filt4.dropna(subset=['Latitude','Longitude','Lat_Lon'],how = "all")
cleanedDf = crime_filt5.copy()

def getCleanedDataFrame():
    return cleanedDf

#crime_filt5.shape
#crime_filt5.isnull().sum()

#To check the count of rows
#crime_filt5['LOC_OF_OCCUR_DESC'].value_counts(dropna=True)

data = crime_filt5.iloc[:,0:15].values
cleanedData = pd.DataFrame(data)

def getCleanedData():
    return cleanedData




