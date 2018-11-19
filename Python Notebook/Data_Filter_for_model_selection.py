#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Data exploration
import pandas as pd

# Numerical
import numpy as np


# In[27]:


#import the functions from their corresponding files
from NYC_GetCleaned_PrecinctData import getPrecinctData
from data_utility import filterData,sortValue,getCount


# In[3]:


#Get cleaned data from NYC_GetCleaned_HistoricData
crimes_original = getPrecinctData()
#crimes_original.head()


# In[4]:


crimes_original.shape


# In[5]:


"""Create a datetime index of times that crimes were reported to have been committed"""

dfCopy = crimes_original.copy()
def eliminate_dates(x):
    if x[2] > '2050':
        x = None
    elif x[2] < '2010':
        x = None
    else: 
        aa= '/'.join(x)
        return (aa)

#get dummy columns for crime categories
#dfCopy = dfCopy.join(dfCopy['LAW_CAT_CD'].str.get_dummies())

#dfCopy = dfCopy.join(dfCopy['BORO_NM'].str.get_dummies())

#Create index with DateTime
dfCopy['CMPLNT_FR_DT'] = dfCopy['CMPLNT_FR_DT'].str.split("/")
dfCopy['CMPLNT_FR_DT'] = dfCopy['CMPLNT_FR_DT'].apply(lambda x: eliminate_dates(x))

#Combing date and time columns
dfCopy['StartTime'] = dfCopy['CMPLNT_FR_DT'] +' '+dfCopy['CMPLNT_FR_TM']
#dfCopy['StartTime'] = dfCopy['CMPLNT_FR_TM']
dfCopy['StartTime'] = pd.to_datetime(dfCopy['StartTime'])


#set full date as index
dfCopy.set_index('StartTime', inplace=True)        

dfCopy.head()


# In[28]:


df_data = dfCopy.copy()
df_data['Time'] = df_data.index.hour


# In[29]:


df_data['Morning'] = df_data['Time'].between(06.01,12.0)
df_data['Afternoon'] = df_data['Time'].between(12.01,17.0)
df_data['Evening'] = df_data['Time'].between(17.01,20.0)
df_data['Night'] = df_data['Time'].between(20.01,06.0)


# In[30]:


data = df_data.drop(['CMPLNT_FR_DT','CMPLNT_FR_TM','CMPLNT_TO_DT','CMPLNT_TO_TM','RPT_DT','ADDR_PCT_CD','Latitude','Longitude','Lat_Lon','Shape_Area','Shape_Leng','the_geom'],axis=1)


# In[31]:


filter_data = filterData(data,'Precinct',44)


# In[32]:


temp = filter_data['OFNS_DESC'].value_counts().head(3).reset_index()
temp


# In[33]:


ofns_data = filter_data.loc[filter_data['OFNS_DESC'].isin(temp['index'])]


# In[34]:


ofns_data['OFNS_DESC'].value_counts()


# In[35]:


ofns_data


# In[36]:


ofns_data.shape


# In[37]:


temp_prem = ofns_data['PREM_TYP_DESC'].value_counts().head(3).reset_index()
temp_prem


# In[38]:


final_data = ofns_data.loc[ofns_data['PREM_TYP_DESC'].isin(temp_prem['index'])]


# In[39]:


final_data.shape


# In[40]:


final_data


# In[41]:


def getFinalData():
    return final_data


# In[42]:


final_data['PREM_TYP_DESC'].value_counts(dropna=False)


# In[43]:


final_data.shape


# In[ ]:




