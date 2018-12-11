#!/usr/bin/env python
# coding: utf-8

# In[3]:


###------NYC Crimes by precinct data -----###


# In[4]:


# Data exploration
import pandas as pd
import numpy as np
from NYC_GetCleaned_HistoricData import getCleanedDataFrame
from data_utility import getCurrentDirectory, readCsvFile
from NYC_GetCleaned_TotalPopulation import getMeanPopulation


# In[5]:


#Get cleaned historic dataframe
crimes_historic = getCleanedDataFrame()


# In[6]:


#Get the total mean population of NYC based on borough
""" Bronx - 1, Brooklyn - 2, Manhattan - 3, Queens - 4, Staten Island - 5 """

bronxPop = getMeanPopulation('Bronx', 1)
brookylnPop = getMeanPopulation('Brooklyn', 2)
manhattanPop = getMeanPopulation('Manhattan', 3)
queensPop = getMeanPopulation('Queens', 4)
statIslandPop = getMeanPopulation('Staten Island', 5)


# In[7]:


print(bronxPop)
print(brookylnPop)
print(manhattanPop)
print(queensPop)
print(statIslandPop)


# In[9]:


## Read CSV
precinct_data = readCsvFile('../Data/NYC_Precinct_Data.csv')
precinct_data.head()


# In[14]:


crimes_historic.info()


# In[15]:


precinct_data.info()


# In[12]:


crimes_historic.shape


# In[13]:


precinct_data.shape


# In[16]:


inner_data=pd.merge(crimes_historic, precinct_data, left_on=['ADDR_PCT_CD'], right_on=['Precinct'], how='inner')
inner_data.shape


# In[17]:


inner_data.head()


# In[1]:


def getPrecinctData():
    return inner_data


# In[ ]:




