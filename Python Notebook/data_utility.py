#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os

# In[2]:


#function to get current directory
def getCurrentDirectory():
    listDirectory = os.listdir('../')
    return listDirectory


# In[3]:


#function to read csv file
def readCsvFile(path):
    crimes_original = pd.read_csv(path, low_memory=False)
    return crimes_original


# In[4]:


#function to filter Data
def filterData(data,column,value):
    filterData = data.loc[data[column] == value]
    return filterData
    


# In[5]:


#function to get count of a value
def getCount(data,column,columnName):    
    data_count = pd.DataFrame({columnName:data.groupby(column).size()}).reset_index()
    return data_count


# In[7]:


#function to sort
def sortValue(data,column,ascBoolean):
    sorted_data = data.sort_values(column,ascending = ascBoolean)
    return sorted_data


# In[ ]:




