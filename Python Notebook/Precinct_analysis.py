#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Data exploration
import pandas as pd

# Numerical
import numpy as np
import data_utility

#Data Visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#import the functions from their corresponding files
from NYC_GetCleaned_PrecinctData import getPrecinctData
from data_utility import filterData
from NYC_GetCleaned_TotalPopulation import getMeanPopulation


# In[3]:


#Get cleaned data from NYC_GetCleaned_HistoricData
crimes_original = getPrecinctData()
crimes_original.head()


# In[4]:


#Get a list of NYC Boroughs
borolist = (crimes_original['BORO_NM'].unique()).tolist()


# In[5]:


#Get the list of number of precinct in each borough
totalPrecinct = len(crimes_original['Precinct'].unique().tolist())


# In[6]:


#Get the list of area of each borough
totalAreaNYC = int(sum(crimes_original['Shape_Area'].unique().tolist()))


# In[26]:


#Define the arrays  
areaList = []
countlist = []
pop = []
popPercent=[]


# In[8]:


def getPrecinctCount(borolist):
        countlist.clear()
        for boro in borolist:
            boro_data = filterData(crimes_original,'BORO_NM', boro)
            precinctCount = len(boro_data['Precinct'].unique().tolist())*100/totalPrecinct
            countlist.append(round(precinctCount,2))
        return countlist


# In[9]:


def getArea(borolist):
        areaList.clear()
        for boro in borolist:
            area_data = filterData(crimes_original,'BORO_NM', boro)
            area = sum(area_data['Shape_Area'].unique().tolist())*100/totalAreaNYC
            areaList.append(round(area,2))
        return areaList


# In[10]:


def getPopulation(borolist):
    pop.clear()
    popPercent.clear()
    i=1
    for boro in borolist:
        popValue = getMeanPopulation(boro,i)
        pop.append(popValue)
        i = i+1
    popsum = int(sum(pop))
    for value in pop:
        popPercentValue = value*100/popsum
        popPercent.append(round(popPercentValue,2))
    return popPercent    
    


# In[13]:


precinct_boro = pd.DataFrame({'Boroughs':borolist,'Precinct (%)':getPrecinctCount(borolist),'Area (%)':getArea(borolist),'Population (%)':getPopulation(borolist)})


# In[15]:


precinct_boro['Precinct/Area'] = precinct_boro['Precinct (%)']/precinct_boro['Area (%)']


# In[27]:


precinct_boro['Precinct/Population'] = precinct_boro['Precinct (%)']/precinct_boro['Population (%)']


# In[16]:


precinct_boro


# In[23]:


a4_dims = (14, 8)
fig, ax = plt.subplots(figsize=a4_dims)
sns.barplot(x='Boroughs', y='Precinct/Population', data=precinct_boro, ax=ax)
plt.ylabel('Ratio of Precinct by Population')
plt.title('Precinct per Population')
plt.show()


# In[22]:


a4_dims = (14, 8)
fig, ax = plt.subplots(figsize=a4_dims)
sns.barplot(x='Boroughs', y='Precinct/Area', data=precinct_boro, ax=ax)
plt.ylabel('Ratio of Precinct per Area')
plt.title('Precinct per Area')
plt.show()


# In[25]:


a4_dims = (14, 8)
fig, ax = plt.subplots(figsize=a4_dims)
plt.plot(precinct_boro['Boroughs'], precinct_boro['Precinct/Population'], color='g')
plt.plot(precinct_boro['Boroughs'],precinct_boro['Precinct/Area'], color='orange')
plt.xlabel('Boroughs')
plt.ylabel('By Population Ratio')
plt.title('Comparison between Precinct by Area and Precinct by Population')
plt.legend()
plt.show()


# In[ ]:




