#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


#function to get current directory
def getCurrentDirectory():
    import os
    print(os.listdir('../'))


# In[3]:


#function to read csv file
def readCSVfile(path):
    crimes_original = pd.read_csv(path, low_memory=False)


# In[5]:





# In[ ]:




