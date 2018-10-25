#!/usr/bin/env python
# coding: utf-8

# In[1]:


###------NYC DATA CRIME-----###


# In[2]:


# Data exploration
import pandas as pd

# Numerical
import numpy as np


# In[3]:


import os
#rint(os.listdir('../'))


# In[4]:



## Read CSV
crimes_original = pd.read_csv('../Data/NYPD_Complaint_Data_Historic.csv', low_memory=False)
#crimes_original.head()


# In[5]:


## Rows, column count
#crimes_original.shape


# In[6]:


#crimes_original.describe(include = [np.number])


# In[7]:


#crimes_original.describe(include = [np.object])


# In[8]:


crime_remCol = crimes_original.drop(['CMPLNT_NUM','KY_CD','JURIS_DESC','ADDR_PCT_CD','PD_CD','PARKS_NM','HADEVELOPT','X_COORD_CD','Y_COORD_CD'], axis=1)
#print(crime_remCol)


# In[9]:


#crime_remCol.info()


# In[10]:


crime_filter = crime_remCol['CMPLNT_FR_DT'].value_counts(dropna=False)
#print(crime_filter)


# In[11]:


# Fill CMPLNT_TO_DT NaNs with CMPLNT_FR_DT values.
crime_remCol['CMPLNT_TO_DT'].fillna(crime_remCol['CMPLNT_FR_DT'], axis = 0, inplace = True)

# Fill CMPLNT_TO_TM NaNs with CMPLNT_FR_TM values.
crime_remCol['CMPLNT_TO_TM'].fillna(crime_remCol['CMPLNT_FR_TM'], axis = 0, inplace = True)


# In[12]:


# All NaNs from 'PD_DESC' series are filled with copy of 'OFNS_DESC' values
crime_remCol['OFNS_DESC'] = np.where(crime_remCol['OFNS_DESC'].isnull(), crime_remCol['PD_DESC'], crime_remCol['OFNS_DESC']) # There is pandas equivalent of np.where -> https://stackoverflow.com/questions/38579532/pandas-equivalent-of-np-where
# And vice versa
crime_remCol['PD_DESC'] = np.where(crime_remCol['PD_DESC'].isnull(), crime_remCol['OFNS_DESC'], crime_remCol['PD_DESC'])


# In[13]:


#Remove Nan for premises.
prem_typ_desc_copy = crime_remCol['PREM_TYP_DESC'].copy(deep = True)
prem_typ_desc_copy_rand = prem_typ_desc_copy.value_counts(normalize = True).sort_values(ascending = False)

# Fill PREM_TYP_DESC values NaN values with values from locations of other incidents.
crime_remCol['PREM_TYP_DESC'] = crime_remCol['PREM_TYP_DESC'].apply(lambda x: np.random.choice([x for x in crime_remCol.prem_typ_desc],
                          replace = True, p = prem_typ_desc_copy_rand ) if (x == np.nan) else x).astype(str)


# In[14]:


# Sanity check
#print(crime_remCol['CMPLNT_TO_DT'].notnull().value_counts()) 
#print(crime_remCol['CMPLNT_FR_DT'].notnull().value_counts())
"""nt(crime_remCol['CMPLNT_TO_TM'].notnull().value_counts()) 
print(crime_remCol['CMPLNT_FR_TM'].notnull().value_counts())
print(crime_remCol['OFNS_DESC'].notnull().value_counts()) 
print(crime_remCol['PD_DESC'].notnull().value_counts())
print(crime_remCol['PREM_TYP_DESC'].notnull().value_counts())"""


# In[15]:


X = crime_remCol.iloc[:,0:15].values
#rint(X)


# In[16]:


from sklearn.base import TransformerMixin

class DataFrameImputer(TransformerMixin):
    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)

data = crime_remCol.iloc[:,0:15].values
XBefore = pd.DataFrame(data)
xt = DataFrameImputer().fit_transform(XBefore)
"""
print('before...')
print(XBefore)
print('after...')
print(xt)
"""



# In[17]:


# Sanity Check
final = xt.iloc[:,0:15].notnull().values
#rint(final)


# In[18]:


def getCleanedData():
    return xt


# In[ ]:




