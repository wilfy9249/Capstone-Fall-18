#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd


# In[18]:


from Data_Filter_for_model_selection import getFinalData


# In[3]:


#Get cleaned historic dataframe
crimes_pred_data = getFinalData()
crimes_pred_data.head()


# In[4]:


crimes_pred_data['PREM_TYP_DESC'].value_counts(dropna=False)


# In[5]:


data_morn = crimes_pred_data.drop(['PD_DESC','CRM_ATPT_CPTD_CD','Precinct','Time','Afternoon','Evening','Night'],axis=1)


# In[6]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
data_morn['OFNS_DESC'] = labelencoder_X.fit_transform(data_morn['OFNS_DESC'])
data_morn['PREM_TYP_DESC'] = labelencoder_X.fit_transform(data_morn['PREM_TYP_DESC'])
data_morn['LAW_CAT_CD'] = labelencoder_X.fit_transform(data_morn['LAW_CAT_CD'])
data_morn['BORO_NM'] = labelencoder_X.fit_transform(data_morn['BORO_NM'])
data_morn['LOC_OF_OCCUR_DESC'] = labelencoder_X.fit_transform(data_morn['LOC_OF_OCCUR_DESC'])
data_morn['Morning'] = labelencoder_X.fit_transform(data_morn['Morning'])

# #Encoding the Dependent Variable
# labelencoder_y = LabelEncoder() 
# Y = labelencoder_y.fit_transform(Y)


# In[7]:


data_morn.head()


# In[8]:


data_morn['PREM_TYP_DESC'].value_counts(dropna=False)


# In[9]:


X = data_morn.iloc[:, 0:5].values
Y = data_morn.iloc[:, 5].values


# In[10]:


X[:,0:5]


# In[11]:


Y


# In[12]:


X.shape


# In[13]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)


# In[14]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[15]:


# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_


# In[16]:


explained_variance


# In[17]:


def getXTrain():
    return X_train

def getXTest():
    return X_test

def getYTrain():
    return Y_train

def getYTest():
    return Y_test


# In[24]:


# Visualising the results

from matplotlib.colors import ListedColormap
def getVisuals(X,Y,classifier,yLabel,title):
        X_set, y_set = X, Y
        X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                             np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
        plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                     alpha = 0.75, cmap = ListedColormap(('red', 'green')))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                        c = ListedColormap(('red', 'green'))(i), label = j)
        plt.title(title)
        plt.xlabel('Crime')
        plt.ylabel(yLabel)
        plt.legend()
        plt.show()


# In[ ]:




