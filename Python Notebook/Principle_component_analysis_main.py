#!/usr/bin/env python
# coding: utf-8

# In[10]:


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd


# In[15]:


def performPCA(X,Y):
    
    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)
    
    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    # Applying PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components = 2)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    explained_variance = pca.explained_variance_ratio_
    
    split_data = [X_train, X_test, Y_train, Y_test]
    
    print("The Variance is",explained_variance)
    
    return split_data


# In[16]:


def getConfusionMatrix(split_data):
    
    X_train = split_data[0]
    X_test = split_data[1]
    Y_train = split_data[2]
    Y_test = split_data[3]

    # Fitting Logistic Regression to the Training set
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(random_state = 0)
    classifier.fit(X_train, Y_train)
    
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    
    #Get Accuracy Score
    from sklearn.metrics import accuracy_score
    print ("Accuracy:",accuracy_score(Y_test,y_pred))
    print("")
    
    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(Y_test, y_pred)
    print("Confusion Matrix is as follows")
    print(cm)
   
def getConfusionMatrixNB(split_data):
    
    X_train = split_data[0]
    X_test = split_data[1]
    Y_train = split_data[2]
    Y_test = split_data[3]

    # Fitting Naive Bayes to the Training set
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(X_train, Y_train)
    
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    
    #Get Accuracy Score
    from sklearn.metrics import accuracy_score
    print ("Accuracy:",accuracy_score(Y_test,y_pred))
    print("")
    
    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(Y_test, y_pred)
    print("Confusion Matrix is as follows")
    print(cm)
    return classifier


# In[13]:


def getXTrain():
    return X_train

def getXTest():
    return X_test

def getYTrain():
    return Y_train

def getYTest():
    return Y_test


# In[14]:


# Visualising the results

from matplotlib.colors import ListedColormap
def getVisuals(X,Y,classifier,yLabel,title):
        X_set, y_set = X, Y
        X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                             np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
        plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                     alpha = 0.5, cmap = ListedColormap(('red', 'green')))
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

