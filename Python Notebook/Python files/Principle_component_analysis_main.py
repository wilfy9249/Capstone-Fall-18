#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import warnings

# In[3]:


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


# In[4]:


def applyModel(split_data):
    
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
    print("")
    print("Accuracy:",accuracy_score(Y_test,y_pred))
    print("")
    warnings.filterwarnings("ignore")
    return classifier
    


# In[ ]:


def applyModelNB(split_data):
    
    X_train = split_data[0]
    X_test = split_data[1]
    Y_train = split_data[2]
    Y_test = split_data[3]

    # Fitting Logistic Regression to the Training set
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(X_train, Y_train)
    
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    
    #Get Accuracy Score
    from sklearn.metrics import accuracy_score
    print ("Accuracy:",accuracy_score(Y_test,y_pred))
    print("")
    warnings.filterwarnings("ignore")
    return classifier


# In[5]:


def getConfusionMatrix(X_test,Y_test,classifier):
    
    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    import warnings
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(Y_test, y_pred)
    print("Confusion Matrix is as follows")
    print(cm)
    print("")
    
    if(cm.shape == (2,2)):
        TP = cm[1, 1]
        TN = cm[0, 0]
        FP = cm[0, 1]
        FN = cm[1, 0]

        from sklearn import metrics

        # calculate classification error
        classification_error = 1 - metrics.accuracy_score(Y_test, y_pred)
        print("Classification Error")
        print(classification_error)
        print("")

        # calculate Sensitivity or Recall
        sensitivity = metrics.recall_score(Y_test, y_pred)
        print("Sensitivity or Recall")
        print(sensitivity)
        print("")

        # calculate Specificity 
        specificity = TN / (TN + FP)
        print("Specificity")
        print(specificity)
        print("")

        # calculate False Positive Rate 
        false_positive_rate = 1 - specificity
        print("False Positive Rate")
        print(false_positive_rate)
        print("")    

        # calculate Precision 
        #precision = TP / float(TP + FP)
        precision = metrics.precision_score(Y_test, y_pred)
        print("Precision")
        print(precision)
        print("")    
    else:
        
        print("Cannot Calculate metrics from 3X3 confusion matrix")
    warnings.filterwarnings("ignore")
    return cm


# In[6]:


def getMetrics(confusion):
    
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]

    from sklearn import metrics
    
    # calculate classification error
    classification_error = (FP + FN) / float(TP + TN + FP + FN)
    print("Classification Error")
    print(classification_error)
    print("")
    
    # calculate Sensitivity or Recall
    sensitivity = TP / float(FN + TP)
    print("Sensitivity or Recall")
    print(sensitivity)
    print("")
    
    # calculate Specificity 
    specificity = TN / (TN + FP)
    print("Specificity")
    print(specificity)
    print("")
    
    # calculate False Positive Rate 
    false_positive_rate = FP / float(TN + FP)
    print("False Positive Rate")
    print(false_positive_rate)
    print("")    
 
    # calculate Precision 
    #precision = TP / float(TP + FP)
    precision = metrics.precision_score(y_test, y_pred_class)
    print("Precision")
    print(precision)
    print("")    


# In[7]:


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
    


# In[8]:


def getXTrain(split_data):
    return split_data[0]

def getXTest(split_data):
    return split_data[1]

def getYTrain(split_data):
    return split_data[2]

def getYTest(split_data):
    return split_data[3]


# In[9]:


# Visualising the results

from matplotlib.colors import ListedColormap
def getVisuals(X,Y,classifier,title):
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
        plt.xlabel('Actual Crime Rate')
        plt.ylabel('Predicted Crime Rate')
        plt.legend()
        plt.show()

