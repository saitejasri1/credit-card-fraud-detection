#!/usr/bin/env python
# coding: utf-8

# # Description:
# # We have 30 features in the data: Time, Amount, V1,...,V28. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction amount. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.
# # In this dataset, we have about 285k transactions of which 492 of them are fraudulent. So the data is quite unbalanced. We will be careful about which evaluation metrics to use. Score accuracy or AUC cannot be used. We will use precision score, recall score, F1-score and confusion matrix as our metrics

# # Importing the libraries

# In[1]:


import pandas as pd   #data prepocessing and to read csv files.
import numpy as np     #linear algebra
import matplotlib.pyplot as plt # to plot haetmaps and confusion matrix
import seaborn as sns
import math    # for mathematical operations
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import time
from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier
from catboost import Pool, CatBoostClassifier, cv
import xgboost as xgb
import lightgbm as lgb
import plotly.offline as py
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
from plotly import tools
init_notebook_mode(connected=True)  
import plotly.figure_factory as ff
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


# # Confiuring or Fetching the dataset from kaggle

# In[2]:


import warnings
warnings.filterwarnings('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Importing the dataset

# In[3]:


# Read data
df=pd.read_csv('creditcard.csv') # df stands for dataframe


# # Checking the features of the dataset

# In[4]:


df.head()


# # Describing the data

# In[5]:


df.describe()


# In[6]:


df.info()


# In[7]:


df.shape


# In[8]:


df.Time.describe() # check time in general


# df.Time.describe() # check time in general

# # Imbalance in the data

# In[9]:


# Determine number of fraud cases in dataset 
fraud = df[df['Class'] == 1] 
valid = df[df['Class'] == 0] 
print(len(fraud))
outlierFraction = len(fraud)/float(len(valid)) 
print("outlier fraction is:",outlierFraction) 
print('Fraud Cases: {}'.format(len(df[df['Class'] == 1]))) 
print('Valid Transactions: {}'.format(len(df[df['Class'] == 0]))) 

print ("total percentage of fradulant transaction is : ",round(fraud.shape[0]/df.shape[0],3))


# # There are 492 fraud cases and 284.315 nonfraudulent cases in the dataset. The percentage of no fraud cases is quite higher than fraud cases.

# # Print the amount details for Total Transaction

# In[10]:


print("Amount details of the total transaction") 
df.Amount.describe() # check Amount in general


# # Print the amount details for Fradulent Transaction

# In[11]:


# Check amount of fraudulent transactions
print("Amount details of the fradulent transaction") 
df[df.Class == 1].Amount.describe()


# # Print the amount details for Non-Fradulent Transaction

# In[12]:


# Check amount of non-fraudulent transactions
print("Amount details of the Non-fradulent transaction") 
df[df.Class == 0].Amount.describe()


# # Histogram plot for fraud and non-fraud

# In[13]:


plt.figure(figsize=(12,5))
sns.countplot(df['Class'])
plt.title("Fraud vs No Fraud Transaction Distributions")
plt.xticks(range(2), ["No Fraud", "Fraud"])
plt.show() 


# # Exploratory Data Analysis 

# # (Distribution based on the amount.)

# In[14]:


plt.figure(figsize=(16,5))

plt.subplot(1,2,1)
sns.distplot(valid['Amount'])
plt.title("Amount Distribution of Non-Fraudulent Transaction")

plt.subplot(1,2,2)
sns.distplot(fraud['Amount'])
plt.title("Amount Distribution of Fraudulent Transaction")

plt.show()


# # (Distribution based on the time.)

# In[15]:


plt.figure(figsize=(16,5))

plt.subplot(1,2,1)
plt.scatter(valid.Time, valid.Amount)
plt.title("Time Distribution of Non-Fraudulent Transaction")
plt.xlabel("time")
plt.ylabel("amount")

plt.subplot(1,2,2)
plt.scatter(fraud.Time, fraud.Amount)
plt.title("Time Distribution of Fraudulent Transaction")
plt.xlabel("time")
plt.ylabel("amount")

plt.show()


# #  Correlation of variables heat map

# In[16]:


correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, cmap = 'coolwarm', center = 0)


# # Resampling

# In[17]:


# Specify features columns
X = df.drop(columns="Class", axis=0)

# Specify target column
y = df["Class"]

# Import required library for resampling
from imblearn.under_sampling import RandomUnderSampler

# Instantiate Random Under Sampler
rus = RandomUnderSampler(random_state=42)

# Perform random under sampling
df_data, df_target = rus.fit_resample(X, y)

# Visualize new classes distributions
sns.countplot(df_target).set_title('Balanced Data Set')


# In[18]:


# Specify features columns
X = df.drop(columns="Class", axis=0)

# Specify target column
y = df["Class"]

# Import required library for resampling
from imblearn.over_sampling import RandomOverSampler

# Instantiate Random Under Sampler
rus = RandomOverSampler(random_state=42)

# Perform random over sampling
df_data, df_target = rus.fit_resample(X, y)

# Visualize new classes distributions
sns.countplot(df_target).set_title('Balanced Data Set')


# # DataPreprocessing_Scaling time, amount

# In[19]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc2 = StandardScaler()
df[['Time']] = sc.fit_transform(df[['Time']])
df[['Amount']] = sc2.fit_transform(df[['Amount']])


# In[20]:


#Making a new dataset with 1:1 fraud and non_frauds
# we know that no.of frauds = 492

fraud_data = df.loc[df['Class'] == 1]
non_fraud_data = df.loc[df['Class'] == 0]

#Selecting 492 rows from non_fraud_data
selected_non_fraud_data = non_fraud_data.sample(492)

#Combining to form new dataset
new_data = pd.concat([fraud_data, selected_non_fraud_data])
sns.countplot(x = 'Class', data = new_data)


# # Checking correlation of different variables to choose what to retain

# In[21]:


correl = new_data.corr()
class_correl = correl[['Class']]
negative = class_correl[class_correl.Class< -0.5]
positive = class_correl[class_correl.Class> 0.5]
print("negative")
print(negative)
print("positive")
print(positive)


# # visualizing the features with high negative correlation

# In[22]:


f, axes = plt.subplots(nrows=2, ncols=4, figsize=(26,16))

f.suptitle('Features With High Negative Correlation', size=35)
sns.boxplot(x="Class", y="V3", data=new_data, ax=axes[0,0])
sns.boxplot(x="Class", y="V9", data=new_data, ax=axes[0,1])
sns.boxplot(x="Class", y="V10", data=new_data, ax=axes[0,2])
sns.boxplot(x="Class", y="V12", data=new_data, ax=axes[0,3])
sns.boxplot(x="Class", y="V14", data=new_data, ax=axes[1,0])
sns.boxplot(x="Class", y="V16", data=new_data, ax=axes[1,1])
sns.boxplot(x="Class", y="V17", data=new_data, ax=axes[1,2])
f.delaxes(axes[1,3])


# # visualizing the features with high positive correlation

# In[23]:


f, axes = plt.subplots(nrows=1, ncols=2, figsize=(18,9))

f.suptitle('Features With High Positive Correlation', size=20)
sns.boxplot(x="Class", y="V4", data=new_data, ax=axes[0])
sns.boxplot(x="Class", y="V11", data=new_data, ax=axes[1])


# # Removing Extreme Outliers

# In[24]:


Q25 = new_data.quantile(0.25)
Q75 = new_data.quantile(0.75)
IQR = Q75-Q25
threshold = 2.5*IQR
print(threshold)
final_data = new_data[~((new_data < (Q25 - threshold)) |(new_data > (Q75 + threshold))).any(axis=1)]
print("Length of data before:",len(new_data))
print("Length of data after:", len(final_data))
print("Extreme outliers:", len(new_data)-len(final_data))


# # Train test split

# In[25]:


X = final_data.iloc[:, :30].values
y = final_data.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[26]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# # Model Building

# # Gradient Boosting algorithms

# # 1. XGBoost

# In[29]:


from xgboost import XGBClassifier

xgb = XGBClassifier(n_jobs=-1, random_state=42, n_estimators=120, max_depth = 5, min_samples_leaf=5)
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)
print(classification_report(y_test, y_pred))
from sklearn.metrics import plot_confusion_matrix, accuracy_score
plot_confusion_matrix(xgb,X_test , y_test, cmap = plt.cm.Blues)
print(confusion_matrix(y_test, y_pred))


# In[36]:



print("Train accuracy",xgb.score(X_train, y_train))
print("Test accuracy",xgb.score(X_test, y_test))


# # 2. LightGBM

# In[ ]:





# In[ ]:


#Accuracy Score
print('accuracy is ',accuracy_score(y_pred,y_test))


# # 3. CatBoost

# In[32]:


Model=CatBoostClassifier(eval_metric='Accuracy',use_best_model=True,random_seed=42)
Model.fit(X_train,y_train,eval_set=(X_test,y_test))


# In[33]:


## CatBoost
y_pred=Model.predict(X_test)
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_pred,y_test))
from sklearn.metrics import plot_confusion_matrix, accuracy_score
plot_confusion_matrix(Model,X_test , y_test, cmap = plt.cm.Blues)
#Accuracy Score
print('accuracy is ',accuracy_score(y_pred,y_test))


# # 4. SGB

# In[35]:


## Gradient Boosting Machine
Model = GradientBoostingClassifier(n_estimators=100, random_state=9)
Model.fit(X_train, y_train)
y_pred = Model.predict(X_test)

# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
from sklearn.metrics import plot_confusion_matrix, accuracy_score
plot_confusion_matrix(Model,X_test , y_test, cmap = plt.cm.Blues)
# Accuracy score

print('accuracy is',accuracy_score(y_pred,y_test))


# # 5. AdaBoost

# In[ ]:


from sklearn.ensemble import AdaBoostClassifier
Model = AdaBoostClassifier(n_estimators=100, random_state=9)
Model.fit(X_train, y_train)
y_pred = Model.predict(X_test)

# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# Accuracy score

print('accuracy is',accuracy_score(y_pred,y_test))


# In[ ]:





# In[ ]:




