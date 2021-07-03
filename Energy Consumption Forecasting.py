#!/usr/bin/env python
# coding: utf-8

# # Project - Hourly Energy Consumption Analysis and Forecasting
# PJM Interconnection LLC (PJM) is a regional transmission organization (RTO) in the United States. The hourly energy consumption data is available on PJM's website. The data set is also available part of Kaggle competition on Hourly Energy Consumption. This project first analyzes the energy consumption data of PJM East region (2001-2018) and than applies XGBoost forecasting algorithm.

# In[1]:


#Import Python Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error
plt.style.use('fivethirtyeight')


# ## Gather Data

# In[2]:


#Read in PJM Hourly Power Consumption data file to dataframe df
df = pd.read_csv('PJME_hourly.csv')


# ## Assess and Clean Data
# The dataset has no null values, and is preprocessed already. The Leap year show higher value logs compared to Other years since there is additional day. 2018 shows lower value since the data is unavailable beyond July

# In[3]:


# Check number of Columns and Rows
num_rows = df.shape[0] #Provide the number of rows in the dataset
num_cols = df.shape[1] #Provide the number of columns in the dataset

print(num_rows, num_cols) 


# In[4]:


#Check the Data
print(df.tail())


# In[5]:


#Converting Datetime column to datetime %Y-%m-%d %H:%M:%S format
df['Datetime'] = pd.to_datetime(df.Datetime, format='%Y-%m-%d %H:%M:%S')


# In[6]:


#Checking Null values
df.isna().sum()


# In[7]:


# Check Total Value Count for Each Year
Yearly_Value_Counts = df['Datetime'].dt.year.value_counts(sort=False)
print(Yearly_Value_Counts)


# ## Explore and Analyze Data

# In[8]:


#Statistics Describing PJME Power Consumption
df['PJME_MW'].describe().T


# In[9]:


# Adding Hour of Day and Month of Year Columns to the Dataframe
df['hour'] = df.Datetime.dt.hour
df['month'] = df.Datetime.dt.month


# In[47]:


# Calculate Year-wise Energy Consumption Mean and Std deviation
df.groupby(pd.Grouper(key='Datetime',freq='Y')).agg({'PJME_MW': ['mean', 'std']})


# In[46]:


# Calculate Month-wise Energy Consumption Mean and Std deviation
df.groupby(['month']).agg({'PJME_MW': ['mean', 'std']})


# In[45]:


# Calculate Hour-wise Energy Consumption Mean and Std deviation
df.groupby(['hour']).agg({'PJME_MW': ['mean', 'std']})


# In[10]:


#Histogram of Power Consumption for PJME
df['PJME_MW'].plot.hist(figsize=(15, 5), bins=200, title='Distribution of PJME Load')


# ## Visualize Data

# In[11]:


# PJME Power Consumption Trend Chart
df[['PJME_MW','Datetime']].plot(x='Datetime',
                                     y='PJME_MW',
                                     kind='scatter',
                                     figsize=(14,4),
                                     title='PJM Load from 2002-2018')


# 

# In[12]:


# PJM Power Consumption Box Plot by Hour of Day
fig, ax = plt.subplots(figsize=(15,5))
sns.boxplot(df.hour, df.PJME_MW)
ax.set_title('PJM Power Consumption by Hour')
ax.set_ylim(0,65000)

# PJM Power Consumption Box Plot by Month of Year
fig, ax = plt.subplots(figsize=(15,5))
sns.boxplot(df.month, df.PJME_MW)
ax.set_title('PJM Power Consumption by Month')
ax.set_ylim(0,65000)


# ## Forecasting using XGBoost

# In[17]:


# Split PJME Energy Conumption Dataset to Train and Test 
split_date = '2015-01-01'
df_train = df.loc[df.Datetime <= split_date].copy()
df_test = df.loc[df.Datetime > split_date].copy()


# In[18]:


# Plot PJME Energy Conumption Train and Test Datasets
df_train[['PJME_MW','Datetime']].plot(x='Datetime',
                                     y='PJME_MW',
                                     kind='scatter',
                                     figsize=(14,4),
                                     title='PJM Load from 2002-2018')
df_test[['PJME_MW','Datetime']].plot(x='Datetime',
                                     y='PJME_MW',
                                     kind='scatter',
                                     figsize=(14,4),
                                     title='PJM Load from 2002-2018')


# In[22]:


# Function to create features 
def create_features(df, label=None):
    """
    Creates time series features from datetime index
    Feature set 1: date, hour, day of the week, month, quarter, year, day of the year, week of the year
    Feature Set 2: Energy Consumption Lag Features to convert forecasting Labelled, suervised ML problem: Lag of 2, 4, 8, 12 and 24 Hours
    Feature Set 3: Rolling Windows - Fixed size windows (4, 8, 12 and 24 Hour windows) over which mean, standard deviation, Maximum and Minimum. 
    """
    df['date'] = df.index
    df['hour'] = df['Datetime'].dt.hour
    df['dayofweek'] = df['Datetime'].dt.dayofweek
    df['month'] = df['Datetime'].dt.month
    df['quarter'] = df['Datetime'].dt.quarter
    df['year'] = df['Datetime'].dt.year
    df['dayofyear'] = df['Datetime'].dt.dayofyear
    df['dayofmonth'] = df['Datetime'].dt.day
    df['weekofyear'] = df['Datetime'].dt.weekofyear
    df['pjme_2_hrs_lag'] = df['PJME_MW'].shift(2)
    df['pjme_4_hrs_lag'] = df['PJME_MW'].shift(4)
    df['pjme_8_hrs_lag'] = df['PJME_MW'].shift(8)
    df['pjme_12_hrs_lag'] = df['PJME_MW'].shift(12)
    df['pjme_24_hrs_lag'] = df['PJME_MW'].shift(24)
    df['pjme_4_hrs_mean'] = df['PJME_MW'].rolling(window = 4).mean()
    df['pjme_8_hrs_mean'] = df['PJME_MW'].rolling(window = 8).mean()
    df['pjme_12_hrs_mean'] = df['PJME_MW'].rolling(window = 12).mean()
    df['pjme_24_hrs_mean'] = df['PJME_MW'].rolling(window = 24).mean()
    df['pjme_4_hrs_std'] = df['PJME_MW'].rolling(window = 4).std()
    df['pjme_8_hrs_std'] = df['PJME_MW'].rolling(window = 8).std()
    df['pjme_12_hrs_std'] = df['PJME_MW'].rolling(window = 12).std()
    df['pjme_24_hrs_std'] = df['PJME_MW'].rolling(window = 24).std()
    df['pjme_4_hrs_max'] = df['PJME_MW'].rolling(window = 4).max()
    df['pjme_8_hrs_max'] = df['PJME_MW'].rolling(window = 8).max()
    df['pjme_12_hrs_max'] = df['PJME_MW'].rolling(window = 12).max()
    df['pjme_24_hrs_max'] = df['PJME_MW'].rolling(window = 24).max()
    df['pjme_4_hrs_min'] = df['PJME_MW'].rolling(window = 4).min()
    df['pjme_8_hrs_min'] = df['PJME_MW'].rolling(window = 8).min()
    df['pjme_12_hrs_min'] = df['PJME_MW'].rolling(window = 12).min()
    df['pjme_24_hrs_min'] = df['PJME_MW'].rolling(window = 24).min()
    
    X = df[['hour','dayofweek','quarter','month','year',
           'dayofyear','dayofmonth','weekofyear' ,'pjme_2_hrs_lag' , 'pjme_4_hrs_lag' , 'pjme_8_hrs_lag' , 'pjme_12_hrs_lag' , 'pjme_24_hrs_lag' , 'pjme_4_hrs_mean',
          "pjme_8_hrs_mean", "pjme_12_hrs_mean" ,"pjme_24_hrs_mean" ,"pjme_4_hrs_std" ,"pjme_8_hrs_std" ,"pjme_12_hrs_std" ,"pjme_24_hrs_std",
           "pjme_4_hrs_max", "pjme_8_hrs_max","pjme_12_hrs_max" ,"pjme_24_hrs_max" ,"pjme_4_hrs_min","pjme_8_hrs_min","pjme_12_hrs_min" ,"pjme_24_hrs_min"]]
    if label:
        y = df[label]
        return X, y
    return X


# In[32]:


# Create feature set for train and test dataset
X_train, Y_train = create_features(df_train, label='PJME_MW')
X_test, Y_test = create_features(df_test, label='PJME_MW')


# In[33]:


#Fit XGB model on the Energy Consumption Dataset
E_model = xgb.XGBRegressor(n_estimators=1000)
E_model.fit(X_train, Y_train,
        eval_set=[(X_train, Y_train), (X_test, Y_test)],
        early_stopping_rounds=50,
       verbose=False)


# In[34]:


#Plot Top 15 features by Significance for forecasting
_ = plot_importance(E_model, height=0.9 ,max_num_features = 15)


# In[35]:


# Predict Energy consumption for 2015-2018 using trained XGB model and set of features used earlier
df_test['MW_Prediction'] = E_model.predict(X_test)
df_all = pd.concat([df_test, df_train], sort=False)


# In[36]:


# Plot original and Model-Predicted energy consumption Output for PJME
_ = df_all[['PJME_MW','MW_Prediction']].plot(figsize=(15, 5))


# In[37]:


# Calculate MSE, MAE and MAPE for Predicted output to quantify model error
mse = mean_squared_error(y_true=df_test['PJME_MW'],
                   y_pred=df_test['MW_Prediction'])
mae = mean_absolute_error(y_true=df_test['PJME_MW'],
                   y_pred=df_test['MW_Prediction'])
def mean_absolute_percentage_error(y_true, y_pred): 
    """Calculates MAPE given y_true and y_pred"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
mape = mean_absolute_percentage_error(y_true=df_test['PJME_MW'],
                   y_pred=df_test['MW_Prediction'])
print(mse,mae,mape)

