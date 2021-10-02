#!/usr/bin/env python
# coding: utf-8

# # Project - Hourly Energy Consumption Analysis and Forecasting
# PJM Interconnection LLC (PJM) is a regional transmission organization (RTO) in the United States. The hourly energy consumption data is available on PJM's website. The data set is also available part of Kaggle competition on Hourly Energy Consumption. This project first analyzes the energy consumption data of PJM East region (2001-2018) and than applies XGBoost forecasting algorithm.

# ## Business Understanding
# Following queries are addressed through this work. 
# ### Energy Analytics
# #### Question 1: What is the trend of energy consumption at PJM East over the years from 2002 - 2018?
# #### Question 2: What is the typical trend of energy consumption at PJM East over months in a year?
# #### Question 3: What is the variation in energy consumption at PJM East through out a day?
# ### Energy Forecasting
# #### Question 4: What are the significant features influencing the forecasting performance?
# #### Question 5: How acurately XGBoost can forecast the energy consumption ?

# In[4]:


# Import Python Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from xgboost import plot_importance
plt.style.use('fivethirtyeight')


# In[5]:


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.utils import check_array


# ## Data Understanding

# In[6]:


# Read in PJM Hourly Power Consumption data file to dataframe df
pjme_energy: object = pd.read_csv('PJME_hourly.csv')


# ## Assess and Explore Data
# The dataset has no null values, and is preprocessed already. The Leap year show higher value logs compared to Other years since there is additional day. 2018 shows lower value since the data is unavailable beyond July

# In[7]:


# Check number of Columns and Rows
num_rows = pjme_energy.shape[0]  # Provide the number of rows in the dataset
num_cols = pjme_energy.shape[1]  # Provide the number of columns in the dataset

print(num_rows, num_cols)


# In[8]:


# Check the Data
print(pjme_energy.tail())


# In[9]:


# Converting Datetime column to datetime %Y-%m-%d %H:%M:%S format
pjme_energy['Datetime'] = pd.to_datetime(
    pjme_energy.Datetime, format='%Y-%m-%d %H:%M:%S')


# In[10]:


# Checking Null values
pjme_energy.isna().sum()


# In[11]:


# Check Total Value Count for Each Year
Yearly_Value_Counts = pjme_energy['Datetime'].dt.year.value_counts(sort=False)
print(Yearly_Value_Counts)


# In[12]:


# Statistics Describing PJME Power Consumption
pjme_statistics = pjme_energy['PJME_MW'].describe().T
print(pjme_statistics)


# In[13]:


# Adding Hour of Day and Month of Year Columns to the Dataframe
pjme_energy['hour'] = pjme_energy.Datetime.dt.hour
pjme_energy['month'] = pjme_energy.Datetime.dt.month


# In[14]:


# Calculate Year-wise Energy Consumption Mean and Std deviation
pjme_energy.groupby(pd.Grouper(key='Datetime', freq='Y')
                    ).agg({'PJME_MW': ['mean', 'std']})


# In[15]:


# Calculate Month-wise Energy Consumption Mean and Std deviation
pjme_energy.groupby(['month']).agg({'PJME_MW': ['mean', 'std']})


# In[16]:


# Calculate Hour-wise Energy Consumption Mean and Std deviation
pjme_energy.groupby(['hour']).agg({'PJME_MW': ['mean', 'std']})


# In[17]:


# Histogram of Power Consumption for PJME
pjme_energy['PJME_MW'].plot.hist(
    figsize=(15, 5), bins=200, title='Distribution of PJME Load')


# ## Visualize Data

# ### Question 1: What is the trend of energy consumption at PJM East over the years from 2002 - 2018?
# Answer: PJM Energy consumption Load does not show increasing or decreasing trend from 2002-2018. The demand seems to be stagnant overall with Yearly average energy consumption in 30000-330000 MW range.

# In[18]:


# PJME Power Consumption Trend Chart
pjme_energy[['PJME_MW', 'Datetime']].plot(x='Datetime',
                                          y='PJME_MW',
                                          kind='scatter',
                                          figsize=(14, 4),
                                          title='PJM Load from 2002-2018')


# ### Question 2: What is the typical trend of energy consumption at PJM East over months in a year?
# Answer: The energy consumption trend shows demand peaking in Months Jun-Sep and again a smaller peak in Dec-Feb.
# ### Question 3: What is the variation in energy consumption at PJM East through out a day?
# Answer: The energy consumption is higher during afternoon-evening hours from 2PM - 7PM after which it starts to decrease. Expectedly, the energy consumption is the lowest during night hours 1AM-6AM

# In[19]:


# PJM Power Consumption Box Plot by Hour of Day
fig, ax = plt.subplots(figsize=(15, 5))
sns.boxplot(x=pjme_energy.hour, y=pjme_energy.PJME_MW)
ax.set_title('PJM Power Consumption by Hour')
ax.set_ylim(0, 65000)

# PJM Power Consumption Box Plot by Month of Year
fig, ax = plt.subplots(figsize=(15, 5))
sns.boxplot(x=pjme_energy.month, y=pjme_energy.PJME_MW)
ax.set_title('PJM Power Consumption by Month')
ax.set_ylim(0, 65000)


# 

# ## Forecasting using XGBoost

# In[20]:


# Split PJME Energy Conumption Dataset to Train and Test
SPLIT_DATE = '2015-01-01'
pjme_energy_train = pjme_energy.loc[pjme_energy.Datetime <= SPLIT_DATE].copy()
pjme_energy_test = pjme_energy.loc[pjme_energy.Datetime > SPLIT_DATE].copy()


# In[21]:


# Plot PJME Energy Conumption Train and Test Datasets
pjme_energy_train[['PJME_MW', 'Datetime']].plot(x='Datetime',
                                                y='PJME_MW',
                                                kind='scatter',
                                                figsize=(14, 4),
                                                title='PJM Load from 2002-2018')
pjme_energy_test[['PJME_MW', 'Datetime']].plot(x='Datetime',
                                               y='PJME_MW',
                                               kind='scatter',
                                               figsize=(14, 4),
                                               title='PJM Load from 2002-2018')


# ## Data Modeling

# In[22]:


# Function to create features
def create_features(energy_data, label=None):
    """
    Creates time series features from datetime index
    - Feature set 1: date, hour, day of the week, month, quarter, year, day of the year,
      week of the year
    - Feature Set 2: Energy Consumption Lag Features to convert forecasting Labelled,
      supervised ML problem: Lag of 2, 4, 8, 12 and 24 Hours
    - Feature Set 3: Rolling Windows - Fixed size windows (4, 8, 12 and 24 Hour windows)
      over which mean, standard deviation, Maximum and Minimum.
    """
    energy_data['date'] = energy_data.index
    energy_data['hour'] = energy_data['Datetime'].dt.hour
    energy_data['dayofweek'] = energy_data['Datetime'].dt.dayofweek
    energy_data['month'] = energy_data['Datetime'].dt.month
    energy_data['quarter'] = energy_data['Datetime'].dt.quarter
    energy_data['year'] = energy_data['Datetime'].dt.year
    energy_data['dayofyear'] = energy_data['Datetime'].dt.dayofyear
    energy_data['dayofmonth'] = energy_data['Datetime'].dt.day
    energy_data['weekofyear'] = energy_data['Datetime'].dt.weekofyear
    energy_data['pjme_2_hrs_lag'] = energy_data['PJME_MW'].shift(2)
    energy_data['pjme_4_hrs_lag'] = energy_data['PJME_MW'].shift(4)
    energy_data['pjme_8_hrs_lag'] = energy_data['PJME_MW'].shift(8)
    energy_data['pjme_12_hrs_lag'] = energy_data['PJME_MW'].shift(12)
    energy_data['pjme_24_hrs_lag'] = energy_data['PJME_MW'].shift(24)
    energy_data['pjme_4_hrs_mean'] = energy_data['PJME_MW'].rolling(
        window=4).mean()
    energy_data['pjme_8_hrs_mean'] = energy_data['PJME_MW'].rolling(
        window=8).mean()
    energy_data['pjme_12_hrs_mean'] = energy_data['PJME_MW'].rolling(
        window=12).mean()
    energy_data['pjme_24_hrs_mean'] = energy_data['PJME_MW'].rolling(
        window=24).mean()
    energy_data['pjme_4_hrs_std'] = energy_data['PJME_MW'].rolling(
        window=4).std()
    energy_data['pjme_8_hrs_std'] = energy_data['PJME_MW'].rolling(
        window=8).std()
    energy_data['pjme_12_hrs_std'] = energy_data['PJME_MW'].rolling(
        window=12).std()
    energy_data['pjme_24_hrs_std'] = energy_data['PJME_MW'].rolling(
        window=24).std()
    energy_data['pjme_4_hrs_max'] = energy_data['PJME_MW'].rolling(
        window=4).max()
    energy_data['pjme_8_hrs_max'] = energy_data['PJME_MW'].rolling(
        window=8).max()
    energy_data['pjme_12_hrs_max'] = energy_data['PJME_MW'].rolling(
        window=12).max()
    energy_data['pjme_24_hrs_max'] = energy_data['PJME_MW'].rolling(
        window=24).max()
    energy_data['pjme_4_hrs_min'] = energy_data['PJME_MW'].rolling(
        window=4).min()
    energy_data['pjme_8_hrs_min'] = energy_data['PJME_MW'].rolling(
        window=8).min()
    energy_data['pjme_12_hrs_min'] = energy_data['PJME_MW'].rolling(
        window=12).min()
    energy_data['pjme_24_hrs_min'] = energy_data['PJME_MW'].rolling(
        window=24).min()

    features = energy_data[['hour', 'dayofweek', 'quarter', 'month', 'year',
                            'dayofyear', 'dayofmonth', 'weekofyear', 'pjme_2_hrs_lag', 'pjme_4_hrs_lag',
                            'pjme_8_hrs_lag', 'pjme_12_hrs_lag', 'pjme_24_hrs_lag', 'pjme_4_hrs_mean',
                            "pjme_8_hrs_mean", "pjme_12_hrs_mean", "pjme_24_hrs_mean", "pjme_4_hrs_std",
                            "pjme_8_hrs_std", "pjme_12_hrs_std", "pjme_24_hrs_std",
                            "pjme_4_hrs_max", "pjme_8_hrs_max", "pjme_12_hrs_max", "pjme_24_hrs_max",
                            "pjme_4_hrs_min", "pjme_8_hrs_min", "pjme_12_hrs_min", "pjme_24_hrs_min"]]
    if label:
        label = energy_data[label]
        return features, label
    return features


# In[23]:


# Create feature set for train and test dataset
pjme_features_train, pjme_label_train = create_features(
    pjme_energy_train, label='PJME_MW')
pjme_features_test, pjme_label_test = create_features(
    pjme_energy_test, label='PJME_MW')


# In[24]:


# Fit XGB model on the Energy Consumption Dataset
energy_forecasting_model = xgb.XGBRegressor(n_estimators=1000)
energy_forecasting_model.fit(pjme_features_train, pjme_label_train,
                             eval_set=[(pjme_features_train, pjme_label_train),
                                       (pjme_features_test, pjme_label_test)],
                             early_stopping_rounds=50,
                             verbose=False)


# ## Evaluate the Results

# ### Question 4: What are the significant features influencing the forecasting performance?
# Answer: Feature importance plot below shows significant features for forecasting. Lag features of (2 Hour Lag, 24 and 4 Hour lag) are most important alongwith Hour (or time) of day and 4 Hour rolling window statistical variables of mean, standard deviation, Max and min values in that order. 

# In[25]:


# Plot Top 15 features by Significance for forecasting
_ = plot_importance(energy_forecasting_model, height=0.9, max_num_features=15)


# In[26]:


# Predict Energy consumption for 2015-2018 using trained XGB model and set of features used earlier
pjme_energy_test['MW_Prediction'] = energy_forecasting_model.predict(
    pjme_features_test)
df_all = pd.concat([pjme_energy_test, pjme_energy_train], sort=False)


# ### Question 5: How acurately XGBoost can forecast the energy consumption?
# Answer: The results show XGBoost accurately predicting energy consumption with 0.5% MAPE.

# In[27]:


# Plot original and Model-Predicted energy consumption Output for PJME
_ = df_all[['PJME_MW', 'MW_Prediction']].plot(figsize=(15, 5))


# In[28]:


# Calculate MSE, MAE and MAPE for Predicted output to quantify model error
mse_xgb = mean_squared_error(y_true=pjme_energy_test['PJME_MW'],
                         y_pred=pjme_energy_test['MW_Prediction'],squared = False)
mae_xgb = mean_absolute_error(y_true=pjme_energy_test['PJME_MW'],
                          y_pred=pjme_energy_test['MW_Prediction'])


def mean_absolute_percentage_error(y_true, y_pred):
    """Calculates MAPE given y_true and y_pred"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


mape_xgb = mean_absolute_percentage_error(y_true=pjme_energy_test['PJME_MW'],
                                      y_pred=pjme_energy_test['MW_Prediction'])
print( 'RMSE using XGB:', mse_xgb,
      '\n '   'MAE using XGB:', mae_xgb, '  '   'MAPE using XGB:', mape_xgb)


# ## Prediction using XGB with Gridsearch

# In[29]:


# Define Pipeline and Parammeter grid
pipeline = Pipeline([
    ('model', energy_forecasting_model)
])

param_grid = {
    'model__max_depth': [2, 3, 6, 8],
    'model__n_estimators': [10, 100, 1000, 2000],
    'model__learning_rate': [0.02, 0.05, 0.1, 0.3],
    'model__min_child_weight': [0.5, 1, 2]
}

grid = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)


# In[ ]:


# Fit the model
grid.fit(pjme_features_train, pjme_label_train)


# In[ ]:


# Print the Best parameters for the model identified using Gridsearch
print(f"Best parameters: {grid.best_params_}")


# In[ ]:


# Predict Energy consumption for 2015-2018 using trained best XGB model found using gridsearch and set of features used earlier
pjme_energy_test['MW_Prediction_grid'] = grid.predict(
    pjme_features_test)
df_all_grid = pd.concat([pjme_energy_test, pjme_energy_train], sort=False)


# In[ ]:


# Calculate MSE, MAE and MAPE for Predicted output to quantify model error
mse_xgb_grid = mean_squared_error(y_true=pjme_energy_test['PJME_MW'],
                         y_pred=pjme_energy_test['MW_Prediction_grid'],squared = False)
mse_xgb_grid = mean_absolute_error(y_true=pjme_energy_test['PJME_MW'],
                          y_pred=pjme_energy_test['MW_Prediction_grid'])
mape_xgb_grid = mean_absolute_percentage_error(y_true=pjme_energy_test['PJME_MW'],
                                      y_pred=pjme_energy_test['MW_Prediction_grid'])
print( 'RMSE using XGB_grid:', mse_xgb_grid,
      '\n '   'MAE using XGB_grid:', mse_xgb_grid, '  '   'MAPE using XGB_grid:', mape_xgb_grid)

