# Energy-Consumption-Forecasting-using-XG-Boost
Contains python code to analyze and forecast Energy Consumption using Kaggle dataset of PJM East Operator

## Installations and Quick Start
Install Python Packages: Pandas, Numpy, Maptplotlib, Seaborn, Xgboost, sklearn
Clone the repo https://github.com/nimishsoni/Energy-Consumption-Forecasting-using-XG-Boost.git


## Project Motivation
This work is part of Udacity Data Science Nanodegree project requirement. Since my expertise is in Energy field and work on similar dataset professionally, I chose Kaggle dataset on Hourly energy consumption of an Electricity transfmission operator in US. 

## Files
Energy Consumption Forecasting.ipynb
Contains code for Analyzing and forecasting Hourly Energy consumption using PJM East dataset (PJME_hourly.csv). 
PJME_hourly.csv 
Contains hourly energy consumption data from 2002-2018 July. 

## Key Results:
### Analytics:
PJM Energy consumption Load does not show increasing or decreasing trend from 2002-2018. 
The demand seems to be stagnant overall with Yearly average energy consumption in 30000-330000 MW range.
The minimum and maximum hourly energy demand show large variation based on time of day and season with minimum record being 14544 MW and maximum being 62009 MW.
The energy consumption trend shows demand peaking in Months Jun-Sep and again a smaller peak in Dec-Feb.

### Forecasting
Used create_features function to add following additional feature set
- Timestamp modified features: Hour, Day, Week, month, quarter
- Lag Features
- Rolling Window Features
Divided the dataset in to train (Up to 2014) and test (2015 onwards)
Used XGBoost algorithm for forecasting with MAPE = 0.58%

## Author
Nimish Soni

## Acknowledgement
Hourly Energy consumption Kaggle Competition
https://www.kaggle.com/robikscube/hourly-energy-consumption

Original Code is here
https://www.kaggle.com/robikscube/tutorial-time-series-forecasting-with-xgboost
