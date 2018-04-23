# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 16:00:51 2018

@author: CallejaL
"""

import pandas as pd
import requests
import datetime
import matplotlib.pyplot as plt
import seaborn as sbs
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller

china=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00381/PRSA_data_2010.1.1-2014.12.31.csv',parse_dates=[[1,2,3,4]],date_parser=lambda x: datetime.datetime.strptime(x,'%Y %m %d %H'))

china.dtypes
'''
No: row number
year: year of data in this row
month: month of data in this row
day: day of data in this row
hour: hour of data in this row
pm2.5: PM2.5 concentration (ug/m^3) ~ pollution, ie the depenedent variable
DEWP: Dew Point (â„ƒ)
TEMP: Temperature (â„ƒ)
PRES: Pressure (hPa)
cbwd: Combined wind direction - categorical
Iws: Cumulated wind speed (m/s)
Is: Cumulated hours of snow
Ir: Cumulated hours of rain 

suspected autocorrelation between:
    DEWP, TEMP, PRES
    Ir, PRES
    Is, PRES, TEMP
    
'''

#create a datetime index
china.index=china.year_month_day_hour
china.drop(['No','year_month_day_hour'],inplace=True,axis=1)
china.dtypes
china.columns=['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
nulls=china.loc[china.isnull().any(axis=1)]
nulls.head()

#confirms that only the pm2.5 column contains null values
null_cols=china.columns[china.isnull().any(axis=0)]
print(null_cols)

china['pollution'].fillna(0,inplace=True)
china=china[24:]

serials=china.corr()
print(serials)
#some very strong serial correlation present

plt.plot(china['pollution'],marker='o',linestyle='solid')
plt.show()

plt.close()
sbs.tsplot(china.pollution)
plt.show()

#plot each year above the others using the same x-axis if possible
plt.close()
fig, (ax1,ax2,ax3,ax4,ax5)=plt.subplots(5,1, sharex=False)
ax1.plot(china['2010']['pollution'])
ax2.plot(china['2011']['pollution'])
ax3.plot(china['2012']['pollution'])
ax4.plot(china['2013']['pollution'])
ax5.plot(china['2014']['pollution'])
plt.show()

#basic stationarity analysis on the dependent variable: pollution
plot_acf(china['pollution'],lags=20)
plot_pacf(china['pollution'],lags=20)
#This appears to be a AR(1) process

#this is a stationary process according to adfuller... no unit roots
adfuller(china['pollution'])[1]

#data normalization
china.head()

#explanation of the min-max scalar transform: http://benalexkeen.com/feature-scaling-with-scikit-learn/