#!/usr/bin/env python
# coding: utf-8

# # Inside the Numbers and Charting the Course: Visualizing Nvidia's Strock Performance and Trends

# **Author:** Hasnain Imtiaz 
# **Date:** 25/04/2023
# **Github:** https://github.com/sakib4535

# #### Project Overview:
# The goal of this project is to analyze the performance of Nvidia's stock over time and identify potential patterns or trends that can help inform investment decisions. The analysis will focus on daily returns, autocorrelation, volatility, seasonality, and moving average metrics.
# 
# #### Data Collection:
# Historical data on Nvidia's stock prices will be collected from reliable sources such as Yahoo Finance or Google Finance. The data will cover a period of several years, including both up and down market conditions.
# 
# #### Data Analysis:
# The collected data will be analyzed using various statistical techniques to identify patterns and trends. The following metrics will be used:
# 
#       1. Daily Returns: The daily returns of Nvidia's stock will be calculated and analyzed to identify trends in the stock's   performance over time.
# 
#       2. Autocorrelation: The autocorrelation of Nvidia's stock prices will be calculated to determine whether there is a relationship between the stock's past performance and its future performance.
# 
#       3. Volatility: The volatility of Nvidia's stock will be calculated to identify the degree of variation in the stock's price over time, which can help to determine the stock's risk level.
# 
#       4. Seasonality: The seasonality of Nvidia's stock will be analyzed to determine whether the stock tends to perform better or worse at certain times of the year.
# 
#       5. Moving Average: The moving average of Nvidia's stock prices will be calculated to identify trends in the stock's price movements over time and to help filter out the noise in the stock's daily price movements.
# 
# #### Data Visualization:
# The results of the analysis will be presented through data visualizations such as line charts, bar charts, and scatter plots. These visualizations will help to highlight any patterns or trends that were identified in the data analysis.
# 
# #### Conclusion:
# By analyzing the daily returns, autocorrelation, volatility, seasonality, and moving average metrics of Nvidia's stock over time, this project will provide valuable insights that can inform investment decisions. The results of the analysis can help investors to better understand the stock's performance and make more informed decisions about when to buy or sell Nvidia's stock.

# In[1]:


get_ipython().system('pip install yfinance')


# In[2]:


import yfinance as yf
dir(yf)


# In[3]:


nvda = yf.Ticker("NVDA")
nvda


# In[4]:


dir(nvda)


# In[5]:


nvda.history()


# In[7]:


nvda.history().head(10)


# In[8]:


import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

nvda = yf.Ticker("NVDA")
history = nvda.history(period="5y")

plt.figure(figsize=(11,6))
plt.plot(history['Close'], color='green')
plt.title("NVDA Closing Price")
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()


# In[9]:


import pandas as pd
from tabulate import tabulate

# Assume 'history' is your dataframe containing the Nvidia stock history
summary = history.describe()

# Display the summary table using tabulate
print(tabulate(summary, headers=summary.columns, tablefmt='fancy_grid'))


# In[19]:


history.columns


# In[20]:


history.index


# In[24]:


history.dtypes


# In[30]:


import seaborn as sns
sns.barplot(history.isnull().sum())
plt.title('Missing Data Barplot')
plt.show()


# ### **Analyzing The Trend

# In[10]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# In[11]:


# Converting the index to a numericla array

history['Index'] = range(len(history))
X = sm.add_constant(history['Index'])
y = history['Close'].values

# fitting Linear Regression to the closing price

model = sm.OLS(y, X).fit()

slope = model.params[1]
intercept = model.params[0]


# In[12]:


# Plotting the Trend line

plt.figure(figsize=(11,6))
plt.plot(history['Close'])
plt.plot(history['Close'].index, slope*history['Index'] + intercept, color='red')
plt.title("Nvidia Closing Price with Trend Line")
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()


# ### Auto Correlation Perspective

# In[13]:


# Plot the autocorrelation and partial autocorrelation functions of the closing price
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, (ax1,ax2) = plt.subplots(2,1,figsize=(12,6))
plot_acf(history['Close'], lags=50, alpha=0.05, use_vlines=True, color='green', ax=ax1)
plot_pacf(history['Close'], lags=50,alpha=0.05, use_vlines=True, color='crimson', ax=ax2)
plt.show()


# ### Analyzing daily Returns

# In[14]:


# Calculate Daily Returns
import scipy.stats as stats

returns = history['Close'].pct_change().dropna()
plt.figure(figsize=(8,5))
n, bins, patches = plt.hist(returns, bins=30, density=True, alpha=0.5)
y = stats.norm.pdf(bins, returns.mean(), returns.std())
plt.plot(bins, y, 'r--', linewidth=2)
plt.hist(returns, bins=30)
plt.title("Nvidia Daily Returns")
plt.xlabel("Daily Returns")
plt.ylabel("Frequency")
plt.show()


# In[15]:


#calculate Volatility(Standard Deviation) of the Daily Returns over a rolling on 30

volatility = returns.rolling(window=30).std()
plt.figure(figsize=(12,6))
plt.plot(volatility, color='blue', label='Rolling Volatility')
plt.fill_between(volatility.index, volatility.mean() + volatility.std(), volatility.mean() - volatility.std(), color='red', alpha=0.2, label='±1 Standard Deviation')
plt.title('NVDA 30-Day Rolling Volatility')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.legend()
plt.show()


# In[16]:


#calculate Volatility(Standard Deviation) of the Daily Returns over a rolling on 30

volatility_7d = returns.rolling(window=7).std()
volatility_30d = returns.rolling(window=30).std()
volatility_90d = returns.rolling(window=90).std()

plt.figure(figsize=(12,6))
plt.plot(volatility_7d, color='blue', label='7-Day Rolling Volatility')
plt.plot(volatility_30d, color='green', label='7-Day Rolling Volatility')
plt.plot(volatility_90d, color='red', label='7-Day Rolling Volatility')
plt.fill_between(volatility_7d.index, volatility_7d.mean() + volatility_7d.std(), volatility_7d.mean() - volatility_7d.std(), color='blue', alpha=0.2, label='±1 Standard Deviation')
plt.fill_between(volatility_30d.index, volatility_30d.mean() + volatility_30d.std(), volatility_30d.mean() - volatility_30d.std(), color='green', alpha=0.2, label='±1 Standard Deviation')
plt.fill_between(volatility_90d.index, volatility_90d.mean() + volatility_90d.std(), volatility_90d.mean() - volatility_90d.std(), color='red', alpha=0.2, label='±1 Standard Deviation')
plt.title('NVDA 30-Day Rolling Volatility')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.legend()
plt.show()


# ### Checking Seasonality 

# In[17]:


# Add a column for the month of each data point
history['Month'] = history.index.month

# Calculate the average closing price for each month
monthly_means = history.groupby('Month')['Close'].mean()

# Plot the average closing price for each month
plt.figure(figsize=(12,6))
sns.barplot(x=monthly_means.index, y=monthly_means.values)
plt.title('NVDA Average Closing Price by Month')
plt.xlabel('Month')
plt.ylabel('Average Closing Price')
plt.show()


# ### Moving Average Analysis

# In[18]:


# Add columns for the 10-day and 30-day moving averages
history['MA10'] = history['Close'].rolling(window=10).mean()
history['MA30'] = history['Close'].rolling(window=30).mean()
history['MA90'] = history['Close'].rolling(window=90).mean()

# Plot the closing price with the moving averages
plt.figure(figsize=(12,6))
plt.plot(history['Close'])
plt.plot(history['MA10'])
plt.plot(history['MA30'])
plt.plot(history['MA90'])
plt.title('NVDA Closing Price with Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend(['Closing Price', '10-day MA', '30-day MA', '90-day MA'])
plt.show()


# In[31]:


help(nvda.history())


# In[ ]:




