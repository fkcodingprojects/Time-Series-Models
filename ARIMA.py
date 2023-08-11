#!/usr/bin/env python
# coding: utf-8

# ## Import Libraries and Load Data

# In[54]:


import pandas as pd
import matplotlib.pyplot as plt
import itertools
import statsmodels.api as sm
import warnings

# Suppress the warning messages
warnings.filterwarnings("ignore")

# Read the CSV file into a pandas DataFrame
file_path = "london_weather[1].csv"
df = pd.read_csv(file_path)

# Prepare the data
df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
df.set_index('date', inplace=True)
mean_temp_data = df['mean_temp'].dropna()


# ## ARIMA Model Building and Selection 

# In[58]:


import warnings
from pmdarima import auto_arima
import itertools
import statsmodels.api as sm

# Suppress the warning messages
warnings.filterwarnings("ignore")

# Set the typical ranges for p, d, q
p = d = q = range(0, 3)

# Take all possible combinations for p, d, and q
pdq = list(itertools.product(p, d, q))

# Using Grid Search, find the optimal ARIMA model that yields the best AIC
best_aic_grid = float("inf")
best_arima_model_grid = None

for param in pdq:
    try:
        model = sm.tsa.ARIMA(mean_temp_data.loc[:'2019'], order=param)
        results = model.fit()
        if results.aic < best_aic_grid:
            best_aic_grid = results.aic
            best_arima_model_grid = results
    except:
        continue

print("Best ARIMA Model from Grid Search (p, d, q):", best_arima_model_grid.params)
print("AIC for Best ARIMA Model from Grid Search:", best_aic_grid)


# In[59]:


import warnings
from pmdarima import auto_arima

# Suppress the warning messages
warnings.filterwarnings("ignore")

# Using Auto-ARIMA, find the optimal ARIMA model
auto_arima_model = auto_arima(mean_temp_data.loc[:'2019'], seasonal=False, suppress_warnings=True)
best_arima_order = auto_arima_model.order

print("Best ARIMA Model Order from Auto-ARIMA:", best_arima_order)


# In[14]:


# Choose the best model based on AIC values
if best_aic_grid < best_aic_auto:
    best_model = best_arima_model_grid
    best_source = "Grid Search"
else:
    best_model = best_arima_model_auto
    best_source = "Auto ARIMA"

print("Best ARIMA Model Source:", best_source)
print("Best ARIMA Model Summary:")
print(best_model.summary())


# ## Forecast Next Year 

# In[60]:


forecast_horizon = 365
forecast_dates = pd.date_range(start=mean_temp_data.index[-1], periods=forecast_horizon, freq='D')

# Forecast using the best ARIMA model
forecast = best_model.get_forecast(steps=forecast_horizon)

# Extract forecasted mean and confidence intervals
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()

# Convert forecast values to integers
forecast_mean_int = forecast_mean.astype(int)
forecast_ci_int = forecast_ci.astype(int)

# Plot the forecast for the next five years
plt.figure(figsize=(14, 7))
plt.plot(mean_temp_data.index, mean_temp_data, label='Observed')
plt.plot(forecast_dates, forecast_mean_int, label='Forecast', alpha=0.7)
plt.fill_between(forecast_dates, forecast_ci_int.iloc[:, 0], forecast_ci_int.iloc[:, 1], color='k', alpha=0.2)
plt.xlabel("Date")
plt.ylabel('Mean Temperature (°C)')
plt.legend()
plt.show()


# In[ ]:




