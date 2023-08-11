#!/usr/bin/env python
# coding: utf-8

# ## Import Libaries and Load Data 

# In[7]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.optimizers import Adam

# Read the CSV file into a pandas DataFrame
file_path = "london_weather[1].csv"
df = pd.read_csv(file_path)

# Prepare the data
df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
df.set_index('date', inplace=True)
mean_temp_data = df['mean_temp'].dropna()


# ## Training and Testing 

# In[8]:


# Normalize the data
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(mean_temp_data.values.reshape(-1, 1))

# Split data into train and test sets
train_size = int(len(normalized_data) * 0.8)
train_data, test_data = normalized_data[:train_size], normalized_data[train_size:]


# ## Model Building 

# In[9]:


# Create sequences for training
sequence_length = 30
X_train, y_train = [], []
for i in range(len(train_data) - sequence_length):
    X_train.append(train_data[i:i+sequence_length])
    y_train.append(train_data[i+sequence_length])
X_train, y_train = np.array(X_train), np.array(y_train)

# Build the GRU model
model = Sequential([
    GRU(units=50, activation='relu', input_shape=(sequence_length, 1)),
    Dense(units=1)
])
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

# Make predictions on test data
X_test = []
for i in range(len(test_data) - sequence_length):
    X_test.append(test_data[i:i+sequence_length])
X_test = np.array(X_test)
y_pred = model.predict(X_test)

# Inverse transform to get the original scale
y_pred = scaler.inverse_transform(y_pred)
test_data = scaler.inverse_transform(test_data[sequence_length:])

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(test_data, y_pred))
print("Root Mean Squared Error:", rmse)


# ## Plot Results 

# In[10]:


# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(mean_temp_data.index[train_size+sequence_length:], y_pred, label='Predictions')
plt.plot(mean_temp_data.index[train_size+sequence_length:], test_data, label='Actual Data')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('GRU Time Series Forecasting')
plt.legend()
plt.show()


# In[ ]:




