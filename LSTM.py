#!/usr/bin/env python
# coding: utf-8

# ## Import Libaries and Load Data 

# In[1]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
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


# ## Normalize Data and Split into Train-Test Sets 

# In[2]:


# Normalize the data
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(mean_temp_data.values.reshape(-1, 1))

# Split data into training and testing sets
train_size = int(0.8 * len(normalized_data))
train_data, test_data = normalized_data[:train_size], normalized_data[train_size:]


# ## Create Sequences for LSTM

# In[3]:


# Create sequences for LSTM
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        sequence = data[i:i+seq_length]
        target = data[i+seq_length]
        sequences.append((sequence, target))
    return sequences

seq_length = 10  # You can adjust this sequence length
train_sequences = create_sequences(train_data, seq_length)
test_sequences = create_sequences(test_data, seq_length)

# Convert sequences to numpy arrays
X_train = np.array([seq for seq, _ in train_sequences])
y_train = np.array([target for _, target in train_sequences])
X_test = np.array([seq for seq, _ in test_sequences])
y_test = np.array([target for _, target in test_sequences])


# ## Build and Train the LSTM Model 

# In[4]:


# Build the LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(seq_length, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)


# ## Make Predictions and Plot Results 

# In[6]:


# Make predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Inverse transform the predictions to the original scale
train_predictions = scaler.inverse_transform(train_predictions)
test_predictions = scaler.inverse_transform(test_predictions)

test_predictions


# In[8]:


plt.figure(figsize=(12, 6))
plt.plot(mean_temp_data.index[seq_length:len(train_predictions)+seq_length], train_predictions, label='Train Predictions', linewidth=1)
plt.plot(mean_temp_data.index[len(train_predictions)+seq_length:len(train_predictions)+len(test_predictions)+seq_length], test_predictions, label='Test Predictions', linewidth=1)
plt.plot(mean_temp_data.index, mean_temp_data.values, label='Actual Data', color='black', alpha=0.3, linewidth=1)
plt.legend()
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('LSTM Time Series Forecasting')
plt.show()

