#!/usr/bin/env python
# coding: utf-8

# ## Import Libaries and Load Data 

# In[1]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense
import matplotlib.pyplot as plt

# Read the CSV file into a pandas DataFrame
file_path = "london_weather[1].csv"
df = pd.read_csv(file_path)

# Prepare the data
df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
df.set_index('date', inplace=True)
mean_temp_data = df['mean_temp'].dropna().values


# ## Normalise Data 

# In[2]:


# Normalize the data
scaler = MinMaxScaler()
mean_temp_data_normalized = scaler.fit_transform(mean_temp_data.reshape(-1, 1))


# ## Train and Testing  

# In[3]:


# Create sequences and targets for training
sequence_length = 30
sequences = []
targets = []
for i in range(len(mean_temp_data_normalized) - sequence_length):
    seq = mean_temp_data_normalized[i:i+sequence_length]
    target = mean_temp_data_normalized[i+sequence_length]
    sequences.append(seq)
    targets.append(target)
sequences = np.array(sequences)
targets = np.array(targets)

# Split data into training and testing sets
train_size = int(0.8 * len(sequences))
train_sequences, test_sequences = sequences[:train_size], sequences[train_size:]
train_targets, test_targets = targets[:train_size], targets[train_size:]


# ## Model Building 

# In[4]:


# Build and train the RNN model
model = Sequential()
model.add(SimpleRNN(50, input_shape=(sequence_length, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(train_sequences, train_targets, epochs=50, batch_size=32, verbose=1)

# Make predictions on test data
test_predictions_normalized = model.predict(test_sequences)
test_predictions = scaler.inverse_transform(test_predictions_normalized)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(mean_temp_data[train_size+sequence_length:], test_predictions))

# Print RMSE
print("RMSE:", rmse)


# ## Plot Result 

# In[6]:


# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(df.index[train_size+sequence_length:train_size+sequence_length+len(test_predictions)], test_predictions, label='Test Predictions')
plt.plot(df.index[train_size+sequence_length:train_size+sequence_length+len(test_predictions)], mean_temp_data[train_size+sequence_length:train_size+sequence_length+len(test_predictions)], label='Actual Data', color='black', alpha=0.3)
plt.legend()
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('RNN Time Series Forecasting')
plt.show()

