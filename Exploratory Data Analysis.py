#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

file_path = "london_weather[1].csv"

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(file_path)

# Dataframe columns
df.columns


# In[5]:


# Assuming you've already loaded the data into the DataFrame df

# Get information about the data types and non-null counts for each column
data_info = df.info()

# Get summary statistics of numerical columns
summary_stats = df.describe(include='all').round()

# Concatenate the information and summary statistics into a single DataFrame
summary_df = pd.concat([data_info, summary_stats], axis=0)

# Print the summary DataFrame
print(summary_df)


# In[9]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming you've already loaded the data into the DataFrame df

# Set Seaborn style
sns.set(style="whitegrid", font_scale=1.2)

# Filter out the 'date' column from the DataFrame
numeric_columns = df.drop(columns=['date']).select_dtypes(include=[int, float]).columns

# Calculate the number of subplots (excluding 'date' column)
num_plots = len(numeric_columns)

# Determine the number of subplot rows and columns
num_rows = (num_plots - 1) // 4 + 1
num_cols = min(num_plots, 4)

# Create a figure with subplots
fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(18, 12))
fig.subplots_adjust(hspace=0.5)  # Adjust the space between subplots

# Histograms with more professional look
for i, column in enumerate(numeric_columns):
    ax = axes[i // num_cols, i % num_cols]
    df[column].hist(ax=ax, bins=20, edgecolor='black', linewidth=1.2, alpha=0.7)
    ax.set_title(column)
    ax.set_xlabel('Values')
    ax.set_ylabel('Frequency')

# Remove any empty subplots if the number of plots is not a multiple of 4
if num_plots % 4 != 0:
    for i in range(num_plots % 4, 4):
        fig.delaxes(axes[-1, i])

plt.show()


# In[10]:


# Calculate the correlation matrix
correlation_matrix = df.corr()

# Set Seaborn style
sns.set(style="white")

# Create a heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap', fontsize=16)

plt.show()


# In[11]:


# Set Seaborn style
sns.set(style="whitegrid", font_scale=1.2)

# Filter out the 'date' column from the DataFrame
numeric_columns = df.drop(columns=['date']).select_dtypes(include=[int, float]).columns

# Calculate the number of subplots (excluding 'date' column)
num_plots = len(numeric_columns)

# Determine the number of subplot rows and columns
num_rows = (num_plots - 1) // 4 + 1
num_cols = min(num_plots, 4)

# Create a figure with subplots
fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(18, 12))
fig.subplots_adjust(hspace=0.5)  # Adjust the space between subplots

# Boxplots with more professional look
for i, column in enumerate(numeric_columns):
    ax = axes[i // num_cols, i % num_cols]
    sns.boxplot(data=df, y=column, ax=ax)
    ax.set_title(column)
    ax.set_ylabel('Values')

# Remove any empty subplots if the number of plots is not a multiple of 4
if num_plots % 4 != 0:
    for i in range(num_plots % 4, 4):
        fig.delaxes(axes[-1, i])

plt.show()


# In[6]:


# Filter out the 'date' column from the DataFrame
numeric_columns = df.drop(columns=['date']).select_dtypes(include=[int, float]).columns

# Determine the number of subplot rows and columns
num_plots = len(numeric_columns)
num_rows = (num_plots - 1) // 2 + 1
num_cols = min(num_plots, 2)

# Create a figure with subplots
fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(12, 8))
fig.subplots_adjust(hspace=0.5)  # Adjust the space between subplots

# Violin plots for each numerical column
for i, column in enumerate(numeric_columns):
    ax = axes[i // num_cols, i % num_cols]
    sns.violinplot(data=df, y=column, ax=ax)
    ax.set_title(column)
    ax.set_ylabel('')

# Remove any empty subplots if the number of plots is not a multiple of 2
if num_plots % 2 != 0:
    fig.delaxes(axes[-1, -1])

plt.show()

