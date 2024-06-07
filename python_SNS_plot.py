# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 21:04:22 2024

@author: chenj
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#import sklearn

# This is used to display any figures you create.
#from renderer import display_fig

# Load the data
df = pd.read_csv("C:/Users/chenj/Desktop/Jun/python/usage_data.csv")


# Convert 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Aggregate the data by month
monthly_data = df.groupby(df['Date'].dt.to_period('M')).sum()
#df['year'] = df['Date'].dt.year
#df['Month'] = df['Date'].dt.month
df['YearMonth'] = df['Date'].dt.strftime('%Y-%m')

avg_time_spent = df.groupby('YearMonth')['Average time spent'].mean()
# Plot the data
plt.figure(figsize=(10, 6))
avg_time_spent.plot(kind='bar', color='skyblue')
plt.title('Average Time Spent by YearMonth')
plt.xlabel('YearMonth')
plt.ylabel('Average Time Spent')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


Total_time_spent = df.groupby('YearMonth')['Time spent'].mean()
# Plot the data
plt.figure(figsize=(10, 6))
Total_time_spent.plot(kind='bar', color='skyblue')
plt.title('total Time Spent by YearMonth')
plt.xlabel('YearMonth')
plt.ylabel('Total Time Spent')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



# Group by 'YearMonth' and 'feather', calculate the sum of 'Time spent'
grouped_data = df.groupby(['YearMonth', 'Feature'])['Time spent'].sum().reset_index()

# Pivot the data to have 'YearMonth' as index and 'feather' as columns
pivot_data = grouped_data.pivot(index='YearMonth', columns='Feature', values='Time spent')

# Plot the data
plt.figure(figsize=(12, 6))
pivot_data.plot(kind='bar', stacked=True)
plt.title('Total Time Spent by YearMonth and Feather')
plt.xlabel('YearMonth')
plt.ylabel('Total Time Spent')
plt.xticks(rotation=45)
plt.legend(title='Feature')
plt.tight_layout()
plt.show()




# stop here. 