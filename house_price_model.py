import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Loading the dataset
df=pd.read_csv('Housing.csv')
print(df.head())

#Inspecting the dataset
print(df.shape)
print(df.info())
print(df.describe())

#Handling missing values
print(df.isnull().sum())

#Visualizing the data
plt.hist(df['price'],bins=20)
plt.xlabel('Price')
plt.ylabel('Frequency') 
plt.title('Distribution of House Prices')
plt.show()
  