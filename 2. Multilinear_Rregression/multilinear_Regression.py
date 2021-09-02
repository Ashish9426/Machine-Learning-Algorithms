import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

##### multiple linear regression #####
# multiple independent variables (RnD, Administration, Marketing, State)
# and one dependent variable (Profit)

# step 1: read the data from data source
df = pd.read_csv('./50_Startups.csv')
# print(df.columns)
# print(df.describe())
# print(df.info())
# print(df.isna().sum())

# step 2: clean the data / prepare the data for ML operation
# - 2.1: remove all missing values (NaN)
# - 2.2: add or remove required columns
# - 2.3: adjust the required data types (data types conversion)

# - 2.4: conversion of textual to numeric values
unique_states = df['State'].unique()
df = df.replace(unique_states, range(1, len(unique_states) + 1))
# print(df.head(10))

# - 2.5: scale the values


# step 3: create the model (formula)
from sklearn.linear_model import LinearRegression

# identify the independent columns
# print(f"covarience between RnD and Profit: {np.cov(df['RnD'], df['Profit'])}")
# print(f"covarience between Administration and Profit: {np.cov(df['Administration'], df['Profit'])}")
# print(f"covarience between Marketing and Profit: {np.cov(df['Marketing'], df['Profit'])}")
# print(f"covarience between State and Profit: {np.cov(df['State'], df['Profit'])}")

print(f"correlation between RnD and Profit: {np.corrcoef(df['RnD'], df['Profit'])}")
print(f"correlation between Administration and Profit: {np.corrcoef(df['Administration'], df['Profit'])}")
print(f"correlation between Marketing and Profit: {np.corrcoef(df['Marketing'], df['Profit'])}")
print(f"correlation between State and Profit: {np.corrcoef(df['State'], df['Profit'])}")

x = df.drop(['Profit', 'State'], axis=1)
y = df['Profit']

# print(x)
# print(y)

model = LinearRegression()
model.fit(x, y)

# step 4: perform the operation (predict the future value)
prediction = model.predict([
    [100000, 120000, 500000],
    [150000, 90000, 100000]
])
print(prediction)

# step 5: model evaluation

# step 6: data visualization of result