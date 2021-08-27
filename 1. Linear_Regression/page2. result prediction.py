import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# step 1: read the data from data source
df = pd.read_csv('./simple_regression.csv')

# print(df.info())
# print(df.isna().sum())

# step 2: clean the data / prepare the data for ML operation
# - 2.1: remove all missing values (NaN)
# - 2.2: add or remove required columns
# - 2.3: adjust the required data types (data types conversion)
# - 2.4: conversion of textual to numeric values
# - 2.5: scale the values

# step 3: create the model (formula)

# identify the dependent variable and independent variable(s)
x = df.drop('GPA', axis=1)
y = df['GPA']

# import the linear regression
from sklearn.linear_model import LinearRegression

# instantiate the model
model = LinearRegression()

# train the model
model.fit(x, y)

# step 4: perform the operation (predict the future value)

# GPA for a student having 1800, 2000, 1900, 1990 SAT score
gpa_values = model.predict([[1800], [2000], [1900], [1990]])
print(f"student having 1800 SAT score may get GPA = {gpa_values[0]}")
print(f"student having 2000 SAT score may get GPA = {gpa_values[1]}")
print(f"student having 1900 SAT score may get GPA = {gpa_values[2]}")
print(f"student having 1990 SAT score may get GPA = {gpa_values[3]}")

# step 5: model evaluation

# step 6: data visualization of result

# get predicted GPA value for every SAT score
predicted_values = model.predict(x)

plt.scatter(x, y)

# draw the best fit regression line
plt.plot(x, predicted_values, color="red")

# get the mean line
mean_values = np.ones(len(df['SAT']))
mean_values = mean_values * df['GPA'].mean()

# draw the mean line
plt.plot(x, mean_values, color="green")

plt.xlabel('SAT Scores')
plt.ylabel('GPA')
plt.title('SAT vs GPA')
plt.show()