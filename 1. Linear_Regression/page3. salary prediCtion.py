import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# step 1: read the data from data source
df = pd.read_csv('./Salary_Data.csv')

# print(df.describe())
# print(df.columns)
# print(df.info())
# print(df.isna().sum())

# step 2: clean the data / prepare the data for ML operation
# - 2.1: remove all missing values (NaN)
# - 2.2: add or remove required columns
# - 2.3: adjust the required data types (data types conversion)
# - 2.4: conversion of textual to numeric values
# - 2.5: scale the values

# step 3: create the model (formula)
x = df.drop('Salary', axis=1)
y = df['Salary']

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x, y)

# step 4: perform the operation (predict the future value)
salaries = model.predict([[14], [15]])
print(f"salary for 14 yrs of experience person = {salaries[0]}")
print(f"salary for 15 yrs of experience person = {salaries[1]}")

# step 5: model evaluation

# step 6: data visualization of result

plt.scatter(x, y)

# best fit regression line
predicted_values = model.predict(x)
plt.plot(x, predicted_values, color="blue")
plt.scatter(x, predicted_values, color="orange")

# mean line
mean_values = np.ones(len(df['Salary']))
mean_values = mean_values * df['Salary'].mean()
plt.plot(x, mean_values, color="green")

plt.xlabel('Years of Experience')
plt.ylabel('Salaries')
plt.title('Experience vs Salary')
plt.show()