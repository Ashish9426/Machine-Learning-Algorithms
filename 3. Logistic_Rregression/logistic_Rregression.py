import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# step 1: read the data from data source
df = pd.read_csv('./heart_disease.csv')
# print(df.columns)
# print(df.isna().sum())
# print(df.info())

# step 2: clean the data / prepare the data for ML operation
# - 2.1: remove all missing values (NaN)
# - 2.2: add or remove required columns
# - 2.3: adjust the required data types (data types conversion)
# - 2.4: conversion of textual to numeric values
# - 2.5: scale the values

# step 3: create the model (formula)
columns = df.columns
for column in columns:
    result = np.corrcoef(df[column], df['target'])
    print(f"correlation between {column} and target = {result[0][1]}")

x = df.drop(['trestbps', 'chol', 'fbs', 'restecg', 'target'], axis=1)
y = df['target']

from sklearn.linear_model import LogisticRegressionCV
model = LogisticRegressionCV(max_iter=1000)
model.fit(x, y)

predictions = model.predict(x)
print(predictions)

# step 4: perform the operation (predict the future value)

# step 5: model evaluation

# step 6: data visualization of result