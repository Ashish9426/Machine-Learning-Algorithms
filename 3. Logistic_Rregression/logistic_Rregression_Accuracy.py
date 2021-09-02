import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# step 1: read the data from data source
df = pd.read_csv('./heart_disease.csv')

# step 2: clean the data / prepare the data for ML operation
# - 2.1: remove all missing values (NaN)
# - 2.2: add or remove required columns
df = df.drop(['trestbps', 'chol', 'fbs', 'restecg'], axis=1)
print(df.columns)
# print(df.info())

# - 2.3: adjust the required data types (data types conversion)
# - 2.4: conversion of textual to numeric values
# - 2.5: scale the values

# step 3: create the model (formula)
from sklearn.linear_model import LogisticRegressionCV

x = df.drop('target', axis=1)
y = df['target']

# split the data into train and test sets
from sklearn.model_selection import train_test_split

# split the data into 80% of train and 20% of test data
# 345345: 86
# 123456: 90
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=123456)

model = LogisticRegressionCV(max_iter=1000)

# train the model using train data
model.fit(x_train, y_train)

# step 4: perform the operation (predict the future value)
y_predictions = model.predict(x_test)
# print(y_test)
# print(y_predictions)
# print(len(y_predictions))

# step 5: model evaluation
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report

cm = confusion_matrix(y_test, y_predictions)
print(cm)

tn = cm[0][0]
fp = cm[0][1]
fn = cm[1][0]
tp = cm[1][1]

accuracy = (tn + tp) / (tn + tp + fn + fp)
print(f"1.1 accuracy: {accuracy * 100}%")
print(f"1.2 accuracy: {accuracy_score(y_test, y_predictions) * 100}%")

print(f"2. f1 score = {f1_score(y_test, y_predictions)}")

print(classification_report(y_test, y_predictions))

# step 6: data visualization of result
# plt.scatter(x['age'], x['oldpeak'], color="red")
# plt.show()
