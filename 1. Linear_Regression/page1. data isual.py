# required mainly for data processing/cleaning
import numpy as np
import pandas as pd

# required mainly for data visualization
import matplotlib.pyplot as plt

# required mainly for performing ML applications
# import sklearn


# read the data
df = pd.read_csv('./simple_regression.csv')
# print(df)
# print(df.columns)
# print(df.describe())
# print(df.info())

# data visualization
x = df['SAT']
y = df['GPA']

plt.scatter(x, y)
plt.xlabel('SAT scores')
plt.ylabel('GPA')
plt.title('SAT vs GPA')
plt.show()