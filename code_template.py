
# Importings
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

data = pd.read_csv('./dataset/iris.csv')
data.dropna()

X, y = data.iloc[], data.loc[['']]

# Model Pre-processing

# Data Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1)

# Model Fitting
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
preds = model.predict(X_test)

# Evaluate
score = mean_absolute_error(preds, y_test)
print(score)
