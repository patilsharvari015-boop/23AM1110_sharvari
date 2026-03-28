

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import matplotlib.pyplot as plt

data= pd.read_csv("Heart.csv")

x= data.iloc[:,:-1]
y= data.iloc[:, -1]

x_test, x_train, y_test, y_train= train_test_split(
    x, y, test_size=0.2, random_state=14
)

model =LogisticRegression(max_iter=20)
model.fit(x_train, y_train)

y_pred=model.predict(x_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report\n:", classification_report(y_test, y_pred))
