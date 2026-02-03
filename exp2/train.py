import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
data = {
    'Hours_Studied': [2, 4, 6, 8, 10, 1, 3, 5, 7, 9],
    'Attendance': [60, 65, 70, 80, 85, 50, 55, 75, 78, 90],
    'Result': [0, 0, 0, 1, 1, 0, 0, 1, 1, 1]
}
df = pd.DataFrame(data)
X = df[['Hours_Studied', 'Attendance']]
y = df['Result']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))