import pandas as pd
import numpy as np
import os
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import matplotlib.pyplot as plt
import seaborn as sns



data = pd.read_csv("Heart.csv")

X = data.iloc[:, :-1]
y = data.iloc[:, -1]




X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=14
)




scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



os.makedirs("results", exist_ok=True)


models = {
    "Model_1": LogisticRegression(max_iter=50),
    "Model_2": LogisticRegression(max_iter=100),
    "Model_3": LogisticRegression(max_iter=200, C=0.5),
    "Model_4": LogisticRegression(max_iter=200, C=2),
    "Model_5": LogisticRegression(max_iter=300, solver='liblinear')
}


results = []



mlflow.set_experiment("Logistic_Regression_Comparison")



for name, model in models.items():

    with mlflow.start_run(run_name=name):

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

        cm = confusion_matrix(y_test, y_pred)

        print("\n=================================")
        print(name)
        print("Accuracy:", acc)
        print("\nConfusion Matrix:\n", cm)
        print("\nClassification Report:\n", classification_report(y_test, y_pred))


        results.append([name, acc])


        # Log parameters
        mlflow.log_param("max_iter", model.max_iter)

        if hasattr(model, "C"):
            mlflow.log_param("C", model.C)


        # Log metric
        mlflow.log_metric("accuracy", acc)


        # Confusion Matrix Plot
        plt.figure()

        sns.heatmap(cm, annot=True, fmt='d')

        plt.title(name + " Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        fig_path = f"results/{name}_confusion_matrix.png"

        plt.savefig(fig_path)

        plt.close()


        mlflow.log_artifact(fig_path)


        # Log Model
        mlflow.sklearn.log_model(model, name)




results_df = pd.DataFrame(results, columns=["Model", "Accuracy"])

print("\n=================================")
print("Model Comparison")
print(results_df)




plt.figure()

plt.bar(results_df["Model"], results_df["Accuracy"])

plt.title("Model Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy")

acc_plot = "results/accuracy_comparison.png"

plt.savefig(acc_plot)

plt.show()




best_model = results_df.loc[results_df["Accuracy"].idxmax()]

print("\n=================================")
print("Best Model")
print("==========")
print(best_model)