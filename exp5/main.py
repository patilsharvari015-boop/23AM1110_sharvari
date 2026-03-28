from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import joblib
import numpy as np
import pandas as pd

app = FastAPI()

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <body>
            <h2>Loan Prediction</h2>
            <form action="/predict" method="post">
                Income: <input type="number" name="income"><br>
                Loan Amount: <input type="number" name="loan"><br>
                <input type="submit">
            </form>
        </body>
    </html>
    """

@app.post("/predict")
def predict(income: float = Form(...), loan: float = Form(...)):

    # Create full feature vector
    data = pd.DataFrame([[0]*len(columns)], columns=columns)

    # Fill only known features
    if "income" in data.columns:
        data["income"] = income
    if "loan" in data.columns:
        data["loan"] = loan

    data = scaler.transform(data)
    prediction = model.predict(data)

    return {"Prediction": int(prediction[0])}