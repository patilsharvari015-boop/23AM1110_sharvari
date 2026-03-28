from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import numpy as np
import pickle

app = FastAPI()

templates = Jinja2Templates(directory="templates")

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


@app.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,
    age: float = Form(...),
    sex: float = Form(...),
    cp: float = Form(...),
    trestbps: float = Form(...),
    chol: float = Form(...),
):

    # model expects 13 features → fill remaining with 0
    data = np.array([[age, sex, cp, trestbps, chol, 0, 0, 0, 0, 0, 0, 0, 0]])

    prediction = model.predict(data)

    if prediction[0] == 1:
        result = "Heart Disease Detected"
    else:
        result = "No Heart Disease"

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "prediction": result}
    )