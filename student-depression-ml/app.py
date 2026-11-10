from fastapi import FastAPI
from predict import predict

app = FastAPI(
    title="Student Depression Prediction API",
    description="Predicts student depression using a machine learning model.",
    version="1.0.0",
)


@app.get("/")
def home():
    return {"message": "Student Depression Prediction API"}


@app.post("/predict")
def predict_depression(student: dict):
    return predict(student)
