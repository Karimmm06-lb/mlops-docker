from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Iris Prediction API", version="1.0")

# Chargé une fois au démarrage
model = joblib.load("model.pkl")
classes = ["setosa", "versicolor", "virginica"]

# Pydantic valide automatiquement les données entrantes
class IrisInput(BaseModel):
    sepal_length: float = 5.1
    sepal_width: float = 3.5
    petal_length: float = 1.4
    petal_width: float = 0.2

@app.get("/")
def home():
    return {"message": "Iris API is running", "docs": "/docs"}

@app.post("/predict")
def predict(data: IrisInput):
    features = np.array([[data.sepal_length, data.sepal_width,
                          data.petal_length, data.petal_width]])
    pred = int(model.predict(features)[0])
    confidence = float(model.predict_proba(features)[0].max())
    return {
        "model": "iris-rf-v1",
        "input": data.dict(),
        "prediction": classes[pred],
        "confidence": round(confidence, 3),
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=6000)
