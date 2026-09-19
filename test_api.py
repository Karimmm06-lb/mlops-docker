from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

def test_home():
    # la racine répond bien
    r = client.get("/")
    assert r.status_code == 200

def test_predict_setosa():
    # petites mesures -> setosa attendu
    r = client.post("/predict", json={
        "sepal_length": 5.1, "sepal_width": 3.5,
        "petal_length": 1.4, "petal_width": 0.2
    })
    assert r.status_code == 200
    assert r.json()["prediction"] == "setosa"

def test_predict_virginica():
    # grandes mesures -> virginica attendu
    r = client.post("/predict", json={
        "sepal_length": 6.3, "sepal_width": 3.3,
        "petal_length": 6.0, "petal_width": 2.5
    })
    assert r.status_code == 200
    assert r.json()["prediction"] == "virginica"
