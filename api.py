from flask import Flask, jsonify, request
import joblib
import numpy as np

app = Flask(__name__)

# On charge le modèle entraîné UNE fois au démarrage
model = joblib.load("model.pkl")
classes = ["setosa", "versicolor", "virginica"]

@app.route("/predict")
def predict():
    # 4 mesures passées en paramètres d'URL, avec valeurs par défaut
    # ex: /predict?sl=6.3&sw=3.3&pl=6.0&pw=2.5
    sl = float(request.args.get("sl", 5.1))  # sepal length
    sw = float(request.args.get("sw", 3.5))  # sepal width
    pl = float(request.args.get("pl", 1.4))  # petal length
    pw = float(request.args.get("pw", 0.2))  # petal width

    features = np.array([[sl, sw, pl, pw]])
    pred = int(model.predict(features)[0])
    confidence = float(model.predict_proba(features)[0].max())

    return jsonify({
        "model": "iris-rf-v1",
        "input": {"sepal_length": sl, "sepal_width": sw,
                  "petal_length": pl, "petal_width": pw},
        "prediction": classes[pred],
        "confidence": round(confidence, 3)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6000)
