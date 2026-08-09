from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# 1. Charger les données (150 fleurs, 4 mesures, 3 espèces)
X, y = load_iris(return_X_y=True)

# 2. Séparer entraînement / test (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Entraîner le modèle
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Évaluer sur les données jamais vues
acc = accuracy_score(y_test, model.predict(X_test))
print(f"Accuracy sur le test : {acc:.3f}")

# 5. Sauvegarder le modèle entraîné
joblib.dump(model, "model.pkl")
print("Modèle sauvegardé dans model.pkl")
