from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/predict')
def predict():
    return jsonify({"model": "v1", "prediction": 42, "confidence": 0.95})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6000)
