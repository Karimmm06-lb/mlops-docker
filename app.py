from flask import Flask
import os

app = Flask(__name__)

@app.route('/')
def home():
    name = os.environ.get('APP_NAME', 'MLOps')
    return f"Hello {name}! 🚀"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

