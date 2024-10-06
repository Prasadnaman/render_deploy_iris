from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load('model.pkl')
FEATURES = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        input_data = pd.DataFrame([data], columns=FEATURES)
        prediction = model.predict(input_data)
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
