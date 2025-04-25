from flask import Flask, request, render_template
import numpy as np
import joblib
import os

app = Flask(__name__)

model_dir = os.path.dirname(os.path.abspath(__file__))
model_file_path = os.path.join(model_dir, 'diabetes_model.joblib')

try:
    model = joblib.load(model_file_path)
except FileNotFoundError:
    print("Error: Trained model file not found. Make sure 'diabetes_model.joblib' exists.")
    model = None

@app.route('/', methods=['GET', 'POST'])
def predict():
    if model is None:
        return "Error: Trained model not loaded."

    prediction_text = ""

    if request.method == 'POST':
        pregnancies = request.form['pregnancies']
        glucose = request.form['glucose']
        bloodpressure = request.form['bloodpressure']
        skinthickness = request.form['skinthickness']
        insulin = request.form['insulin']
        bmi = request.form['bmi']
        diabetespedigreefunction = request.form['diabetespedigreefunction']
        age = request.form['age']

        try:
            features = np.array([[float(pregnancies), float(glucose), float(bloodpressure), float(skinthickness), float(insulin), float(bmi), float(diabetespedigreefunction), float(age)]])
            prediction = model.predict(features)

            if prediction[0] == 1:
                prediction_text = "The model predicts: Diabetes"
            else:
                prediction_text = "The model predicts: No Diabetes"
        except ValueError:
            prediction_text = "Please enter valid numerical values for all features."

    return render_template('index.html', prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)