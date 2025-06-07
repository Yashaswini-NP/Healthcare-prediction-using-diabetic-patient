from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model and scaler
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from form
        features = [float(x) for x in request.form.values()]
        
        # Scale input
        transformed_features = scaler.transform([features])
        
        # Make prediction
        prediction = model.predict(transformed_features)
        result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
        
        return render_template('index.html', prediction_text=f'Prediction: {result}')
    
    except:
        return render_template('index.html', prediction_text="Invalid Input! Please enter valid numbers.")
    
@app.route('/powerbi')
def powerbi():
    return render_template('powerbi_report.html')


if __name__ == "__main__":
    app.run(debug=True)
