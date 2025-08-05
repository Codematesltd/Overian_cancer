
from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
from pathlib import Path
import os
from llm_utils import get_llm_context, analyze_histopath_image

app = Flask(__name__)

# Load the model and other files
model_path = Path('notebooks/xgboost_model.pkl')
preprocessor_path = Path('notebooks/preprocessor.pkl')
threshold_path = Path('notebooks/threshold.pkl')

model = joblib.load(model_path)
preprocessor = joblib.load(preprocessor_path)
threshold = joblib.load(threshold_path)

TOGETHER_API_KEY = "your_together_api_key"  # Replace with your actual key
together.api_key = TOGETHER_API_KEY

@app.route('/')
def index():
    return render_template('index.html')  # Directly show index page

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html')
    
    try:
        data = request.get_json()
        
        # Extract features from request
        features = {
            'age': float(data['age']),
            'CA125': float(data['CA125']),
            'HE4': float(data['HE4']),
            'CA19_9': float(data['CA19_9']),
            'AFP': float(data['AFP']),
            'GGT': float(data['GGT']),
            'CEA': float(data['CEA']),
            'HGB': float(data['HGB']),
            'ALP': float(data['ALP']),
            'CA72_4': float(data['CA72_4']),
            'Ca': float(data['Ca']),
            'menopausal_status': data['menopausal_status'],
            'family_history': int(data['family_history']),
            'smoking_status': data['smoking_status'],
            'alcohol': int(data['alcohol'])
        }

        # Create feature array for prediction
        feature_array = np.array([[
            features['age'], features['CA125'], features['HE4'],
            features['CA19_9'], features['AFP'], features['GGT'],
            features['CEA'], features['HGB'], features['ALP'],
            features['CA72_4'], features['Ca'], features['menopausal_status'],
            features['family_history'], features['smoking_status'], features['alcohol']
        ]])

        # Make prediction
        prediction = model.predict_proba(feature_array)[0][1]
        risk_level = "High Risk" if prediction >= threshold else "Low Risk"
        prediction_percentage = round(prediction * 100, 2)
        
        # Calculate AI confidence based on prediction probability
        confidence = round((prediction if prediction >= 0.5 else 1 - prediction) * 100, 2)


        # Get LLM context (dummy)
        context = get_llm_context(prediction_percentage, risk_level, get_recommendation(prediction_percentage))

        return jsonify({
            'success': True,
            'prediction': prediction_percentage,
            'risk_level': risk_level,
            'confidence': confidence,
            'features': features,
            'recommendation': get_recommendation(prediction_percentage),
            'context': context
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

def get_recommendation(prediction_percentage):
    if prediction_percentage < 20:
        return "Continue regular screening. No immediate action required."
    elif prediction_percentage < 50:
        return "Schedule follow-up tests in 3 months."
    else:
        return "Immediate consultation with oncologist recommended."

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        # Handle contact form submission (dummy handler)
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')
        # You can add logic to send email or store message here
        return render_template('contact.html', success=True, name=name)
    return render_template('contact.html')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'histopath_image' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'})
    file = request.files['histopath_image']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'})
    # Save the file to a folder (e.g., 'uploads')
    upload_folder = os.path.join(os.getcwd(), 'uploads')
    os.makedirs(upload_folder, exist_ok=True)
    file.save(os.path.join(upload_folder, file.filename))
    # You can add image analysis logic here if needed
    return jsonify({'success': True})

@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    if 'histopath_image' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'})
    file = request.files['histopath_image']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'})
    # Save the file
    upload_folder = os.path.join(os.getcwd(), 'uploads')
    os.makedirs(upload_folder, exist_ok=True)
    file_path = os.path.join(upload_folder, file.filename)
    file.save(file_path)
    # Dummy LLM image analysis
    summary = analyze_histopath_image(file_path)
    return jsonify({'success': True, 'summary': summary})

if __name__ == '__main__':
    app.run(debug=True)
