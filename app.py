from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
from pathlib import Path
import os
import logging
from dotenv import load_dotenv
import together
import io

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load model and preprocessor
model_path = Path('notebooks/new_xgboost_model.pkl')
preprocessor_path = Path('notebooks/new_preprocessor.pkl')
threshold_path = Path('notebooks/new_threshold.pkl')

model = joblib.load(model_path)
preprocessor = joblib.load(preprocessor_path)
threshold = joblib.load(threshold_path)

# Load Together API key from .env
load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
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
        # Build a DataFrame matching the training columns
        df = pd.DataFrame([{
            'age': float(data['age']),
            'CA125': float(data['CA125']),
            'HE4': float(data['HE4']),
            'CA19-9': float(data.get('CA19-9', data.get('CA19_9'))),
            'AFP': float(data['AFP']),
            'GGT': float(data['GGT']),
            'CEA': float(data['CEA']),
            'HGB': float(data['HGB']),
            'ALP': float(data['ALP']),
            'CA72-4': float(data.get('CA72-4', data.get('CA72_4'))),
            'Ca': float(data['Ca']),
            'menopausal_status': data['menopausal_status'],
            'family_history': data['family_history'],
            'smoking_status': data['smoking_status'],
            'alcohol': data['alcohol']
        }])
        # Apply preprocessing to get the right shape
        X_proc = preprocessor.transform(df)
        prob = model.predict_proba(X_proc)[0][1]
        prob = float(prob)
        pct = float(round(prob * 100, 2))
        risk = "High Risk" if prob >= float(threshold) else "Low Risk"
        confidence = float(round((prob if prob >= 0.5 else 1 - prob) * 100, 2))

        return jsonify({
            'success': True,
            'prediction': pct,
            'risk_level': risk,
            'confidence': confidence,
            'recommendation': get_recommendation(pct)
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

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

@app.route('/llm_image_confirm', methods=['POST'])
def llm_image_confirm():
    try:
        if 'histopath_image' not in request.files or 'features' not in request.form:
            return jsonify({'success': False, 'error': 'Missing features or image.'}), 400

        # Save image
        image_file = request.files['histopath_image']
        upload_folder = os.path.join(app.root_path, 'static', 'uploads')
        os.makedirs(upload_folder, exist_ok=True)
        filename = image_file.filename
        path = os.path.join(upload_folder, filename)
        image_file.save(path)
        public_url = f"{request.host_url.rstrip('/')}/static/uploads/{filename}"

        # Parse features
        try:
            features = pd.read_json(io.StringIO(request.form['features']), typ='series').to_dict()
        except Exception:
            features = request.form['features']

        # Build prompt
        prompt_text = (
            "You are an expert pathologist AI. Analyze these clinical features and this histopathology image, "
            "then give a cancer risk prediction and explanation in simple language for a physician.\n\n"
            f"Clinical Features:\n{features}\n"
            f"Image URL: {public_url}"
        )

        # Call Together Complete API
        response = together.Complete.create(
            model="meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
            prompt=prompt_text,
            max_tokens=1000,
            temperature=0.4,
            stop=["</s>"]
        )

        # Extract text
        text = ""
        if isinstance(response, dict):
            # check for X together API shapes
            if "output" in response and "choices" in response["output"]:
                text = response["output"]["choices"][0].get("text", "")
            elif "choices" in response:
                text = response["choices"][0].get("text", "")
            else:
                text = str(response)
        else:
            text = str(response)

        lines = text.strip().split('\n')
        prediction = lines[0] if lines else text
        explanation = "\n".join(lines[1:]) if len(lines) > 1 else ""

        return jsonify({
            'success': True,
            'llm_prediction': prediction,
            'llm_explanation': explanation,
            'image_url': public_url
        })
    except Exception as e:
        logging.error(f"Error in /llm_image_confirm: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)