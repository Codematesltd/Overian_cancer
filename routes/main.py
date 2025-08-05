from flask import Blueprint, render_template, request, jsonify
from flask_login import login_required, current_user
import joblib
import numpy as np
from models import Prediction

main_bp = Blueprint('main', __name__)

# Load the model
model = joblib.load('notebooks/xgboost_model.pkl')
scaler = joblib.load('notebooks/preprocessor.pkl')  # If you have a scaler
threshold = joblib.load('notebooks/threshold.pkl')  # Your threshold file

@main_bp.route('/')
def index():
    return render_template('index.html')

@main_bp.route('/about')
def about():
    return render_template('about.html')

@main_bp.route('/contact')
def contact():
    return render_template('contact.html')

@main_bp.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'GET':
        return render_template('predict.html')
    
    try:
        data = request.get_json()
        
        # Extract all features from the request
        features_dict = {
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
            'family_history': data['family_history'],
            'smoking_status': data['smoking_status'],
            'alcohol': data['alcohol']
        }
        
        # Create feature array for model prediction
        features = np.array([[
            features_dict['age'], features_dict['CA125'], features_dict['HE4'],
            features_dict['CA19_9'], features_dict['AFP'], features_dict['GGT'],
            features_dict['CEA'], features_dict['HGB'], features_dict['ALP'],
            features_dict['CA72_4'], features_dict['Ca']
        ]])
        
        # Make prediction
        prediction = model.predict_proba(features)[0][1]  # Get probability of positive class
        
        # Save prediction to Supabase
        Prediction.create(
            user_id=current_user.id,
            **features_dict,
            prediction_result=float(prediction)
        )

        # Format prediction result
        risk_level = "High Risk" if prediction >= threshold else "Low Risk"
        prediction_percentage = round(prediction * 100, 2)
        
        return jsonify({
            'prediction': prediction_percentage,
            'risk_level': risk_level,
            'success': True
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@main_bp.route('/history')
@login_required
def prediction_history():
    response = Prediction.get_user_predictions(current_user.id)
    predictions = response.data
    return render_template('history.html', predictions=predictions)
