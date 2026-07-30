"""
ECOGRID AI: FLASK BACKEND API
Serves the trained model for real-time predictions
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
import warnings
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Global variables for models
classification_models = None
regression_models = None
label_encoder = None
feature_cols = None

def load_models():
    """Load and train the models on startup"""
    global classification_models, regression_models, label_encoder, feature_cols
    
    try:
        # Load the integrated dataset
        df = pd.read_csv('ecogrid_triple_dataset_matrix.csv')
        
        # Feature columns
        feature_cols = [
            'Hour_Sin', 'Hour_Cos', 'DayOfWeek', 'IsWeekend', 
            'Ambient_Temp_C', 'Temp_Rolling_Mean',
            'CO2_Level', 'Light', 'Humidity',
            'Outside_Temp_C', 'Windspeed',
            'Relative_Compactness', 'Overall_Height'
        ]
        
        # Preprocess data
        df = df.dropna(subset=['HVAC_Power_kW', 'Occupancy_Category'])
        
        # Encode categorical variables
        label_encoder = LabelEncoder()
        df['Occupancy_Label'] = label_encoder.fit_transform(df['Occupancy_Category'])
        
        # Prepare features
        X = df[feature_cols]
        y_cls = df['Occupancy_Label']
        y_reg = df['HVAC_Power_kW']
        
        # Train classification models (LightGBM only)
        lgb_cls_params = {
            'n_estimators': 50, 'max_depth': 2, 'learning_rate': 0.1,
            'reg_lambda': 30.0, 'reg_alpha': 15.0, 'min_child_samples': 20,
            'subsample': 0.7, 'colsample_bytree': 0.7, 'class_weight': 'balanced',
            'random_state': 42, 'verbose': -1
        }
        
        m1_cls = lgb.LGBMClassifier(**lgb_cls_params).fit(X, y_cls)
        m2_cls = lgb.LGBMClassifier(**lgb_cls_params).fit(X, y_cls)
        classification_models = {'lgb1': m1_cls, 'lgb2': m2_cls}
        
        # Train regression models (LightGBM only)
        m1_reg = lgb.LGBMRegressor(**lgb_cls_params).fit(X, y_reg)
        m2_reg = lgb.LGBMRegressor(**lgb_cls_params).fit(X, y_reg)
        regression_models = {'lgb1': m1_reg, 'lgb2': m2_reg}
        
        print("Models trained successfully")
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        return False

@app.route('/')
def home():
    """Serve the frontend"""
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': classification_model is not None,
        'message': 'EcoGrid AI API is running'
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Make predictions based on sensor data using trained models
    Expected JSON payload:
    {
        "hour": 14,
        "day_of_week": 2,
        "ambient_temp_c": 34.5,
        "co2_level": 800,
        "light": 400,
        "humidity": 45,
        "outside_temp_c": 32.0,
        "windspeed": 5.0,
        "relative_compactness": 0.9,
        "overall_height": 7.0
    }
    """
    try:
        data = request.get_json()
        
        # Extract features
        hour = data.get('hour', 12)
        day_of_week = data.get('day_of_week', 2)
        ambient_temp_c = data.get('ambient_temp_c', 25.0)
        co2_level = data.get('co2_level', 500)
        light = data.get('light', 300)
        humidity = data.get('humidity', 50)
        outside_temp_c = data.get('outside_temp_c', 24.0)
        windspeed = data.get('windspeed', 3.0)
        relative_compactness = data.get('relative_compactness', 0.8)
        overall_height = data.get('overall_height', 3.5)
        
        # Calculate time-based features
        hour_sin = np.sin(2 * np.pi * hour / 24.0)
        hour_cos = np.cos(2 * np.pi * hour / 24.0)
        is_weekend = 1 if day_of_week >= 5 else 0
        temp_rolling_mean = ambient_temp_c  # Simplified for single prediction
        
        # Create feature vector in correct order
        features = np.array([[
            hour_sin, hour_cos, day_of_week, is_weekend,
            ambient_temp_c, temp_rolling_mean,
            co2_level, light, humidity,
            outside_temp_c, windspeed,
            relative_compactness, overall_height
        ]])
        
        # Use trained models for predictions
        if classification_models and regression_models:
            # Classification prediction (occupancy)
            cls_prob_lgb1 = classification_models['lgb1'].predict_proba(features)[0]
            cls_prob_lgb2 = classification_models['lgb2'].predict_proba(features)[0]
            cls_prob = (cls_prob_lgb1 + cls_prob_lgb2) / 2
            occupancy_pred = np.argmax(cls_prob)
            occupancy = label_encoder.inverse_transform([occupancy_pred])[0]
            
            # Calculate confidence based on probability
            confidence = np.max(cls_prob)
            
            # Regression prediction (power)
            power_pred_lgb1 = regression_models['lgb1'].predict(features)[0]
            power_pred_lgb2 = regression_models['lgb2'].predict(features)[0]
            predicted_power = round((power_pred_lgb1 + power_pred_lgb2) / 2, 2)
        else:
            # Fallback to rule-based if models not loaded
            if co2_level > 1000:
                occupancy = 'High'
            elif co2_level > 600:
                occupancy = 'Medium'
            else:
                occupancy = 'Low'
            
            occupancy_weights = {"High": 28.5, "Medium": 14.0, "Low": 3.5}
            base_power = 12.0
            occupancy_power = occupancy_weights[occupancy]
            temp_adjustment = max(0, (ambient_temp_c - 26) * 2.1)
            predicted_power = round(base_power + occupancy_power + temp_adjustment, 2)
            confidence = 0.85
        
        response = {
            'status': 'success',
            'predictions': {
                'occupancy': occupancy,
                'hvac_power_kw': predicted_power,
                'confidence': round(confidence, 3)
            },
            'input_features': {
                'hour': hour,
                'day_of_week': day_of_week,
                'ambient_temp_c': ambient_temp_c,
                'co2_level': co2_level,
                'light': light,
                'humidity': humidity,
                'outside_temp_c': outside_temp_c,
                'windspeed': windspeed,
                'relative_compactness': relative_compactness,
                'overall_height': overall_height
            },
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/features', methods=['GET'])
def get_features():
    """Get information about required features"""
    return jsonify({
        'required_features': [
            'hour (0-23)',
            'day_of_week (0-6, Monday=0)',
            'ambient_temp_c (Celsius)',
            'co2_level (ppm)',
            'light (lux)',
            'humidity (%)',
            'outside_temp_c (Celsius)',
            'windspeed (m/s)',
            'relative_compactness (0-1)',
            'overall_height (meters)'
        ],
        'model_info': {
            'name': 'EcoGrid Triple-Dataset AI',
            'version': '1.0',
            'type': 'Classification + Regression'
        }
    })

if __name__ == '__main__':
    load_models()
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)