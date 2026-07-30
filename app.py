"""
ECOGRID AI: FLASK BACKEND API
Serves the trained model for real-time predictions
Multi-model ensemble with confusion matrix analysis
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
import warnings
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import xgboost as xgb
warnings.filterwarnings('ignore')

# Global variables for models
classification_models = None
regression_models = None
label_encoder = None
feature_cols = None
scaler = None
model_metrics = None
validation_results = None

def load_models():
    """Load and train multiple diverse ML models on startup"""
    global classification_models, regression_models, label_encoder, feature_cols, scaler, model_metrics, validation_results
    
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
        
        # Scale features for neural networks and SVM
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split for validation (40% holdout as per evaluation criteria)
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_cls, test_size=0.4, stratify=y_cls, random_state=42
        )
        X_train_scaled, X_val_scaled, _, _ = train_test_split(
            X_scaled, y_cls, test_size=0.4, stratify=y_cls, random_state=42
        )
        
        # Train diverse classification models (Multiple architectures as per criteria)
        print("Training diverse ML models...")
        
        # 1. LightGBM (Gradient Boosting)
        lgb_cls_params = {
            'n_estimators': 50, 'max_depth': 2, 'learning_rate': 0.1,
            'reg_lambda': 30.0, 'reg_alpha': 15.0, 'min_child_samples': 20,
            'subsample': 0.7, 'colsample_bytree': 0.7, 'class_weight': 'balanced',
            'random_state': 42, 'verbose': -1
        }
        lgb_model = lgb.LGBMClassifier(**lgb_cls_params).fit(X_train, y_train)
        
        # 2. XGBoost (Gradient Boosting)
        xgb_model = xgb.XGBClassifier(
            n_estimators=50, max_depth=2, learning_rate=0.1,
            reg_lambda=30.0, reg_alpha=15.0, subsample=0.7, colsample_bytree=0.7,
            scale_pos_weight='balanced', random_state=42, verbosity=0
        ).fit(X_train, y_train)
        
        # 3. Random Forest (Bagging)
        rf_model = RandomForestClassifier(
            n_estimators=50, max_depth=2, min_samples_split=20,
            class_weight='balanced', random_state=42, n_jobs=-1
        ).fit(X_train, y_train)
        
        # 4. Gradient Boosting (sklearn)
        gb_model = GradientBoostingClassifier(
            n_estimators=50, max_depth=2, learning_rate=0.1,
            subsample=0.7, random_state=42
        ).fit(X_train, y_train)
        
        # 5. MLP (Neural Network)
        mlp_model = MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=200, learning_rate_init=0.01,
            alpha=0.01, random_state=42, early_stopping=True, validation_fraction=0.2
        ).fit(X_train_scaled, y_train)
        
        # 6. SVM (Support Vector Machine)
        svm_model = SVC(
            C=1.0, kernel='rbf', gamma='scale', class_weight='balanced',
            probability=True, random_state=42
        ).fit(X_train_scaled, y_train)
        
        classification_models = {
            'lightgbm': lgb_model,
            'xgboost': xgb_model,
            'random_forest': rf_model,
            'gradient_boosting': gb_model,
            'mlp': mlp_model,
            'svm': svm_model
        }
        
        # Train regression models
        lgb_reg = lgb.LGBMRegressor(**lgb_cls_params).fit(X_train, y_train)
        xgb_reg = xgb.XGBRegressor(
            n_estimators=50, max_depth=2, learning_rate=0.1,
            reg_lambda=30.0, reg_alpha=15.0, subsample=0.7, colsample_bytree=0.7,
            random_state=42, verbosity=0
        ).fit(X_train, y_train)
        rf_reg = RandomForestRegressor(
            n_estimators=50, max_depth=2, min_samples_split=20,
            random_state=42, n_jobs=-1
        ).fit(X_train, y_train)
        
        regression_models = {
            'lightgbm': lgb_reg,
            'xgboost': xgb_reg,
            'random_forest': rf_reg
        }
        
        # Calculate model metrics and confusion matrix
        model_metrics = {}
        for name, model in classification_models.items():
            if name in ['mlp', 'svm']:
                X_test = X_val_scaled
            else:
                X_test = X_val
            
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_val, y_pred)
            cm = confusion_matrix(y_val, y_pred)
            
            model_metrics[name] = {
                'accuracy': accuracy,
                'confusion_matrix': cm.tolist(),
                'classification_report': classification_report(y_val, y_pred, output_dict=True)
            }
        
        # Store validation results for evaluation
        validation_results = {
            'feature_importance': {},
            'cross_validation_scores': {}
        }
        
        # Get feature importance from tree-based models
        for name in ['lightgbm', 'xgboost', 'random_forest', 'gradient_boosting']:
            if hasattr(classification_models[name], 'feature_importances_'):
                validation_results['feature_importance'][name] = dict(zip(
                    feature_cols, classification_models[name].feature_importances_.tolist()
                ))
        
        # Cross-validation scores
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for name, model in classification_models.items():
            if name in ['mlp', 'svm']:
                X_cv = X_scaled
            else:
                X_cv = X
            
            cv_scores = cross_val_score(model, X_cv, y_cls, cv=cv, scoring='accuracy')
            validation_results['cross_validation_scores'][name] = {
                'mean': cv_scores.mean(),
                'std': cv_scores.std(),
                'scores': cv_scores.tolist()
            }
        
        print("Diverse ML models trained successfully")
        print(f"Model accuracies: { {name: metrics['accuracy'] for name, metrics in model_metrics.items()} }")
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        return False
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        return False

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    """Serve the frontend"""
    return render_template('index.html')

@app.route('/health')
def health_check():
    """Simple health check for Render"""
    return 'OK', 200

@app.route('/api/health', methods=['GET'])
def api_health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': classification_models is not None,
        'num_models': len(classification_models) if classification_models else 0,
        'model_types': list(classification_models.keys()) if classification_models else [],
        'message': 'EcoGrid AI API is running'
    })

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': classification_models is not None,
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
            # Classification prediction (occupancy) - Ensemble of diverse models
            cls_predictions = []
            cls_probabilities = []
            
            for name, model in classification_models.items():
                if name in ['mlp', 'svm']:
                    features_scaled = scaler.transform(features)
                    prob = model.predict_proba(features_scaled)[0]
                else:
                    prob = model.predict_proba(features)[0]
                
                cls_probabilities.append(prob)
                cls_predictions.append(np.argmax(prob))
            
            # Weighted ensemble based on model accuracy
            weights = [model_metrics[name]['accuracy'] for name in classification_models.keys()]
            weights = np.array(weights) / sum(weights)  # Normalize weights
            
            # Weighted probability averaging
            cls_prob = np.average(cls_probabilities, axis=0, weights=weights)
            occupancy_pred = np.argmax(cls_prob)
            occupancy = label_encoder.inverse_transform([occupancy_pred])[0]
            
            # Calculate confidence based on weighted probability
            confidence = np.max(cls_prob)
            
            # Individual model predictions for transparency
            model_predictions = {}
            for i, name in enumerate(classification_models.keys()):
                pred_label = label_encoder.inverse_transform([cls_predictions[i]])[0]
                model_predictions[name] = {
                    'prediction': pred_label,
                    'confidence': float(cls_probabilities[i][cls_predictions[i]]),
                    'weight': float(weights[i])
                }
            
            # Regression prediction (power) - Ensemble of diverse models
            reg_predictions = []
            for name, model in regression_models.items():
                pred = model.predict(features)[0]
                reg_predictions.append(pred)
            
            # Simple averaging for regression
            predicted_power = round(np.mean(reg_predictions), 2)
            
            # Individual regression predictions
            regression_details = {}
            for i, name in enumerate(regression_models.keys()):
                regression_details[name] = float(reg_predictions[i])
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
            'model_details': {
                'classification_models': model_predictions,
                'regression_models': regression_details,
                'ensemble_method': 'weighted_averaging',
                'num_models': len(classification_models)
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
            'version': '2.0',
            'type': 'Multi-Model Ensemble Classification + Regression',
            'model_types': ['LightGBM', 'XGBoost', 'Random Forest', 'Gradient Boosting', 'MLP', 'SVM']
        }
    })

@app.route('/api/model-metrics', methods=['GET'])
def get_model_metrics():
    """Get detailed model metrics including confusion matrices"""
    if model_metrics is None:
        return jsonify({
            'status': 'error',
            'message': 'Model metrics not available'
        }), 404
    
    return jsonify({
        'status': 'success',
        'model_metrics': model_metrics,
        'validation_results': validation_results,
        'feature_columns': feature_cols,
        'num_classes': len(label_encoder.classes_),
        'class_labels': label_encoder.classes_.tolist()
    })

@app.route('/api/confusion-matrix', methods=['GET'])
def get_confusion_matrix():
    """Get confusion matrix for evaluation"""
    if model_metrics is None:
        return jsonify({
            'status': 'error',
            'message': 'Confusion matrix not available'
        }), 404
    
    # Return confusion matrices for all models
    confusion_matrices = {}
    for name, metrics in model_metrics.items():
        confusion_matrices[name] = {
            'confusion_matrix': metrics['confusion_matrix'],
            'accuracy': metrics['accuracy'],
            'classification_report': metrics['classification_report']
        }
    
    return jsonify({
        'status': 'success',
        'confusion_matrices': confusion_matrices,
        'class_labels': label_encoder.classes_.tolist(),
        'feature_importance': validation_results['feature_importance'] if validation_results else None
    })

# Load models when module is imported (for Gunicorn)
print("Starting model loading...")
load_models()
print("Model loading completed successfully")

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')