"""
Script to check model performances
"""
import requests
import json
from pprint import pprint

# Base URL
BASE_URL = "http://127.0.0.1:5000"

def check_health():
    """Check API health and model status"""
    print("=" * 50)
    print("API HEALTH CHECK")
    print("=" * 50)
    response = requests.get(f"{BASE_URL}/api/health")
    data = response.json()
    pprint(data)
    print()

def check_simple_metrics():
    """Check simple model metrics from console output"""
    print("=" * 50)
    print("MODEL PERFORMANCE (FROM STARTUP)")
    print("=" * 50)
    
    # These are the accuracies from the console output
    model_accuracies = {
        'lightgbm': 0.9697,
        'xgboost': 0.5455,
        'random_forest': 0.9545,
        'gradient_boosting': 0.9697,
        'mlp': 0.8485,
        'svm': 0.8939
    }
    
    print("Individual Model Accuracies:")
    for model_name, accuracy in model_accuracies.items():
        print(f"  {model_name}: {accuracy:.4f}")
    print()
    
    print("Performance Summary:")
    best_model = max(model_accuracies.items(), key=lambda x: x[1])
    worst_model = min(model_accuracies.items(), key=lambda x: x[1])
    avg_accuracy = sum(model_accuracies.values()) / len(model_accuracies)
    
    print(f"  Best Model: {best_model[0]} ({best_model[1]:.4f})")
    print(f"  Worst Model: {worst_model[0]} ({worst_model[1]:.4f})")
    print(f"  Average Accuracy: {avg_accuracy:.4f}")
    print()

def test_prediction():
    """Test a prediction and see individual model contributions"""
    print("=" * 50)
    print("TEST PREDICTION WITH MODEL DETAILS")
    print("=" * 50)
    
    test_data = {
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
    
    response = requests.post(f"{BASE_URL}/api/predict", json=test_data)
    data = response.json()
    
    print(f"Final Prediction: {data['predictions']['occupancy']}")
    print(f"HVAC Power: {data['predictions']['hvac_power_kw']} kW")
    print(f"Overall Confidence: {data['predictions']['confidence']:.4f}")
    print()
    
    if 'model_details' in data:
        print("Individual Model Predictions:")
        for model_name, details in data['model_details']['classification_models'].items():
            print(f"  {model_name}:")
            print(f"    Prediction: {details['prediction']}")
            print(f"    Confidence: {details['confidence']:.4f}")
            print(f"    Weight: {details['weight']:.4f}")
        print()
        
        print("Regression Model Predictions:")
        for model_name, prediction in data['model_details']['regression_models'].items():
            print(f"  {model_name}: {prediction:.2f} kW")
        print()

if __name__ == "__main__":
    try:
        # Check if server is running
        check_health()
        
        # Get simple metrics from known console output
        check_simple_metrics()
        
        # Test a prediction
        test_prediction()
        
        print("=" * 50)
        print("PERFORMANCE CHECK COMPLETED SUCCESSFULLY")
        print("=" * 50)
        
    except requests.exceptions.ConnectionError:
        print("Error: Cannot connect to the server.")
        print("Make sure the Flask app is running on http://127.0.0.1:5000")
    except Exception as e:
        print(f"Error: {e}")