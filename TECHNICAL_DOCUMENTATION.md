# EcoGrid AI - Technical Documentation

## 🏗️ System Architecture

### **Overview**
EcoGrid AI is a full-stack web application combining Flask backend, React-style frontend, and machine learning models for energy prediction and occupancy detection.

### **Architecture Diagram**
```
┌─────────────────────────────────────────────────────────────┐
│                     User Interface Layer                    │
│  ┌──────────────────┐         ┌──────────────────┐        │
│  │   index.html     │         │   WebGL Canvas   │        │
│  │   (Frontend)     │         │   (3D Animation) │        │
│  └────────┬─────────┘         └──────────────────┘        │
└───────────┼────────────────────────────────────────────────┘
            │ HTTP/JSON
            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                         │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Flask Web Application                     │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐    │  │
│  │  │   Routes   │  │   Models   │  │   Utils    │    │  │
│  │  └────────────┘  └────────────┘  └────────────┘    │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────┐
│                   Machine Learning Layer                     │
│  ┌──────────────────────────────────────────────────────┐  │
│  │            Multi-Model Ensemble System                │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐ │  │
│  │  │LightGBM  │ │ XGBoost  │ │RandomForest│ │   SVM  │ │  │
│  │  └──────────┘ └──────────┘ └──────────┘ └────────┘ │  │
│  │  ┌──────────┐ ┌──────────┐                         │  │
│  │  │Gradient  │ │   MLP     │                         │  │
│  │  │ Boosting │ │ (Neural)  │                         │  │
│  │  └──────────┘ └──────────┘                         │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────┐
│                      Data Layer                              │
│  ┌──────────────────┐         ┌──────────────────┐        │
│  │  CSV Datasets    │         │  Feature Storage  │        │
│  │  (Training Data) │         │  (Scaled Values) │        │
│  └──────────────────┘         └──────────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

## 🔧 Technology Stack

### **Backend Technologies**
- **Framework**: Flask 2.0+
- **Language**: Python 3.8+
- **ML Libraries**:
  - LightGBM: Gradient boosting framework
  - XGBoost: Extreme gradient boosting
  - Scikit-learn: Machine learning algorithms
  - Pandas: Data manipulation
  - NumPy: Numerical computing

### **Frontend Technologies**
- **Core**: HTML5, CSS3, JavaScript
- **3D Graphics**: Three.js (WebGL)
- **Animations**: GSAP (GreenSock Animation Platform)
- **Styling**: Custom CSS with CSS Variables
- **No Framework**: Vanilla JavaScript for performance

### **Data Processing**
- **Format**: CSV files
- **Libraries**: Pandas, NumPy
- **Preprocessing**: Scikit-learn StandardScaler
- **Encoding**: LabelEncoder for categorical data

## 📁 Project Structure

```
EcoGrid/
├── app.py                          # Main Flask application
├── requirements.txt                # Python dependencies
├── Procfile                        # Render deployment config
├── .gitignore                      # Git ignore rules
├── README.md                       # Project documentation
├── templates/
│   └── index.html                  # Frontend interface
├── ecogrid_triple_dataset_matrix.csv  # Training dataset
├── ecogrid_triple_dataset_integration.py  # Data integration
├── ecogrid_triple_ai_engine.py          # ML pipeline
├── check_model_performance.py           # Performance testing
└── USER_FLOW_JOURNEY.md                 # User documentation
```

## 🔌 API Endpoints

### **1. Health Check**
- **Endpoint**: `GET /api/health`
- **Purpose**: System status verification
- **Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_types": ["lightgbm", "xgboost", "random_forest", "gradient_boosting", "mlp", "svm"],
  "num_models": 6,
  "message": "EcoGrid AI API is running"
}
```

### **2. Prediction**
- **Endpoint**: `POST /api/predict`
- **Purpose**: Generate energy and occupancy predictions
- **Request Body**:
```json
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
```

- **Response**:
```json
{
  "predictions": {
    "occupancy": "Medium",
    "hvac_power_kw": 12.45,
    "confidence": 0.89
  },
  "model_details": {
    "classification_models": {
      "lightgbm": {"prediction": "Medium", "confidence": 0.92, "weight": 0.25},
      "xgboost": {"prediction": "High", "confidence": 0.78, "weight": 0.15},
      "random_forest": {"prediction": "Medium", "confidence": 0.88, "weight": 0.20},
      "gradient_boosting": {"prediction": "Medium", "confidence": 0.91, "weight": 0.20},
      "mlp": {"prediction": "Low", "confidence": 0.72, "weight": 0.10},
      "svm": {"prediction": "Medium", "confidence": 0.85, "weight": 0.10}
    },
    "regression_models": {
      "lightgbm": 12.3,
      "xgboost": 13.1,
      "random_forest": 12.5,
      "gradient_boosting": 12.4,
      "mlp": 11.8,
      "svm": 12.7
    }
  },
  "timestamp": "2026-07-31T14:30:00Z"
}
```

### **3. Model Metrics**
- **Endpoint**: `GET /api/model-metrics`
- **Purpose**: Retrieve detailed model performance metrics
- **Response**:
```json
{
  "status": "success",
  "num_classes": 3,
  "class_labels": ["High", "Low", "Medium"],
  "model_metrics": {
    "lightgbm": {"accuracy": 0.9697},
    "xgboost": {"accuracy": 0.5455},
    "random_forest": {"accuracy": 0.9545},
    "gradient_boosting": {"accuracy": 0.9697},
    "mlp": {"accuracy": 0.8485},
    "svm": {"accuracy": 0.8939}
  },
  "validation_results": {
    "cross_validation_scores": {
      "lightgbm": {"mean": 0.72, "std": 0.32, "scores": [0.85, 0.65, 0.70, 0.68, 0.72]}
    },
    "feature_importance": {
      "lightgbm": {"Hour_Sin": 0.15, "Ambient_Temp_C": 0.12, "CO2_Level": 0.10}
    }
  }
}
```

### **4. Confusion Matrix**
- **Endpoint**: `GET /api/confusion-matrix`
- **Purpose**: Get confusion matrix for classification models
- **Response**:
```json
{
  "class_labels": ["High", "Low", "Medium"],
  "confusion_matrices": {
    "lightgbm": {
      "confusion_matrix": [[12, 1, 1], [0, 38, 0], [0, 0, 14]],
      "accuracy": 0.9697,
      "classification_report": {
        "High": {"precision": 1.0, "recall": 0.86, "f1-score": 0.92},
        "Low": {"precision": 0.97, "recall": 1.0, "f1-score": 0.99},
        "Medium": {"precision": 0.93, "recall": 1.0, "f1-score": 0.97}
      }
    }
  }
}
```

## 🤖 Machine Learning Pipeline

### **Data Integration**
```python
# File: ecogrid_triple_dataset_integration.py
# Purpose: Integrate multiple datasets into unified training matrix

1. Load individual datasets (energy, occupancy, environmental)
2. Perform time-based alignment
3. Handle missing values
4. Create engineered features:
   - Hour_Sin, Hour_Cos (cyclical time encoding)
   - Temp_Rolling_Mean (temporal smoothing)
   - DayOfWeek, IsWeekend (temporal features)
5. Save unified dataset: ecogrid_triple_dataset_matrix.csv
```

### **Model Training Pipeline**
```python
# File: ecogrid_triple_ai_engine.py
# Purpose: Train and evaluate ML models

class EcoGridTripleEngineFinal:
    def __init__(self):
        # Feature sets
        self.core_feature_cols = [...]
        self.enhanced_feature_cols = [...]
        
        # Model hyperparameters
        self.lgb_params = {
            'n_estimators': 50,
            'max_depth': 2,
            'learning_rate': 0.1,
            'reg_lambda': 30.0,
            'class_weight': 'balanced'
        }
    
    def run_pipeline(self):
        # 1. Load integrated dataset
        # 2. Time-based train-test split (40% test)
        # 3. Label encoding (prevent data leakage)
        # 4. Train 6 classification models
        # 5. Train 6 regression models
        # 6. Evaluate with time-series cross-validation
        # 7. Generate performance visualizations
```

### **Multi-Model Ensemble System**

#### **Classification Models (Occupancy Detection)**
1. **LightGBM Classifier**
   - Gradient boosting framework
   - Handles categorical features well
   - Fast training and prediction
   - High accuracy (97%)

2. **XGBoost Classifier**
   - Extreme gradient boosting
   - Regularization capabilities
   - Handles missing values
   - Moderate accuracy (55%)

3. **Random Forest Classifier**
   - Ensemble of decision trees
   - Robust to overfitting
   - Feature importance analysis
   - High accuracy (95%)

4. **Gradient Boosting Classifier**
   - Sequential tree building
   - Error correction focus
   - High accuracy (97%)

5. **MLP Classifier**
   - Multi-layer perceptron
   - Neural network approach
   - Requires feature scaling
   - Moderate accuracy (85%)

6. **SVM Classifier**
   - Support vector machine
   - Effective in high dimensions
   - Kernel trick for non-linear
   - Good accuracy (89%)

#### **Regression Models (Energy Prediction)**
Same 6 algorithms configured for regression tasks to predict HVAC power consumption.

### **Prediction Logic**

#### **Weighted Ensemble Approach**
```python
def predict_occupancy(input_features):
    # Get predictions from all 6 models
    model_predictions = []
    for model in classification_models:
        pred = model.predict(input_features)
        conf = model.predict_proba(input_features).max()
        model_predictions.append((pred, conf))
    
    # Weight models by validation accuracy
    weights = {
        'lightgbm': 0.25,
        'random_forest': 0.20,
        'gradient_boosting': 0.20,
        'xgboost': 0.15,
        'mlp': 0.10,
        'svm': 0.10
    }
    
    # Weighted voting
    weighted_votes = defaultdict(float)
    for (pred, conf), (model, weight) in zip(model_predictions, weights.items()):
        weighted_votes[pred] += conf * weight
    
    # Return prediction with highest weighted vote
    final_prediction = max(weighted_votes, key=weighted_votes.get)
    confidence = weighted_votes[final_prediction] / sum(weights.values())
    
    return final_prediction, confidence
```

#### **Regression Ensemble**
```python
def predict_hvac_power(input_features):
    # Get predictions from all 6 regression models
    predictions = []
    for model in regression_models:
        pred = model.predict(input_features)
        predictions.append(pred)
    
    # Simple averaging (can be weighted by RMSE)
    final_prediction = np.mean(predictions)
    
    return final_prediction
```

## 🎨 Frontend Architecture

### **HTML Structure**
```html
<!DOCTYPE html>
<html>
<head>
  <title>ECOGRID AI — Smart Energy Prediction</title>
  <style>/* CSS styles */</style>
</head>
<body>
  <!-- Visual Effects -->
  <div class="hud-overlay"></div>
  <div class="vignette"></div>
  <div class="grain"></div>
  <div id="webgl-container"></div>
  
  <!-- Navigation -->
  <nav>
    <div class="logo-group">
      <div class="logo">ECOGRID</div>
      <div class="status-badge">System Online</div>
    </div>
  </nav>
  
  <!-- Hero Section -->
  <div class="content">
    <section>
      <div class="hero-eyebrow">Triple-Dataset AI Engine</div>
      <div class="hero-title">Smart Energy Prediction</div>
      <div class="hero-sub">Real-time occupancy detection...</div>
    </section>
  </div>
  
  <!-- Prediction Form -->
  <div class="form-section">
    <div class="form-grid">
      <!-- 10 input fields -->
    </div>
    <button class="predict-btn">Generate Prediction</button>
  </div>
  
  <!-- Results Section -->
  <div class="content">
    <section>
      <div class="results-section">
        <!-- Prediction results -->
      </div>
    </section>
  </div>
  
  <!-- Scripts -->
  <script src="three.js"></script>
  <script src="gsap.js"></script>
  <script>
    // Application logic
  </script>
</body>
</html>
```

### **CSS Architecture**
```css
:root {
  --blue: #0074ff;
  --red: #ff1133;
  --ink: #020204;
  --green: #00ff88;
}

/* Global Styles */
body {
  background: var(--ink);
  color: #fff;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto;
}

/* Visual Effects */
.hud-overlay {
  background-image: linear-gradient(...);
  mix-blend-mode: screen;
}

.vignette {
  background: radial-gradient(...);
}

.grain {
  background-image: url("data:image/svg+xml...");
  mix-blend-mode: overlay;
}

/* Component Styles */
.form-section {
  background: rgba(6,6,13,.62);
  backdrop-filter: blur(22px);
  border: 1px solid rgba(0,116,255,.22);
}

/* Responsive Design */
@media (max-width: 720px) {
  .form-grid { grid-template-columns: 1fr; }
}
```

### **JavaScript Architecture**
```javascript
// WebGL 3D Animation
(function() {
  const container = document.getElementById('webgl-container');
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  
  // Create orbital rings
  function createOrbitalRings() {
    const geometries = [
      new THREE.TorusGeometry(3, 0.02, 16, 100),
      new THREE.TorusGeometry(4, 0.015, 16, 100),
      new THREE.TorusGeometry(5, 0.01, 16, 100)
    ];
    
    geometries.forEach((geometry, i) => {
      const material = new THREE.MeshBasicMaterial({ 
        color: 0x0074ff,
        transparent: true,
        opacity: 0.6 - (i * 0.15)
      });
      const ring = new THREE.Mesh(geometry, material);
      scene.add(ring);
      orbitalRings.push(ring);
    });
  }
  
  // Animation loop
  function animate() {
    requestAnimationFrame(animate);
    orbitalRings.forEach((ring, i) => {
      ring.rotation.x += 0.001 * (i + 1);
      ring.rotation.y += 0.002 * (i + 1);
    });
    renderer.render(scene, camera);
  }
})();

// Prediction API Call
async function makePrediction() {
  const formData = {
    hour: parseInt(document.getElementById('hour').value),
    day_of_week: parseInt(document.getElementById('day_of_week').value),
    ambient_temp_c: parseFloat(document.getElementById('ambient_temp_c').value),
    // ... other fields
  };
  
  try {
    const response = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(formData)
    });
    
    const data = await response.json();
    displayResults(data);
  } catch (error) {
    console.error('Prediction failed:', error);
  }
}

// Results Display
function displayResults(data) {
  const occupancyResult = document.getElementById('occupancy-result');
  const powerResult = document.getElementById('power-result');
  const confidenceResult = document.getElementById('confidence-result');
  
  occupancyResult.textContent = data.predictions.occupancy;
  occupancyResult.className = 'result-value occupancy-' + 
    data.predictions.occupancy.toLowerCase();
  
  powerResult.textContent = data.predictions.hvac_power_kw.toFixed(2) + ' kW';
  confidenceResult.textContent = (data.predictions.confidence * 100).toFixed(1) + '%';
  
  document.getElementById('results-content').style.display = 'block';
}
```

## 🗄️ Database & Data Management

### **Data Storage**
- **Format**: CSV files
- **Training Data**: `ecogrid_triple_dataset_matrix.csv`
- **Features**: 19 columns including time, environmental, and energy data
- **Size**: 165 rows (sample dataset)

### **Data Schema**
```csv
hourly_timestamp,Hour_Sin,Hour_Cos,DayOfWeek,IsWeekend,Ambient_Temp_C,
Temp_Rolling_Mean,Occupancy_Category,HVAC_Power_kW,CO2_Level,Light,
Humidity,Appliance_Energy_Wh,Lighting_Energy_Wh,Outside_Temp_C,
Windspeed,Efficiency_Category,Relative_Compactness,Overall_Height
```

### **Feature Engineering**
```python
# Cyclical time encoding
df['Hour_Sin'] = np.sin(2 * np.pi * df['hour']/24)
df['Hour_Cos'] = np.cos(2 * np.pi * df['hour']/24)

# Temporal features
df['DayOfWeek'] = pd.to_datetime(df['hourly_timestamp']).dt.dayofweek
df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)

# Rolling averages
df['Temp_Rolling_Mean'] = df['Ambient_Temp_C'].rolling(window=3).mean()
```

## 🔒 Security & Performance

### **Security Measures**
- **Input Validation**: Server-side parameter validation
- **Error Handling**: Graceful error responses
- **CORS**: Configured for local development
- **No Secrets**: No API keys or sensitive data stored

### **Performance Optimization**
- **Model Caching**: Models loaded once at startup
- **Feature Scaling**: Pre-computed scalers
- **Efficient Algorithms**: LightGBM for fast inference
- **Asynchronous Processing**: Non-blocking API calls

### **Scalability Considerations**
- **Horizontal Scaling**: Stateless Flask application
- **Load Balancing**: Can be deployed behind load balancer
- **Database Integration**: Can be upgraded to PostgreSQL
- **Caching**: Redis can be added for result caching

## 🚀 Deployment

### **Local Development**
```bash
# Install dependencies
pip install -r requirements.txt

# Run development server
python app.py

# Access application
http://127.0.0.1:5000
```

### **Render Deployment**
```yaml
# Procfile
web: gunicorn app:app --bind 0.0.0.0:$PORT --workers 4 --timeout 120

# Environment Variables
PORT: (auto-assigned by Render)
PYTHON_VERSION: 3.8+
```

### **Docker Deployment (Optional)**
```dockerfile
FROM python:3.8-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]
```

## 🧪 Testing & Validation

### **Model Performance Testing**
```python
# File: check_model_performance.py
# Purpose: Validate model metrics via API

def check_health():
    response = requests.get("http://127.0.0.1:5000/api/health")
    return response.json()

def test_prediction():
    test_data = {
        "hour": 14,
        "day_of_week": 2,
        "ambient_temp_c": 34.5,
        # ... other fields
    }
    response = requests.post("http://127.0.0.1:5000/api/predict", json=test_data)
    return response.json()
```

### **Performance Metrics**
- **Classification Accuracy**: 97% (LightGBM, Gradient Boosting)
- **Regression RMSE**: 0.20 kW (Test set)
- **Cross-Validation**: 72.6% ± 32.3% (Time-series CV)
- **Inference Time**: < 100ms per prediction

## 📊 Monitoring & Logging

### **Application Logging**
```python
# Flask logging configuration
app.logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
app.logger.addHandler(handler)
```

### **Performance Monitoring**
- **Response Time**: API endpoint timing
- **Error Rate**: Failed prediction requests
- **Model Accuracy**: Periodic validation checks
- **System Health**: Health check endpoint

## 🔧 Development Workflow

### **Git Workflow**
```bash
# Feature development
git checkout -b feature/new-model
git add .
git commit -m "Add new ML model"
git push origin feature/new-model

# Main branch updates
git checkout main
git merge feature/new-model
git push origin main
```

### **Code Quality**
- **PEP 8**: Python style guide compliance
- **ESLint**: JavaScript linting (if added)
- **Type Hints**: Python type annotations
- **Documentation**: Docstrings for functions

## 🐛 Troubleshooting

### **Common Issues**

#### **1. Port Already in Use**
```bash
# Error: Address already in use
# Solution: Kill existing process or change port
lsof -ti:5000 | xargs kill -9
```

#### **2. Model Loading Errors**
```python
# Error: Model file not found
# Solution: Ensure training pipeline runs before app startup
python ecogrid_triple_ai_engine.py
```

#### **3. CORS Issues**
```python
# Error: CORS policy blocking requests
# Solution: Add CORS support to Flask
from flask_cors import CORS
CORS(app)
```

#### **4. Memory Issues**
```python
# Error: Out of memory during model training
# Solution: Reduce model complexity or dataset size
self.lgb_params['n_estimators'] = 30  # Reduce from 50
```

## 📈 Future Technical Enhancements

### **Planned Upgrades**
1. **Real-time Data Integration**
   - WebSocket support for live sensor data
   - Streaming predictions
   - Dynamic model updating

2. **Advanced ML Features**
   - Deep learning models (LSTM, Transformer)
   - Automated hyperparameter tuning
   - Model explainability (SHAP values)

3. **Infrastructure Improvements**
   - Database migration (PostgreSQL)
   - Redis caching layer
   - Kubernetes deployment
   - CI/CD pipeline

4. **Frontend Enhancements**
   - React.js migration
   - Advanced visualizations (D3.js)
   - Mobile app development
   - PWA capabilities

---

**Document Version**: 1.0  
**Last Updated**: 2026-07-31  
**Technical Contact**: Development Team  
**Application Version**: EcoGrid AI v1.0