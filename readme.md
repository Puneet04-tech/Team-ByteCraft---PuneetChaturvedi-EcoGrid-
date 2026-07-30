# EcoGrid AI - Smart Energy Prediction System

An advanced AI-powered energy management system that combines triple-dataset integration, diverse machine learning models, and an interactive 3D web interface for real-time occupancy detection and HVAC power forecasting.

## 🌟 Features

### Machine Learning Capabilities (ML Track Compliant)
- **Multi-Model Ensemble**: 6 diverse ML architectures (LightGBM, XGBoost, Random Forest, Gradient Boosting, MLP, SVM)
- **Triple-Dataset Integration**: Combines occupancy detection, appliances energy consumption, and energy efficiency datasets
- **Real-time Predictions**: Instant occupancy classification and HVAC power forecasting
- **Weighted Ensemble**: Accuracy-weighted model averaging for optimal predictions
- **Comprehensive Validation**: 40% holdout validation set, 5-fold cross-validation, confusion matrix analysis
- **Overfitting Prevention**: Strong regularization, time-series validation, proper train-test splitting
- **Data Leakage Protection**: Proper train-test splitting before preprocessing
- **Model Transparency**: Individual model predictions, confidence scores, and feature importance

### Interactive Web Interface
- **3D Orbital Motion Visualization**: Stunning Three.js-based orbital ring animations
- **GSAP Scroll Animations**: Smooth scroll-based choreography effects
- **Interactive Prediction Form**: User-friendly interface for sensor data input
- **Real-time Results**: Instant API responses with confidence scores
- **Responsive Design**: Fixed prediction form with background orbital animations
- **Multiple Orbital Rings**: Main and side orbital rings with identical motion design

## 🚀 Tech Stack

### Backend
- **Flask**: Web framework for API serving
- **LightGBM**: Gradient boosting framework for ML models
- **XGBoost**: Extreme gradient boosting for ML models
- **Random Forest**: Bagging ensemble method
- **Gradient Boosting**: sklearn's boosting implementation
- **MLP**: Multi-layer perceptron neural network
- **SVM**: Support vector machine classifier
- **Pandas & NumPy**: Data processing and numerical operations
- **Scikit-learn**: Machine learning utilities and preprocessing
- **Gunicorn**: Production WSGI HTTP server

### Frontend
- **Three.js**: 3D graphics library for orbital motion visualization
- **GSAP**: High-performance animation library with ScrollTrigger
- **HTML5/CSS3**: Modern web technologies
- **JavaScript**: Client-side logic and API integration

### Deployment
- **Render**: Cloud platform for web service deployment
- **Git**: Version control and deployment automation

## 📋 Prerequisites

- Python 3.11 or higher
- pip (Python package manager)
- Git (for deployment)
- Node.js (optional, for local development)

## 🔧 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Puneet04-tech/Team-ByteCraft---PuneetChaturvedi-EcoGrid-.git
cd Team-ByteCraft---PuneetChaturvedi-EcoGrid-
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Prepare Dataset
Ensure the integrated dataset file exists:
```bash
ecogrid_triple_dataset_matrix.csv
```

## 🎯 Usage

### Local Development

#### Start the Flask Server
```bash
python app.py
```

The application will be available at:
- **Web Interface**: http://127.0.0.1:5000
- **API Endpoint**: http://127.0.0.1:5000/api/predict
- **Health Check**: http://127.0.0.1:5000/health

#### Using the Web Interface
1. Open your browser and navigate to http://127.0.0.1:5000
2. Fill in the sensor data form with:
   - Hour (0-23)
   - Day of Week (0-6, Monday=0)
   - Ambient Temperature (°C)
   - CO2 Level (ppm)
   - Light (lux)
   - Humidity (%)
   - Outside Temperature (°C)
   - Windspeed (m/s)
   - Relative Compactness (0-1)
   - Overall Height (meters)
3. Click "Generate Prediction" to get instant results
4. Watch the orbital rings animate as you scroll

### API Usage

#### Prediction Endpoint
```bash
POST /api/predict
Content-Type: application/json
```

#### Request Body
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

#### Response
```json
{
  "status": "success",
  "predictions": {
    "occupancy": "Medium",
    "hvac_power_kw": 42.35,
    "confidence": 0.892
  },
  "model_details": {
    "classification_models": {
      "lightgbm": {
        "prediction": "Medium",
        "confidence": 0.95,
        "weight": 0.18
      },
      "xgboost": {
        "prediction": "Medium",
        "confidence": 0.88,
        "weight": 0.15
      },
      "random_forest": {
        "prediction": "High",
        "confidence": 0.82,
        "weight": 0.17
      }
    },
    "regression_models": {
      "lightgbm": 42.1,
      "xgboost": 43.2,
      "random_forest": 41.8
    },
    "ensemble_method": "weighted_averaging",
    "num_models": 6
  },
  "input_features": {
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
  },
  "timestamp": "2026-07-31T12:34:56.789123"
}
```

#### Health Check
```bash
GET /health
```
Returns: `OK`

#### API Health Endpoint
```bash
GET /api/health
```
```json
{
  "status": "healthy",
  "model_loaded": true,
  "num_models": 6,
  "model_types": ["lightgbm", "xgboost", "random_forest", "gradient_boosting", "mlp", "svm"],
  "message": "EcoGrid AI API is running"
}
```

#### Model Metrics Endpoint
```bash
GET /api/model-metrics
```
```json
{
  "status": "success",
  "model_metrics": {
    "lightgbm": {
      "accuracy": 0.97,
      "confusion_matrix": [[...], [...], [...]],
      "classification_report": {...}
    },
    "xgboost": {...},
    "random_forest": {...}
  },
  "validation_results": {
    "feature_importance": {...},
    "cross_validation_scores": {...}
  },
  "num_classes": 3,
  "class_labels": ["High", "Low", "Medium"]
}
```

#### Confusion Matrix Endpoint
```bash
GET /api/confusion-matrix
```
```json
{
  "status": "success",
  "confusion_matrices": {
    "lightgbm": {
      "confusion_matrix": [[...], [...], [...]],
      "accuracy": 0.97,
      "classification_report": {...}
    },
    "xgboost": {...}
  },
  "class_labels": ["High", "Low", "Medium"],
  "feature_importance": {...}
}
```

## 🧠 Model Architecture

### Feature Engineering
- **Time-based Features**: Hour sine/cosine transformations, day of week, weekend indicators
- **Environmental Features**: Temperature, humidity, CO2 levels, light intensity
- **Building Features**: Relative compactness, overall height
- **Weather Features**: Outside temperature, windspeed
- **Feature Scaling**: StandardScaler for neural networks and SVM

### Multi-Model Ensemble Configuration

#### Classification Models (6 diverse architectures)
1. **LightGBM Classifier**
   - Gradient boosting with leaf-wise growth
   - Strong regularization (lambda=30.0, alpha=15.0)
   - Balanced class weights
   - Conservative depth (max_depth=2)

2. **XGBoost Classifier**
   - Extreme gradient boosting
   - Similar regularization to LightGBM
   - Scale-pos-weight for class balance
   - Robust to overfitting

3. **Random Forest Classifier**
   - Bagging ensemble method
   - Multiple decision trees
   - Class weight balancing
   - Feature importance analysis

4. **Gradient Boosting Classifier**
   - sklearn's boosting implementation
   - Sequential tree building
   - Subsampling for diversity
   - Different learning patterns

5. **MLP (Neural Network)**
   - Multi-layer perceptron: (64, 32) hidden layers
   - Early stopping for validation
   - L2 regularization (alpha=0.01)
   - Adaptive learning rate

6. **SVM (Support Vector Machine)**
   - RBF kernel for non-linear patterns
   - Class weight balancing
   - Probability estimation
   - Different decision boundaries

#### Regression Models (3 diverse architectures)
1. **LightGBM Regressor**
2. **XGBoost Regressor** 
3. **Random Forest Regressor**

### Ensemble Strategy
- **Weighted Averaging**: Models weighted by validation accuracy
- **Probability Averaging**: Soft voting for classification
- **Prediction Transparency**: Individual model predictions exposed
- **Dynamic Weighting**: Normalized weights based on performance

### Validation Strategy (Evaluation Criteria Compliant)
- **40% Holdout Set**: Separate validation dataset for evaluation
- **5-fold Cross-Validation**: Ensures model generalization
- **Stratified Sampling**: Maintains class distribution
- **Confusion Matrix Analysis**: Per-model performance metrics
- **Feature Importance**: Tree-based model interpretability
- **Time-series Splitting**: Prevents temporal autocorrelation

### Performance Metrics
- **Multi-Model Accuracy**: Individual and ensemble accuracy scores
- **Confusion Matrices**: Detailed per-class performance
- **Classification Reports**: Precision, recall, F1-score per class
- **Cross-Validation Scores**: Mean accuracy with standard deviation
- **Feature Importance**: Relative importance of input features
- **Confidence Scores**: Probability-based prediction confidence

## 🌐 Deployment

### Render Deployment

#### 1. Prepare Repository
Ensure the following files are present:
- `app.py` - Flask application
- `requirements.txt` - Python dependencies
- `Procfile` - Gunicorn configuration
- `templates/index.html` - Frontend template
- `ecogrid_triple_dataset_matrix.csv` - Dataset

#### 2. Deploy to Render
1. Create account at [render.com](https://render.com)
2. Connect GitHub repository
3. Create new Web Service
4. Configure:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Runtime**: Python 3
5. Deploy

#### 3. Access Your Application
- Render will provide a URL like: `https://your-app-name.onrender.com`
- API endpoint: `https://your-app-name.onrender.com/api/predict`

### Environment Variables
- `PORT`: Automatically set by Render (default: 5000)
- `PYTHON_VERSION`: Optional, set to 3.11

## 📁 Project Structure

```
Team-ByteCraft---PuneetChaturvedi-EcoGrid-/
├── app.py                          # Flask application with ML models
├── requirements.txt                # Python dependencies
├── Procfile                        # Render deployment configuration
├── ecogrid_triple_dataset_matrix.csv  # Integrated dataset
├── templates/
│   └── index.html                 # 3D orbital motion frontend
├── ecogrid_triple_ai_engine.py     # Core ML engine
├── ecogrid_triple_dataset_integration.py  # Data integration
├── run_ecogrid_triple.py          # Main execution script
├── EcoGrid.ipynb                  # Jupyter notebook (optional)
└── readme.md                      # This file
```

## 🎨 Frontend Features

### 3D Orbital Motion
- **Main Orbital Rings**: 4 rings with different radii, colors, and rotation speeds
- **Side Orbital Rings**: 2 rings on each side with identical motion design
- **Elliptical Motion**: Expanding and contracting based on sine/cosine waves
- **Scroll-based Animation**: Rings move and rotate as you scroll
- **Mouse Parallax**: Interactive response to mouse movement
- **Particle Effects**: Floating particles for visual depth

### Animation Timeline
- **Entrance Animation**: Rings scale up with stagger effect
- **Continuous Motion**: Rings revolve and breathe continuously
- **Scroll Choreography**: GSAP ScrollTrigger for scroll-based effects
- **Wobble Motion**: Subtle random rotation for natural movement

### UI Components
- **Fixed Prediction Form**: Centered form that stays during scroll
- **Results Section**: Fades in when you scroll down
- **Loading States**: Visual feedback during prediction processing
- **Responsive Design**: Works on different screen sizes

## 🔬 Model Training Process

### Data Integration
1. **Load Datasets**: Occupancy, appliances energy, energy efficiency
2. **Preprocessing**: Handle missing values, encode categorical variables
3. **Feature Engineering**: Create time-based and environmental features
4. **Integration**: Merge datasets on common features
5. **Validation**: Ensure data quality and consistency

### Training Pipeline
1. **Split Data**: Train-test split with stratification
2. **Encode Labels**: Transform categorical targets
3. **Train Models**: LightGBM ensemble training
4. **Validate**: Cross-validation and holdout testing
5. **Evaluate**: Accuracy, RMSE, and confidence metrics

### Production Deployment
1. **Model Loading**: Train models on application startup
2. **Feature Extraction**: Real-time feature calculation
3. **Prediction**: Ensemble prediction averaging
4. **Response**: JSON response with confidence scores

## 🐛 Troubleshooting

### Common Issues

#### Port Binding Issues
- **Problem**: Port already in use
- **Solution**: Change port in app.py or kill existing process

#### Model Loading Errors
- **Problem**: Dataset file not found
- **Solution**: Ensure `ecogrid_triple_dataset_matrix.csv` exists

#### Import Errors
- **Problem**: Missing dependencies
- **Solution**: Run `pip install -r requirements.txt`

#### Render Deployment Timeout
- **Problem**: Port scan timeout
- **Solution**: Check Procfile and app.py configuration

### Debug Mode
Enable debug mode for detailed error messages:
```python
app.run(debug=True, host='0.0.0.0')
```

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature-name`
3. **Make changes and test thoroughly**
4. **Commit with descriptive messages**
5. **Push to branch**: `git push origin feature-name`
6. **Submit a pull request**

### Development Guidelines
- Follow PEP 8 style guidelines
- Add comments for complex logic
- Test changes before committing
- Update documentation as needed

## 📊 Performance Optimization

### Model Optimization
- **Reduced Tree Count**: 50 trees for faster inference
- **Limited Depth**: Max depth of 2 to prevent overfitting
- **Regularization**: Strong L1/L2 regularization
- **Ensemble Averaging**: Multiple models for stability

### Web Performance
- **Lazy Loading**: Load models on startup
- **Caching**: Cache feature calculations
- **Efficient Animation**: GSAP for smooth performance
- **Responsive Design**: Optimize for different devices

## 🔒 Security Considerations

- **Input Validation**: Validate all sensor data inputs
- **Error Handling**: Comprehensive error catching
- **CORS Configuration**: Configured for API access
- **Environment Variables**: Sensitive data in environment variables
- **Dependency Management**: Regular dependency updates

## 📈 Future Enhancements

### Planned Features
- **Real-time Sensor Integration**: IoT device connectivity
- **Historical Analytics**: Prediction history and trends
- **Model Retraining**: Automated model updates
- **User Authentication**: Secure user access
- **Mobile App**: Native mobile application
- **Dashboard**: Advanced analytics dashboard

## 🏆 Evaluation Criteria Compliance

### ML Track Requirements Met

#### ✅ Effective Use of Locally Running AI/ML Models
- **6 Different Model Architectures**: LightGBM, XGBoost, Random Forest, Gradient Boosting, MLP, SVM
- **No API-Based Models**: All models run locally
- **Diverse Implementations**: Tree-based, neural networks, and kernel methods
- **Ensemble Approach**: Combines multiple local models for robustness

#### ✅ Confusion Matrix & Performance Analysis
- **Individual Model Confusion Matrices**: Available via `/api/confusion-matrix` endpoint
- **Classification Reports**: Precision, recall, F1-score for each class
- **Feature Importance**: Tree-based model interpretability
- **Cross-Validation Scores**: 5-fold CV with mean and standard deviation
- **40% Holdout Validation**: Separate dataset for evaluation

#### ✅ Overfitting Prevention
- **Strong Regularization**: L1/L2 regularization (lambda=30.0, alpha=15.0)
- **Conservative Architecture**: Max depth=2, limited tree count
- **Cross-Validation**: 5-fold CV to ensure generalization
- **Proper Splitting**: Train-test split before preprocessing
- **Stratified Sampling**: Maintains class distribution

#### ✅ Innovative Features
- **Multi-Model Ensemble**: Weighted accuracy-based model averaging
- **Prediction Transparency**: Individual model predictions exposed
- **Real-time Feature Scaling**: StandardScaler for neural networks
- **Dynamic Weighting**: Models weighted by validation performance
- **Comprehensive Metrics**: Multiple evaluation endpoints for analysis
- **Interactive 3D Visualization**: Novel frontend with orbital motion
- **Diverse Model Types**: Gradient boosting, bagging, neural networks, SVM

### Novel Applications
- **Weighted Ensemble System**: Accuracy-based dynamic model weighting
- **Multi-Architecture Approach**: Combines fundamentally different ML paradigms
- **Transparent Predictions**: Shows individual model contributions
- **Real-time ML API**: Instant predictions with confidence scores
- **3D Visualization**: Interactive orbital motion for result presentation
- **Comprehensive Analysis**: Detailed metrics and confusion matrices via API

### Model Improvements
- **Deep Learning**: Neural network models
- **Ensemble Methods**: More diverse model types
- **Feature Selection**: Automatic feature importance
- **Hyperparameter Tuning**: Automated optimization

## 📝 License

This project is developed for educational and research purposes. Please contact the maintainers for licensing information.

## 👥 Team

**Team ByteCraft - Puneet Chaturvedi**

- **Project Lead**: Puneet Chaturvedi
- **AI/ML Development**: Team ByteCraft
- **Frontend Development**: Team ByteCraft
- **Deployment**: Team ByteCraft

## 📞 Support

For support and questions:
- **Issues**: GitHub Issues
- **Email**: [team email]
- **Documentation**: This README file

## 🙏 Acknowledgments

- **Dataset Sources**: UCI Machine Learning Repository
- **Libraries**: Flask, LightGBM, Three.js, GSAP
- **Platform**: Render for cloud deployment
- **Community**: Open-source contributors

---

**Note**: This project demonstrates advanced AI-powered energy management with interactive visualization. The system combines machine learning, web development, and 3D graphics to create an engaging user experience while providing accurate predictions for energy optimization.

**Generated with [Devin](https://devin.ai)**