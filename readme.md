# EcoGrid AI - Smart Energy Prediction System

An advanced AI-powered energy management system that combines triple-dataset integration, machine learning models, and an interactive 3D web interface for real-time occupancy detection and HVAC power forecasting.

## 🌟 Features

### Machine Learning Capabilities
- **Triple-Dataset Integration**: Combines occupancy detection, appliances energy consumption, and energy efficiency datasets
- **Real-time Predictions**: Instant occupancy classification and HVAC power forecasting
- **Robust Model Training**: LightGBM-based ensemble models with cross-validation
- **Overfitting Prevention**: Comprehensive regularization and time-series validation
- **Data Leakage Protection**: Proper train-test splitting and preprocessing pipelines

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
  "message": "EcoGrid AI API is running"
}
```

## 🧠 Model Architecture

### Feature Engineering
- **Time-based Features**: Hour sine/cosine transformations, day of week, weekend indicators
- **Environmental Features**: Temperature, humidity, CO2 levels, light intensity
- **Building Features**: Relative compactness, overall height
- **Weather Features**: Outside temperature, windspeed

### Model Configuration
- **Classification Model**: LightGBM Classifier
  - Ensemble of 2 models for robustness
  - Balanced class weights for handling imbalance
  - Strong regularization (lambda=30.0, alpha=15.0)
  - Conservative depth (max_depth=2) to prevent overfitting
  
- **Regression Model**: LightGBM Regressor
  - Ensemble of 2 models for stability
  - Same hyperparameters as classification model
  - Optimized for HVAC power prediction

### Validation Strategy
- **5-fold Cross-Validation**: Ensures model generalization
- **Time-series Splitting**: Prevents temporal autocorrelation
- **Stratified Sampling**: Maintains class distribution
- **Holdout Set**: 40% test set for final evaluation

### Performance Metrics
- **Classification Accuracy**: ~85-90% (depends on data quality)
- **Regression RMSE**: Optimized for HVAC power prediction
- **Class Balance**: Improved recall for minority classes
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