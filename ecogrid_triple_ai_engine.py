"""
ECOGRID AI: TRIPLE-DATASET AI ENGINE
Enhanced Multi-Task Heterogeneous Gradient Boosting with Occupancy + Appliances + Energy Efficiency
"""

import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import lightgbm as lgb
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings('ignore')
np.random.seed(42)

def console_log(msg: str, delay: float = 0.01):
    """Outputs structured log messages directly to standard output."""
    try:
        print(msg, flush=True)
    except (OSError, UnicodeEncodeError):
        # Fallback for encoding issues - remove special characters
        safe_msg = ''.join(char for char in msg if ord(char) < 128)
        print(safe_msg, flush=True)
    time.sleep(delay)

class EcoGridTripleEngine:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
        # Core EcoGrid features
        self.core_feature_cols = ['Hour_Sin', 'Hour_Cos', 'DayOfWeek', 'IsWeekend', 'Ambient_Temp_C', 'Temp_Rolling_Mean']
        
        # Enhanced features from triple dataset
        self.enhanced_feature_cols = [
            'Hour_Sin', 'Hour_Cos', 'DayOfWeek', 'IsWeekend', 
            'Ambient_Temp_C', 'Temp_Rolling_Mean',
            'CO2_Level', 'Light', 'Humidity',
            'Outside_Temp_C', 'Windspeed',
            'Relative_Compactness', 'Overall_Height'
        ]

        # Hyperparameters with stronger regularization to prevent overfitting
        self.lgb_params = {
            'n_estimators': 100, 
            'max_depth': 3,           # Reduced depth
            'learning_rate': 0.05,     # Higher learning rate with fewer trees
            'reg_lambda': 20.0,        # Stronger L2 regularization
            'reg_alpha': 10.0,         # L1 regularization
            'min_child_samples': 10,   # Minimum samples per leaf
            'subsample': 0.8,          # Row sampling
            'colsample_bytree': 0.8,   # Feature sampling
            'random_state': 42, 
            'verbose': -1
        }
        self.cat_params = {
            'iterations': 100, 
            'depth': 3,                 # Reduced depth
            'learning_rate': 0.05,     # Higher learning rate
            'l2_leaf_reg': 20.0,        # Stronger regularization
            'random_state': 42, 
            'verbose': 0
        }

    def run_pipeline(self, use_enhanced_features=True):
        feature_set = "Enhanced" if use_enhanced_features else "Core"
        console_log("="*80)
        console_log(f" [START] ECOGRID AI: TRIPLE-DATASET PIPELINE ({feature_set} FEATURES)")
        console_log("="*80)

        # 1. Load Triple-Dataset Integrated Data
        console_log("\n[STEP 1/5] Loading Triple-Dataset Integrated EcoGrid Matrix...")
        
        try:
            df = pd.read_csv('ecogrid_triple_dataset_matrix.csv')
            console_log(f" -> Loaded integrated dataset: {df.shape}")
            console_log(f" -> Available columns: {df.columns.tolist()}")
        except FileNotFoundError:
            console_log(" [ERROR] ecogrid_triple_dataset_matrix.csv not found.")
            console_log("    Run ecogrid_triple_dataset_integration.py first.")
            return

        # 2. Data Preprocessing
        console_log("\n[STEP 2/5] Preprocessing Triple-Dataset Features...")
        
        # Handle missing values
        df = df.dropna(subset=['HVAC_Power_kW', 'Occupancy_Category'])
        console_log(f" -> After dropping missing values: {df.shape}")
        
        # Encode categorical variables
        df['Occupancy_Label'] = self.label_encoder.fit_transform(df['Occupancy_Category'])
        df['Efficiency_Label'] = LabelEncoder().fit_transform(df['Efficiency_Category'])
        console_log(f" -> Occupancy classes: {self.label_encoder.classes_}")
        
        # Select feature set
        if use_enhanced_features:
            self.feature_cols = self.enhanced_feature_cols
            console_log(f" -> Using enhanced feature set with {len(self.feature_cols)} features")
        else:
            self.feature_cols = self.core_feature_cols
            console_log(f" -> Using core feature set with {len(self.feature_cols)} features")
        
        # Ensure features exist
        missing_features = [col for col in self.feature_cols if col not in df.columns]
        if missing_features:
            console_log(f" [ERROR] Missing features: {missing_features}")
            return

        # 3. Enhanced Train-Test Split with stratification
        X = df[self.feature_cols]
        y_cls = df["Occupancy_Label"]
        y_reg = df["HVAC_Power_kW"]

        # Use stratified split for classification to maintain class distribution
        X_train, X_test, y_train_cls, y_test_cls = train_test_split(
            X, y_cls, test_size=0.3, stratify=y_cls, random_state=42
        )
        # Use same indices for regression
        train_indices = X_train.index
        test_indices = X_test.index
        y_train_reg = y_reg.loc[train_indices]
        y_test_reg = y_reg.loc[test_indices]

        console_log(f" -> Dataset: {len(df)} Rows | Train: {len(X_train)} | Test: {len(X_test)}")

        # 4. Enhanced Model Training
        console_log("\n[STEP 3/5] Training Enhanced Dual-Track Ensemble...")
        
        # Classification Models (Occupancy Detection)
        m1_cls = lgb.LGBMClassifier(**self.lgb_params).fit(X_train, y_train_cls)
        m2_cls = CatBoostClassifier(**self.cat_params).fit(X_train, y_train_cls)

        # Regression Models (Energy Prediction)
        m1_reg = lgb.LGBMRegressor(**self.lgb_params).fit(X_train, y_train_reg)
        m2_reg = CatBoostRegressor(**self.cat_params).fit(X_train, y_train_reg)

        # 5. Comprehensive Metrics Extraction with Cross-Validation
        console_log("\n[STEP 4/5] Evaluating Enhanced Model Performance...")
        
        # Classification Metrics
        tr_prob = (m1_cls.predict_proba(X_train) + m2_cls.predict_proba(X_train)) / 2
        te_prob = (m1_cls.predict_proba(X_test) + m2_cls.predict_proba(X_test)) / 2
        train_acc = accuracy_score(y_train_cls, np.argmax(tr_prob, axis=1))
        test_acc = accuracy_score(y_test_cls, np.argmax(te_prob, axis=1))

        # Regression Metrics
        te_pred_reg = (m1_reg.predict(X_test) + m2_reg.predict(X_test)) / 2
        tr_pred_reg = (m1_reg.predict(X_train) + m2_reg.predict(X_train)) / 2
        train_rmse = np.sqrt(mean_squared_error(y_train_reg, tr_pred_reg))
        test_rmse = np.sqrt(mean_squared_error(y_test_reg, te_pred_reg))
        train_mae = mean_absolute_error(y_train_reg, tr_pred_reg)
        test_mae = mean_absolute_error(y_test_reg, te_pred_reg)

        console_log(f" [METRICS] CLASSIFICATION (Occupancy):")
        console_log(f"    Train Accuracy: {train_acc*100:.1f}% | Test Accuracy: {test_acc*100:.1f}%")
        console_log(f" [METRICS] REGRESSION (Energy Prediction):")
        console_log(f"    Train RMSE: {train_rmse:.2f} kW | Test RMSE: {test_rmse:.2f} kW")
        console_log(f"    Train MAE:  {train_mae:.2f} kW | Test MAE:  {test_mae:.2f} kW")

        # Cross-Validation for more robust evaluation
        console_log(f" [CROSS-VALIDATION] Running 5-fold CV to detect overfitting...")
        
        # Classification CV
        cv_cls = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores_cls = cross_val_score(m1_cls, X, y_cls, cv=cv_cls, scoring='accuracy')
        console_log(f"    Classification CV Accuracy: {cv_scores_cls.mean()*100:.1f}% (+/- {cv_scores_cls.std()*100:.1f}%)")
        
        # Regression CV
        cv_reg = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores_reg = -cross_val_score(m1_reg, X, y_reg, cv=cv_reg, scoring='neg_root_mean_squared_error')
        console_log(f"    Regression CV RMSE: {cv_scores_reg.mean():.2f} kW (+/- {cv_scores_reg.std():.2f} kW)")

        # Overfitting Detection
        acc_gap = train_acc - test_acc
        rmse_gap = test_rmse - train_rmse
        console_log(f" [OVERFITTING ANALYSIS]:")
        console_log(f"    Accuracy Gap (Train-Test): {acc_gap*100:.1f}%")
        console_log(f"    RMSE Gap (Test-Train): {rmse_gap:.2f} kW")
        
        if acc_gap > 0.1 or rmse_gap > 0.5:
            console_log(f"    WARNING: Potential overfitting detected!")
        else:
            console_log(f"    OK: Model generalization appears reasonable")

        # Feature Importance Analysis
        console_log("\n[STEP 5/5] Feature Importance Analysis...")
        self.analyze_feature_importance(m1_reg, X_train.columns)

        # Live API Endpoint Demonstration (commented out due to encoding issues)
        # self.live_api_endpoint(m1_cls, m2_cls, m1_reg, m2_reg, 
        #                      hour=14, day_of_week=2, temp=34.5, temp_avg=33.8,
        #                      co2=800, light=400, humidity=45, outside_temp=32.0, windspeed=5.0)
        console_log(" [NOTE] Live API endpoint skipped due to console encoding issues")

        # Generate Enhanced Visualization
        self.plot_and_save(df, y_test_reg, te_pred_reg, use_enhanced_features)

    def analyze_feature_importance(self, model, feature_names):
        """Analyze and display feature importance."""
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            feature_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
            feature_imp = feature_imp.sort_values('Importance', ascending=False)
            
            console_log(" [FEATURES] TOP 10 FEATURE IMPORTANCE:")
            for idx, row in feature_imp.head(10).iterrows():
                console_log(f"    {row['Feature']}: {row['Importance']:.2f}")

    def live_api_endpoint(self, m1_cls, m2_cls, m1_reg, m2_reg, hour, day_of_week, temp, temp_avg,
                         co2, light, humidity, outside_temp, windspeed):
        """Enhanced live API endpoint with triple-dataset features."""
        console_log("\n[ENHANCED API ENDPOINT] Real-time EcoGrid AI Prediction...")
        
        hour_sin = np.sin(2 * np.pi * hour / 24.0)
        hour_cos = np.cos(2 * np.pi * hour / 24.0)
        is_weekend = 1 if day_of_week >= 5 else 0

        # Check if using enhanced features
        if len(self.feature_cols) > 6:  # Enhanced features
            payload = pd.DataFrame([[
                hour_sin, hour_cos, day_of_week, is_weekend, temp, temp_avg,
                co2, light, humidity, outside_temp, windspeed, 0.75, 3.5  # Default building values
            ]], columns=self.feature_cols)
        else:  # Core features
            payload = pd.DataFrame([[hour_sin, hour_cos, day_of_week, is_weekend, temp, temp_avg]], 
                                 columns=self.feature_cols)

        voted_probs = (m1_cls.predict_proba(payload) + m2_cls.predict_proba(payload)) / 2
        assigned_state = self.label_encoder.classes_[np.argmax(voted_probs, axis=1)[0]]

        blended_pred = (m1_reg.predict(payload)[0] + m2_reg.predict(payload)[0]) / 2

        console_log(f" -> INCOMING TELEMETRY:")
        console_log(f"    Time: {hour}:00 | Day: {day_of_week} | Temp: {temp}°C")
        console_log(f"    CO2: {co2} ppm | Light: {light} lux | Humidity: {humidity}%")
        console_log(f"    Outside Temp: {outside_temp} C | Windspeed: {windspeed} m/s")
        console_log(f" -> SPATIAL MODEL STATE: [{assigned_state.upper()}] Occupancy")
        console_log(f" -> KINETIC FORECAST:    [{blended_pred:.2f} kW] Electrical Load")

        # Enhanced routing logic with building efficiency consideration
        if assigned_state == "High" and blended_pred > 50.0:
            console_log(" �️ ROUTING ACTION: PRE-COOLING ENGAGED + EFFICIENCY OPTIMIZATION")
        elif assigned_state == "Low":
            console_log(" [ACTION] DEEP HIBERNATION MODE + MINIMAL VENTILATION")
        else:
            console_log(" [ACTION] ADAPTIVE COMFORT MODE + ENERGY RECOVERY")
        console_log("="*80)

    def plot_and_save(self, df, y_test_reg, te_pred_reg, use_enhanced_features):
        """Generate enhanced visualization."""
        plt.figure(figsize=(14, 6))
        
        # Actual vs Predicted
        plt.subplot(1, 2, 1)
        plt.plot(df["hourly_timestamp"].iloc[-len(y_test_reg):].values, y_test_reg.values, 
                label="Actual Load (kW)", color="steelblue", linewidth=2)
        plt.plot(df["hourly_timestamp"].iloc[-len(y_test_reg):].values, te_pred_reg, 
                label="Predicted Load (kW)", color="darkorange", linewidth=2, linestyle="--")
        plt.title("EcoGrid AI: Actual vs Predicted HVAC Power", fontsize=12, fontweight="bold")
        plt.xlabel("Timestamp", fontsize=10)
        plt.ylabel("Power (kW)", fontsize=10)
        plt.legend(fontsize=9)
        plt.xticks(rotation=45)

        # Residual Analysis
        plt.subplot(1, 2, 2)
        residuals = y_test_reg.values - te_pred_reg
        plt.scatter(te_pred_reg, residuals, alpha=0.5, color='purple')
        plt.axhline(y=0, color='red', linestyle='--')
        plt.title("Residual Analysis", fontsize=12, fontweight="bold")
        plt.xlabel("Predicted Values (kW)", fontsize=10)
        plt.ylabel("Residuals (kW)", fontsize=10)
        
        feature_suffix = "Enhanced" if use_enhanced_features else "Core"
        plt.tight_layout()
        plt.savefig(f"ecogrid_triple_dataset_predictions_{feature_suffix.lower()}.png", dpi=150)
        console_log(f" [OUTPUT] Enhanced prediction plot saved: ecogrid_triple_dataset_predictions_{feature_suffix.lower()}.png")

# Run the Triple-Dataset EcoGrid AI pipeline
if __name__ == "__main__":
    engine = EcoGridTripleEngine()
    
    print("\n" + "="*80)
    print("ECOGRID AI: TRIPLE-DATASET ENHANCED PIPELINE")
    print("="*80)
    print("Choose feature set:")
    print("1. Enhanced Features (Recommended) - Uses all triple dataset features")
    print("2. Core Features - Uses basic EcoGrid features only")
    print("="*80)
    
    # Run with enhanced features by default
    engine.run_pipeline(use_enhanced_features=True)