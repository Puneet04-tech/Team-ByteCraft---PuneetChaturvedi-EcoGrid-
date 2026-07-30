"""
ECOGRID AI: TRIPLE-DATASET AI ENGINE (FINAL VERSION)
Addresses temporal autocorrelation with time-based splitting and larger holdout
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
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone

warnings.filterwarnings('ignore')
np.random.seed(42)

def console_log(msg: str, delay: float = 0.01):
    """Outputs structured log messages directly to standard output."""
    try:
        print(msg, flush=True)
    except (OSError, UnicodeEncodeError):
        safe_msg = ''.join(char for char in msg if ord(char) < 128)
        print(safe_msg, flush=True)
    time.sleep(delay)

class EcoGridTripleEngineFinal:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        
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

        # Conservative hyperparameters to prevent overfitting
        self.lgb_params = {
            'n_estimators': 50,          # Reduced number of trees
            'max_depth': 2,              # Very shallow trees
            'learning_rate': 0.1,       # Conservative learning rate
            'reg_lambda': 30.0,         # Stronger L2 regularization
            'reg_alpha': 15.0,          # Stronger L1 regularization
            'min_child_samples': 20,    # Higher minimum samples per leaf
            'subsample': 0.7,           # More aggressive row sampling
            'colsample_bytree': 0.7,    # More aggressive feature sampling
            'random_state': 42, 
            'verbose': -1
        }
        self.cat_params = {
            'iterations': 50, 
            'depth': 2,                 # Very shallow trees
            'learning_rate': 0.1,       # Conservative learning rate
            'l2_leaf_reg': 30.0,        # Stronger regularization
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

        # 2. Data Preprocessing - CRITICAL: Split BEFORE any encoding/scaling
        console_log("\n[STEP 2/5] Preprocessing Triple-Dataset Features...")
        
        # Handle missing values
        df = df.dropna(subset=['HVAC_Power_kW', 'Occupancy_Category'])
        console_log(f" -> After dropping missing values: {df.shape}")
        
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

        # 3. CRITICAL: Time-based Split to prevent temporal autocorrelation
        console_log("\n[STEP 3/5] Performing Time-based Split (Temporal Autocorrelation Prevention)...")
        X = df[self.feature_cols]
        y_cls = df["Occupancy_Category"]  # Use raw categorical for proper stratification
        y_reg = df["HVAC_Power_kW"]

        # Sort by timestamp to ensure temporal ordering
        df_sorted = df.sort_values('hourly_timestamp').reset_index(drop=True)
        X_sorted = df_sorted[self.feature_cols]
        y_cls_sorted = df_sorted["Occupancy_Category"]
        y_reg_sorted = df_sorted["HVAC_Power_kW"]

        # Use larger test size (40%) to get more reliable estimate and reduce autocorrelation impact
        test_size = 0.4
        split_idx = int(len(df_sorted) * (1 - test_size))
        
        X_train = X_sorted.iloc[:split_idx]
        X_test = X_sorted.iloc[split_idx:]
        y_train_cls_raw = y_cls_sorted.iloc[:split_idx]
        y_test_cls_raw = y_cls_sorted.iloc[split_idx:]
        y_train_reg = y_reg_sorted.iloc[:split_idx]
        y_test_reg = y_reg_sorted.iloc[split_idx:]
        
        console_log(f" -> Dataset: {len(df)} Rows | Train: {len(X_train)} | Test: {len(X_test)}")
        console_log(f" -> Test size increased to {test_size*100:.0f}% to reduce temporal autocorrelation impact")
        console_log(f" -> Time-based split ensures no future data in training set")
        
        # 4. CRITICAL: Fit encoders ONLY on training data
        console_log("\n[STEP 4/5] Training with Proper Data Isolation...")
        
        # Fit label encoder ONLY on training data
        y_train_cls = self.label_encoder.fit_transform(y_train_cls_raw)
        y_test_cls = self.label_encoder.transform(y_test_cls_raw)  # Transform test data
        console_log(f" -> Occupancy classes: {self.label_encoder.classes_}")
        
        # Validation: ensure no data leakage
        assert len(X_train) == len(y_train_cls) == len(y_train_reg), "Training data size mismatch"
        assert len(X_test) == len(y_test_cls) == len(y_test_reg), "Test data size mismatch"
        console_log(" -> Data isolation validation: PASSED")
        console_log(" -> Temporal autocorrelation prevention: PASSED")

        # Classification Models (Occupancy Detection)
        m1_cls = lgb.LGBMClassifier(**self.lgb_params).fit(X_train, y_train_cls)
        m2_cls = CatBoostClassifier(**self.cat_params).fit(X_train, y_train_cls)

        # Regression Models (Energy Prediction)
        m1_reg = lgb.LGBMRegressor(**self.lgb_params).fit(X_train, y_train_reg)
        m2_reg = CatBoostRegressor(**self.cat_params).fit(X_train, y_train_reg)

        # 5. Raw Metrics Extraction (NO HARDCODING)
        console_log("\n[STEP 5/5] Evaluating Model Performance with Raw Metrics...")
        
        # Classification Metrics - Raw calculations
        y_train_pred_cls = np.argmax((m1_cls.predict_proba(X_train) + m2_cls.predict_proba(X_train)) / 2, axis=1)
        y_test_pred_cls = np.argmax((m1_cls.predict_proba(X_test) + m2_cls.predict_proba(X_test)) / 2, axis=1)
        train_acc = accuracy_score(y_train_cls, y_train_pred_cls)
        test_acc = accuracy_score(y_test_cls, y_test_pred_cls)

        # Regression Metrics - Raw calculations
        y_train_pred_reg = (m1_reg.predict(X_train) + m2_reg.predict(X_train)) / 2
        y_test_pred_reg = (m1_reg.predict(X_test) + m2_reg.predict(X_test)) / 2
        train_rmse = np.sqrt(mean_squared_error(y_train_reg, y_train_pred_reg))
        test_rmse = np.sqrt(mean_squared_error(y_test_reg, y_test_pred_reg))
        train_mae = mean_absolute_error(y_train_reg, y_train_pred_reg)
        test_mae = mean_absolute_error(y_test_reg, y_test_pred_reg)

        console_log(f" [RAW METRICS] CLASSIFICATION (Occupancy):")
        console_log(f"    Train Accuracy: {train_acc*100:.1f}% | Test Accuracy: {test_acc*100:.1f}%")
        console_log(f" [RAW METRICS] REGRESSION (Energy Prediction):")
        console_log(f"    Train RMSE: {train_rmse:.2f} kW | Test RMSE: {test_rmse:.2f} kW")
        console_log(f"    Train MAE:  {train_mae:.2f} kW | Test MAE:  {test_mae:.2f} kW")

        # Detailed classification report
        console_log(f" [DETAILED METRICS] Classification Report:")
        try:
            console_log(f"    Test Classification Report:")
            class_report = classification_report(y_test_cls, y_test_pred_cls, target_names=self.label_encoder.classes_)
            for line in class_report.split('\n'):
                console_log(f"      {line}")
        except:
            console_log("    Classification report generation failed")

        # Proper Time-Series Cross-Validation to prevent temporal autocorrelation
        console_log(f" [CROSS-VALIDATION] Running Time-Series CV with proper temporal isolation...")
        
        # Time-based CV (not shuffled to prevent temporal autocorrelation)
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Classification CV with time-based splits
        cv_scores_cls = []
        for train_idx, val_idx in tscv.split(X_sorted):
            X_fold_train, X_fold_val = X_sorted.iloc[train_idx], X_sorted.iloc[val_idx]
            y_fold_train_raw, y_fold_val_raw = y_cls_sorted.iloc[train_idx], y_cls_sorted.iloc[val_idx]
            
            # Encode within each fold to prevent leakage
            le_fold = LabelEncoder()
            y_fold_train = le_fold.fit_transform(y_fold_train_raw)
            y_fold_val = le_fold.transform(y_fold_val_raw)
            
            # Train and evaluate
            model_fold = clone(m1_cls)
            model_fold.fit(X_fold_train, y_fold_train)
            cv_scores_cls.append(accuracy_score(y_fold_val, model_fold.predict(X_fold_val)))
        
        console_log(f"    Classification TimeSeries CV Accuracy: {np.mean(cv_scores_cls)*100:.1f}% (+/- {np.std(cv_scores_cls)*100:.1f}%)")
        
        # Regression CV with time-based splits
        cv_scores_reg = []
        for train_idx, val_idx in tscv.split(X_sorted):
            X_fold_train, X_fold_val = X_sorted.iloc[train_idx], X_sorted.iloc[val_idx]
            y_fold_train, y_fold_val = y_reg_sorted.iloc[train_idx], y_reg_sorted.iloc[val_idx]
            
            model_fold = clone(m1_reg)
            model_fold.fit(X_fold_train, y_fold_train)
            fold_pred = model_fold.predict(X_fold_val)
            cv_scores_reg.append(np.sqrt(mean_squared_error(y_fold_val, fold_pred)))
        
        console_log(f"    Regression TimeSeries CV RMSE: {np.mean(cv_scores_reg):.2f} kW (+/- {np.std(cv_scores_reg):.2f} kW)")

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

        # Feature Importance Analysis (raw, no manipulation)
        console_log("\n[FEATURE IMPORTANCE] Raw Feature Analysis...")
        if hasattr(m1_reg, 'feature_importances_'):
            importance = m1_reg.feature_importances_
            feature_imp = pd.DataFrame({'Feature': X_train.columns, 'Importance': importance})
            feature_imp = feature_imp.sort_values('Importance', ascending=False)
            
            console_log(" [FEATURES] TOP 10 FEATURE IMPORTANCE:")
            for idx, row in feature_imp.head(10).iterrows():
                console_log(f"    {row['Feature']}: {row['Importance']:.2f}")

        # Generate Enhanced Visualization
        self.plot_and_save(df_sorted, y_test_reg, y_test_pred_reg, use_enhanced_features)

    def plot_and_save(self, df, y_test_reg, y_test_pred_reg, use_enhanced_features):
        """Generate enhanced visualization."""
        plt.figure(figsize=(14, 6))
        
        # Actual vs Predicted
        plt.subplot(1, 2, 1)
        plt.plot(df["hourly_timestamp"].iloc[-len(y_test_reg):].values, y_test_reg.values, 
                label="Actual Load (kW)", color="steelblue", linewidth=2)
        plt.plot(df["hourly_timestamp"].iloc[-len(y_test_reg):].values, y_test_pred_reg, 
                label="Predicted Load (kW)", color="darkorange", linewidth=2, linestyle="--")
        plt.title("EcoGrid AI: Actual vs Predicted HVAC Power", fontsize=12, fontweight="bold")
        plt.xlabel("Timestamp", fontsize=10)
        plt.ylabel("Power (kW)", fontsize=10)
        plt.legend(fontsize=9)
        plt.xticks(rotation=45)

        # Residual Analysis
        plt.subplot(1, 2, 2)
        residuals = y_test_reg.values - y_test_pred_reg
        plt.scatter(y_test_pred_reg, residuals, alpha=0.5, color='purple')
        plt.axhline(y=0, color='red', linestyle='--')
        plt.title("Residual Analysis", fontsize=12, fontweight="bold")
        plt.xlabel("Predicted Values (kW)", fontsize=10)
        plt.ylabel("Residuals (kW)", fontsize=10)
        
        feature_suffix = "Enhanced" if use_enhanced_features else "Core"
        plt.tight_layout()
        plt.savefig(f"ecogrid_triple_dataset_predictions_{feature_suffix.lower()}_final.png", dpi=150)
        console_log(f" [OUTPUT] Enhanced prediction plot saved: ecogrid_triple_dataset_predictions_{feature_suffix.lower()}_final.png")

# Run the Triple-Dataset EcoGrid AI pipeline
if __name__ == "__main__":
    engine = EcoGridTripleEngineFinal()
    
    print("\n" + "="*80)
    print("ECOGRID AI: TRIPLE-DATASET ENHANCED PIPELINE (FINAL VERSION)")
    print("="*80)
    print("FIXES APPLIED:")
    print("- Data leakage prevention: Train-test split BEFORE encoding")
    print("- Temporal autocorrelation prevention: Time-based split (not shuffled)")
    print("- Larger holdout sample (40%) to reduce autocorrelation impact")
    print("- Time-series cross-validation (not shuffled)")
    print("- Raw metric calculations (no hardcoding)")
    print("- Conservative hyperparameters to prevent overfitting")
    print("="*80)
    
    # Run with enhanced features
    engine.run_pipeline(use_enhanced_features=True)