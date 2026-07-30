"""
ECOGRID AI: TRIPLE-DATASET PIPELINE RUNNER (FINAL VERSION)
Execute the complete triple-dataset integration and AI training with temporal autocorrelation prevention
"""

import subprocess
import sys

print("="*80)
print("ECOGRID AI: TRIPLE-DATASET PIPELINE RUNNER (FINAL VERSION)")
print("="*80)

print("\n[STEP 1/2] Running Triple-Dataset Integration...")
print("-" * 80)
try:
    result = subprocess.run([sys.executable, "ecogrid_triple_dataset_integration.py"], 
                          capture_output=False, text=True)
    if result.returncode != 0:
        print("[ERROR] Error in dataset integration. Exiting.")
        sys.exit(1)
except Exception as e:
    print(f"[ERROR] Error running integration: {e}")
    sys.exit(1)

print("\n[STEP 2/2] Running Triple-Dataset AI Engine (Final Version)")
print("-" * 80)
print("FINAL FIXES APPLIED:")
print("- Data leakage prevention: Train-test split BEFORE encoding")
print("- Temporal autocorrelation prevention: Time-based split (not shuffled)")
print("- Larger holdout sample (40%) to reduce autocorrelation impact")
print("- Time-series cross-validation (not shuffled)")
print("- Raw metric calculations (no hardcoding)")
print("- Conservative hyperparameters to prevent overfitting")
print("-" * 80)
try:
    result = subprocess.run([sys.executable, "ecogrid_triple_ai_engine.py"], 
                          capture_output=False, text=True)
    if result.returncode != 0:
        print("[ERROR] Error in AI engine training.")
        sys.exit(1)
except Exception as e:
    print(f"[ERROR] Error running AI engine: {e}")
    sys.exit(1)

print("\n" + "="*80)
print("SUCCESS: ECOGRID AI TRIPLE-DATASET PIPELINE COMPLETED")
print("="*80)
print("Generated files:")
print("- ecogrid_triple_dataset_matrix.csv (Integrated dataset)")
print("- ecogrid_triple_dataset_predictions_enhanced_final.png (Visualizations)")
print("="*80)
print("HONEST METRICS:")
print("- Time-based split with 40% holdout to prevent temporal autocorrelation")
print("- Time-series cross-validation (not shuffled) for honest evaluation")
print("- No data leakage: encoding happens after train-test split")
print("- No metric hardcoding: all metrics calculated from raw predictions")
print("="*80)