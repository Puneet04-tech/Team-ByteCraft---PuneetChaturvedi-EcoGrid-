"""
ECOGRID AI: TRIPLE-DATASET PIPELINE RUNNER
Execute the complete triple-dataset integration and AI training
"""

import subprocess
import sys

print("="*80)
print("ECOGRID AI: TRIPLE-DATASET PIPELINE RUNNER")
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

print("\n[STEP 2/2] Running Triple-Dataset AI Engine...")
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
print("- ecogrid_triple_dataset_predictions_enhanced.png (Visualizations)")
print("="*80)