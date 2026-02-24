import joblib
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def test_load(path, name):
    full_path = os.path.join(BASE_DIR, path)
    print(f"Testing {name} at {full_path}...")
    try:
        model = joblib.load(full_path)
        print(f"✓ {name} loaded successfully.")
    except Exception as e:
        print(f"✗ {name} failed to load.")
        import traceback
        traceback.print_exc()

test_load("models/rainfall/best_model.joblib", "Rainfall Model")
test_load("models/drought/best_model.joblib", "Drought Model")
test_load("models/heatwave/best_model.joblib", "Heatwave Model")
test_load("models/crop_impact/best_model.joblib", "Crop Model")
