
import joblib
import pandas as pd
import numpy as np
import os

def inspect_model(path, name):
    full_path = os.path.join("models", path)
    if not os.path.exists(full_path):
        print(f"Model {name} not found at {full_path}")
        return

    print(f"\n=== Inspecting {name} ===")
    try:
        model = joblib.load(full_path)
        print(f"Type: {type(model)}")
        
        # specific checks
        if hasattr(model, "feature_names_in_"):
            print(f"Features: {model.feature_names_in_}")
        elif hasattr(model, "n_features_in_"):
            print(f"Expected {model.n_features_in_} features (names not stored)")
            
        if hasattr(model, "coef_"):
            print("Coefficients:")
            if hasattr(model, "feature_names_in_"):
                for f, c in zip(model.feature_names_in_, model.coef_):
                    print(f"  {f}: {c:.4f}")
            else:
                print(f"  {model.coef_}")
        
        if hasattr(model, "feature_importances_"):
             print("Feature Importances:")
             if hasattr(model, "feature_names_in_"):
                for f, c in zip(model.feature_names_in_, model.feature_importances_):
                    print(f"  {f}: {c:.4f}")
             else:
                print(f"  {model.feature_importances_}")

    except Exception as e:
        print(f"Error loading {name}: {e}")

if __name__ == "__main__":
    inspect_model("rainfall/best_model.joblib", "Rainfall Model")
    inspect_model("drought/best_model.joblib", "Drought Model")
    inspect_model("heatwave/best_model.joblib", "Heatwave Model")
