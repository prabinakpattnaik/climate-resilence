import sys
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.data_loader import load_drought_data

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def run_experiment():
    output_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(output_dir, "training_log.txt")
    sys.stdout = Logger(log_path)

    try:
        print("=" * 60)
        print("EXPERIMENT 2: DROUGHT PREDICTION (Regression)")
        print("=" * 60)

        # 1. Load Data
        print("\n[Data] Loading and splitting data...")
        X, y = load_drought_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"   Train shape: {X_train.shape}, Test shape: {X_test.shape}")

        # 2. Define Models
        models = {
            "Ridge Regression": {
                "model": Pipeline([
                    ("scaler", StandardScaler()),
                    ("regressor", Ridge())
                ]),
                "params": {
                    "regressor__alpha": [0.1, 1.0, 10.0]
                }
            },
            "SVR": {
                "model": Pipeline([
                    ("scaler", StandardScaler()),
                    ("regressor", SVR())
                ]),
                "params": {
                    "regressor__C": [0.1, 1, 10],
                    "regressor__kernel": ["rbf", "linear"]
                }
            },
            "Random Forest": {
                "model": Pipeline([
                    ("scaler", StandardScaler()),
                    ("regressor", RandomForestRegressor(random_state=42))
                ]),
                "params": {
                    "regressor__n_estimators": [50, 100],
                    "regressor__max_depth": [5, 10, None]
                }
            }
        }

        results = []
        best_model = None
        best_score = -float("inf")
        best_name = ""

        # 3. Train and Evaluate
        print("\n[Experiment] Training candidate models...")
        for name, config in models.items():
            print(f"   > Training {name}...")
            
            search = GridSearchCV(config["model"], config["params"], cv=3, scoring='r2', n_jobs=-1)
            search.fit(X_train, y_train)
            model = search.best_estimator_
            print(f"     Best Params: {search.best_params_}")

            # Evaluate
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            print(f"     R2: {r2:.4f} | RMSE: {rmse:.2f}")
            
            results.append({
                "Model": name,
                "R2": r2,
                "RMSE": rmse,
                "MAE": mae,
                "Model Object": model
            })
            
            if r2 > best_score:
                best_score = r2
                best_model = model
                best_name = name

        # 4. Save WINNER
        print("\n" + "="*60)
        print(f"WINNER: {best_name} (R2: {best_score:.4f})")
        print("="*60)

        joblib.dump(best_model, os.path.join(output_dir, "best_model.joblib"))
        joblib.dump(X.columns.tolist(), os.path.join(output_dir, "features.joblib"))
        
        # 5. Visualization
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Ensure plots directory exists in Climate_AI/static/plots
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        plots_dir = os.path.join(root_dir, "static", "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # Re-predict with best model for plotting
        y_pred_best = best_model.predict(X_test)
        
        # Plot 1: Actual vs Predicted
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred_best, alpha=0.5, color='orange')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel("Actual Drought Score (0-100)")
        plt.ylabel("Predicted Drought Score (0-100)")
        plt.title(f"Drought Prediction: Actual vs Predicted ({best_name})")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "drought_actual_vs_pred.png"))
        plt.close()

        # Plot 2: Residuals
        residuals = y_test - y_pred_best
        plt.figure(figsize=(10, 6))
        sns.histplot(residuals, kde=True, color='orange')
        plt.xlabel("Residuals")
        plt.title(f"Residual Distribution ({best_name})")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "drought_residuals.png"))
        plt.close()

        print(f"Plots saved to {plots_dir}")

        # 6. Generate Report
        # EXPERIMENT_REPORT (Consolidated)
        exp_report_path = os.path.join(output_dir, "EXPERIMENT_REPORT.md")
        with open(exp_report_path, "w") as f:
            f.write("# Experiment Report: Drought Prediction\n\n")
            f.write(f"**Best Model:** {best_name}\n")
            f.write(f"**Test R2 Score:** {best_score:.4f}\n\n")

            f.write("## 1. Overview\n")
            f.write("This model calculates a drought severity score (0-100) based on rainfall deficits and monsoon strength.\n\n")

            f.write("## 2. Methodology\n")
            f.write("- **Data Source:** IMD Rainfall Records (Historical).\n")
            f.write("- **Features:** 3/6 month rolling averages, rainfall deficit %, and previous year's drought status.\n\n")

            f.write("## 3. Visual Performance Benchmarks\n")
            f.write("### Actual vs Predicted\n")
            f.write("![Actual vs Predicted](/static/plots/drought_actual_vs_pred.png)\n\n")
            f.write("### Error Distribution (Residuals)\n")
            f.write("![Residual Distribution](/static/plots/drought_residuals.png)\n\n")

            f.write("## 4. Comparison Metrics\n")
            f.write("| Model | R2 Score | RMSE | MAE |\n")
            f.write("|-------|----------|------|-----|\n")
            for res in results:
                f.write(f"| {res['Model']} | {res['R2']:.4f} | {res['RMSE']:.2f} | {res['MAE']:.2f} |\n")
        
        print(f"Report saved to {output_dir}")

    finally:
        sys.stdout = sys.stdout.terminal

if __name__ == "__main__":
    run_experiment()
