import sys
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.data_loader import load_crop_data

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
        print("EXPERIMENT 4: CROP YIELD IMPACT (Regression)")
        print("  v2: Added XGBoost candidate")
        print("=" * 60)

        # 1. Load Data
        print("\n[Data] Loading and splitting data...")
        X, y, encoders = load_crop_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"   Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        print(f"   Features: {X.columns.tolist()}")

        # 2. Define Models
        models = {
            "Lasso": {
                "model": Pipeline([
                    ("scaler", StandardScaler()),
                    ("regressor", Lasso(random_state=42))
                ]),
                "params": {
                    "regressor__alpha": [0.01, 0.1, 1.0]
                }
            },
            "Decision Tree": {
                "model": Pipeline([
                    ("scaler", StandardScaler()),
                    ("regressor", DecisionTreeRegressor(random_state=42))
                ]),
                "params": {
                    "regressor__max_depth": [5, 10, None],
                    "regressor__min_samples_leaf": [2, 5]
                }
            },
            "Random Forest": {
                "model": Pipeline([
                    ("scaler", StandardScaler()),
                    ("regressor", RandomForestRegressor(random_state=42))
                ]),
                "params": {
                    "regressor__n_estimators": [50, 100],
                    "regressor__max_depth": [10, 20]
                }
            },
            "XGBoost": {
                "model": XGBRegressor(random_state=42, verbosity=0, tree_method='hist'),
                "params": {
                    "n_estimators": [100, 200, 300],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "max_depth": [3, 5, 7],
                    "subsample": [0.8, 1.0],
                    "colsample_bytree": [0.8, 1.0],
                    "reg_alpha": [0, 0.1],
                    "reg_lambda": [1, 2]
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

            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)

            print(f"     R2: {r2:.4f} | RMSE: {rmse:.2f} | MAE: {mae:.2f}")

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

        # Save encoders
        for name, encoder in encoders.items():
            joblib.dump(encoder, os.path.join(output_dir, f"{name}_encoder.joblib"))

        # 5. Visualization
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns

        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        plots_dir = os.path.join(root_dir, "static", "plots")
        os.makedirs(plots_dir, exist_ok=True)

        y_pred_best = best_model.predict(X_test)

        # Plot 1: Actual vs Predicted
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred_best, alpha=0.5, color='green')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel("Actual Yield Deviation (%)")
        plt.ylabel("Predicted Yield Deviation (%)")
        plt.title(f"Crop Yield Impact: Actual vs Predicted ({best_name})")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "crop_actual_vs_pred.png"))
        plt.close()

        # Plot 2: Residuals
        residuals = y_test - y_pred_best
        plt.figure(figsize=(10, 6))
        sns.histplot(residuals, kde=True, color='green')
        plt.xlabel("Residuals")
        plt.title(f"Residual Distribution ({best_name})")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "crop_residuals.png"))
        plt.close()

        # Plot 3: Feature Importance
        if hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
            feat_names = X.columns.tolist()
        elif hasattr(best_model, 'named_steps') and hasattr(best_model.named_steps.get('regressor', None), 'feature_importances_'):
            importances = best_model.named_steps['regressor'].feature_importances_
            feat_names = X.columns.tolist()
        else:
            importances = None

        if importances is not None:
            sorted_idx = np.argsort(importances)
            plt.figure(figsize=(10, 6))
            plt.barh([feat_names[i] for i in sorted_idx], importances[sorted_idx], color='forestgreen')
            plt.xlabel("Feature Importance")
            plt.title(f"Feature Importance ({best_name})")
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "crop_feature_importance.png"))
            plt.close()
            print("Feature importance plot saved.")

        print(f"Plots saved to {plots_dir}")

        # 6. Generate Report
        exp_report_path = os.path.join(output_dir, "EXPERIMENT_REPORT.md")
        with open(exp_report_path, "w") as f:
            f.write("# Experiment Report: Crop Yield Impact (v2)\n\n")
            f.write(f"**Best Model:** {best_name}\n")
            f.write(f"**Test R2 Score:** {best_score:.4f}\n\n")

            f.write("## 1. Overview\n")
            f.write("This model predicts how climate anomalies (rainfall/temperature) impact crop yields in India.\n")
            f.write("**v2 upgrade:** Added XGBoost with regularization as a candidate model.\n\n")

            f.write("## 2. Methodology\n")
            f.write("- **Data Source:** Kaggle India Crop Yield Dataset (1997-2020).\n")
            f.write("- **Features:** Rainfall, Rainfall Anomaly, Fertilizer usage, Pesticide usage, Crop Type, State, Season.\n")
            f.write("- **Models:** Lasso, Decision Tree, Random Forest, XGBoost.\n\n")

            f.write("## 3. Visual Performance Benchmarks\n")
            f.write("### Actual vs Predicted\n")
            f.write("![Actual vs Predicted](/static/plots/crop_actual_vs_pred.png)\n\n")
            f.write("### Error Distribution (Residuals)\n")
            f.write("![Residual Distribution](/static/plots/crop_residuals.png)\n\n")
            if importances is not None:
                f.write("### Feature Importance\n")
                f.write("![Feature Importance](/static/plots/crop_feature_importance.png)\n\n")

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
