import sys
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Add parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.data_loader import load_heatwave_data

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
        print("EXPERIMENT 3: HEATWAVE PREDICTION (Classification)")
        print("=" * 60)

        # 1. Load Data
        print("\n[Data] Loading and splitting data...")
        X, y = load_heatwave_data()
        # Stratified split for classification
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        print(f"   Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        print(f"   Heatwave Ratio: {y.mean():.2%}")

        # 2. Define Models
        models = {
            "Logistic Regression": {
                "model": Pipeline([
                    ("scaler", StandardScaler()),
                    ("classifier", LogisticRegression(max_iter=1000, class_weight='balanced'))
                ]),
                "params": {
                    "classifier__C": [0.1, 1.0, 10.0]
                }
            },
            "Random Forest": {
                "model": RandomForestClassifier(random_state=42, class_weight='balanced'),
                "params": {
                    "n_estimators": [50, 100],
                    "max_depth": [5, 10, None]
                }
            },
            "Gradient Boosting": {
                "model": GradientBoostingClassifier(random_state=42),
                "params": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.05, 0.1],
                    "max_depth": [3, 5]
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
            
            search = GridSearchCV(config["model"], config["params"], cv=3, scoring='f1', n_jobs=-1)
            search.fit(X_train, y_train)
            model = search.best_estimator_
            print(f"     Best Params: {search.best_params_}")

            # Evaluate
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            print(f"     Accuracy: {acc:.4f} | F1: {f1:.4f}")
            
            results.append({
                "Model": name,
                "Accuracy": acc,
                "F1": f1,
                "Model Object": model
            })
            
            if f1 > best_score:
                best_score = f1
                best_model = model
                best_name = name

        # 4. Save WINNER
        print("\n" + "="*60)
        print(f"WINNER: {best_name} (F1 Score: {best_score:.4f})")
        print("="*60)

        joblib.dump(best_model, os.path.join(output_dir, "best_model.joblib"))
        joblib.dump(X.columns.tolist(), os.path.join(output_dir, "features.joblib"))
        
        # 5. Visualization
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix, roc_curve, auc

        # Ensure plots directory exists in Climate_AI/static/plots
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        plots_dir = os.path.join(root_dir, "static", "plots")
        os.makedirs(plots_dir, exist_ok=True)

        y_pred_best = best_model.predict(X_test)
        
        # Plot 1: Confusion Matrix
        cm = confusion_matrix(y_test, y_pred_best)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', cbar=False)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Heatwave Confusion Matrix ({best_name})")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "heatwave_confusion_matrix.png"))
        plt.close()

        # Plot 2: ROC Curve
        if hasattr(best_model, "predict_proba"):
            y_prob = best_model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve ({best_name})')
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "heatwave_roc_curve.png"))
            plt.close()

        print(f"Plots saved to {plots_dir}")
        
        # 6. Generate Report
        # EXPERIMENT_REPORT (Consolidated)
        exp_report_path = os.path.join(output_dir, "EXPERIMENT_REPORT.md")
        with open(exp_report_path, "w") as f:
            f.write("# Experiment Report: Heatwave Prediction\n\n")
            f.write(f"**Best Model:** {best_name}\n")
            f.write(f"**Test F1 Score:** {best_score:.4f}\n\n")

            f.write("## 1. Overview\n")
            f.write("This model classifies days as 'Heatwave' or 'Normal' based on previous days' temperatures and humidity.\n\n")

            f.write("## 2. Methodology\n")
            f.write("- **Data Source:** Open-Meteo Historical Weather API (Hyderabad).\n")
            f.write("- **Definition:** Max Temp > 40°C or >4.5°C above normal.\n")
            f.write("- **Features:** 3-day temperature lags, humidity, and calendar month.\n\n")

            f.write("## 3. Visual Performance Benchmarks\n")
            f.write("### Confusion Matrix\n")
            f.write("![Confusion Matrix](/static/plots/heatwave_confusion_matrix.png)\n\n")
            if hasattr(best_model, "predict_proba"):
                f.write("### ROC Curve\n")
                f.write("![ROC Curve](/static/plots/heatwave_roc_curve.png)\n\n")

            f.write("## 4. Comparison Metrics\n")
            f.write("| Model | Accuracy | F1 Score |\n")
            f.write("|-------|----------|----------|\n")
            for res in results:
                f.write(f"| {res['Model']} | {res['Accuracy']:.4f} | {res['F1']:.4f} |\n")
        
        print(f"Report saved to {output_dir}")

    finally:
        sys.stdout = sys.stdout.terminal

if __name__ == "__main__":
    run_experiment()

if __name__ == "__main__":
    run_experiment()
