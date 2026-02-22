# Climate AI POC - Project Structure

The following is the required folder structure for the final working application.

```
Climate_AI/
├── app.py                      # Main FastAPI application (Deployment Ready)
├── requirements.txt            # Python dependencies
├── PROJECT_DOCUMENTATION.md    # Main project overview
├── tasks.md                    # Tracker
├── implementation_plan.md      # Plan
├── walkthrough.md              # Changes walkthrough
│
├── data/
│   ├── hyderabad_rainfall_data.csv  # IMD Dataset
│   ├── hyderabad_temperature.csv    # Open-Meteo Dataset
│   └── crop_yield_india.csv         # Kaggle Dataset
│
├── utils/
│   └── data_loader.py          # Centralized data loading logic
│
├── models/
│   ├── rainfall/
│   │   ├── experiment_rainfall.py   # Training Experiment
│   │   ├── best_model.joblib        # Saved Model
│   │   ├── features.joblib          # Feature names
│   │   ├── EXPERIMENT_REPORT.md     # Experiment Results
│   │   └── TECHNICAL_REPORT.md      # Leadership Doc
│   │
│   ├── drought/
│   │   ├── experiment_drought.py
│   │   ├── best_model.joblib
│   │   ├── features.joblib
│   │   ├── EXPERIMENT_REPORT.md
│   │   └── TECHNICAL_REPORT.md
│   │
│   ├── heatwave/
│   │   ├── experiment_heatwave.py
│   │   ├── best_model.joblib
│   │   ├── features.joblib
│   │   ├── EXPERIMENT_REPORT.md
│   │   └── TECHNICAL_REPORT.md
│   │
│   └── crop_impact/
│       ├── experiment_crop.py
│       ├── best_model.joblib
│       ├── (encoders).joblib        # crop/state/season encoders
│       ├── EXPERIMENT_REPORT.md
│       └── TECHNICAL_REPORT.md
│
├── static/                     # Frontend Assets
│   ├── index.html
│   ├── styles.css
│   └── app.js
│
└── kml_files/                  # Geo-visuals
    ├── (various .kml files)
```
