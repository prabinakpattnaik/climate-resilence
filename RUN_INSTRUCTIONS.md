# How to Run the Resilience Grid Dashboard

This guide will help you start the **Urban Climate Vulnerability & Resilience Dashboard**.

## 1. Environment Setup

Ensure you have the required Python dependencies installed. You can install them using the updated `requirements.txt`:

```bash
pip install -r requirements.txt
```

**Additional Spatial Libraries:**
The resilience grid requires spatial processing libraries:
```bash
pip install shapely pykml lxml
```

## 2. Starting the Application

The application uses **FastAPI** as the backend and serves the dashboard as a static site.

### Command to Start:
Run the following command in the project root:

```bash
python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

- `--reload`: Auto-restarts the server when you make code changes.
- `--port 8000`: Changes the port to 8000 (default).

## 3. Accessing the Dashboard

Once the server is running, open your web browser and navigate to:

**[http://localhost:8000](http://localhost:8000)**

## 4. Using the Resilience Features

### Visualize the Resilience Grid:
1.  Click the **"Resilience Grid"** button in the analysis panel.
2.  Wait for the AI to calculate the live risk scores (fetching real-time weather from Open-Meteo).
3.  Click on individual **grid cells** on the map to see specific risk factors (e.g., "Drainage Proximity").

### Find a Safe Route:
1.  Click the **"Safe Route Finder"** button.
2.  **Click on the map** to drop a **Start** marker.
3.  **Click again** on a different location to drop a **Destination** marker.
4.  Click **"Find Safest Path"**.
5.  The system will draw a blue dashed line of the safest route avoiding flood-prone areas.

## 5. Troubleshooting
- **Missing Models:** If you see "Model not loaded" in the dashboard, ensure the `.joblib` files are correctly placed in the `models/` directory.
- **Weather API:** The live grid requires an internet connection to fetch rainfall data from Open-Meteo.
