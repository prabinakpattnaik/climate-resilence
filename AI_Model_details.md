The project integrates several AI and spatial models to provide a comprehensive climate resilience framework for Hyderabad. Here is the breakdown:



1\. \*\*Machine Learning Models (Predictive)\*\*

These are trained on real historical datasets for Hyderabad/Telangana and are loaded as .joblib files in the backend:



Rainfall Prediction Model: A regression model (typically Random Forest or Gradient Boosting) that predicts monthly rainfall based on historical lags (1, 2, 3, and 12-month historical rainfall data).

Drought Risk Model: Analyzes multi-month rainfall deficits and monsoon strength to categorize current drought vulnerability.

Heatwave Risk Model: A classification model that calculates the probability of a heatwave event by analyzing 7-day temperature averages, humidity levels, and seasonal trends.

Crop Impact Model: Predicts the percentage of yield deviation for specific crops in Telangana based on rainfall anomalies and environmental features.



2\. \*\*Spatial \& Nowcasting Models (Real-Time)\*\*

These are custom logic engines developed specifically for this "Resilience Grid" framework:



Resilience Grid Engine (

utils/grid\_logic.py

): This isn't a single ML model but a Spatial Multi-Criteria Analysis (SMCA) model. It breaks Hyderabad into a 500m x 500m grid and calculates a "Vulnerability Score" (0-100) by weighing:

Proximity to Drainage/Nalas (from KML data).

Proximity to Water Bodies/Lakes (from KML data).

Historical Flooding Hotspots (from KML data).

Live Rainfall intensity (Live "Nowcasting" from Open-Meteo API).

Safe Routing Engine (

utils/routing\_logic.py

): Uses the A (A-Star) Graph Search Algorithm\*. It treats the Resilience Grid as a graph where high-vulnerability cells have a "100x cost penalty," forcing the AI to find the safest possible path around flood-prone areas.



3\. \*\*Localization Model\*\*

Reverse Geocoding (OSM/Nominatim): Translates raw map coordinates into human-readable locality names (e.g., "Banjara Hills", "Khairatabad") to make the route breakdown understandable for citizens.

