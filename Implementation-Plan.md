Resilience Grid Prototype Implementation
This document tracks the technical implementation of the Resilience Grid component for the Urban Flooding Framework.

Objective
Create a Python module that divides a specified bounding box in Hyderabad into 500m x 500m cells and calculates a Static Vulnerability Score for each.

Technical Components
1. Grid Generator (
utils/grid_logic.py
) [NEW]
Functionality: Generates a set of geographic coordinates defining the grid.
Logic:
Bounding box for Hyderabad: ~[17.2, 78.2] to [17.6, 78.6].
Cell Size: 0.0045 degrees (~500m).
Data Structure: GeoJSON or Pandas DataFrame with cell_id, 
lat
, lon, and 
score
.
2. Vulnerability Scoring
Drainage Proximity: Distance to nearest object in 
Hyd_Nalas.kml
 or 
Hyd_Canals&Drains.kml
.
Water Body Score: Penalty for being close to 
Hyd_Tanks&Lakes.kml
 which may overflow.
Elevation (Placeholder): For the prototype, we will use a simplified "sink" logic or static elevation offsets if DEM data is not immediately available.
3. Demonstration Script (
prototype_grid.py
) [NEW]
Purpose: A CLI script to run the grid generation for a 5km x 5km sample area (e.g., around Khairatabad/GHMC HQ).
Output: A CSV file with grid scores and a summary report.
Implementation Tasks
 Implement GridCell class and basic scoring logic.
 Parse KML files for proximity weights.
 Generate sample grid for Khairatabad area.
 Integrate grid visualization into the web dashboard.
Phase 2: Live Weather & Nowcasting [NEW]
Transition from static vulnerability to real-time risk assessment.

1. Weather Data Integration (
utils/weather_service.py
)
Source: Open-Meteo Historical/Current API.
Data Points: Current rainfall (last 1h, last 6h), precipitation probability.
Logic: Fetch Hyderabad-specific data every time the grid is requested or as a background cache.
2. Dynamic Risk Calculation
Static Vulnerability (SV): The base score (0-100) from drainage/spatial data.
Rainfall Factor (RF): A multiplier or additive score based on current rainfall.
Total Risk: Score = SV + (Current_Rainfall_mm * Multiplier).
Thresholds: If Current_Rainfall > 50mm in 1h, even "Safe" cells enter "Warning" status.
3. Dashboard Warning System
Risk Level Gauge: A real-time indicator on the dashboard (Safe / Warning / Critical).
Auto-Refresh: The grid will reflect live data without a full page reload.
Phase 3: Route Safety Optimization [NEW]
Helping citizens navigate safely during active flood events.

1. Grid-Aware Routing Logic
Objective: Calculate paths between two points that avoid grid cells with vulnerability_score > 70.
Implementation: Use A* or Dijkstra's algorithm where the "cost" of a path segment is weighted by the underlying grid cell's risk.
Data Source: Existing Resilience Grid scores.
2. Dashboard Integration
Feature: A "Safe Route" tab on the dashboard.
Controls: Input "Start" and "Destination" (via map click or search).
Visualization: Draw the "Safest Path" vs "Fastest Path" (standard) to highlight risk avoidance.