Urban Flooding Resilience Framework Walkthrough
I have implemented a prototype for the Resilience Grid framework, which moves Beyond simple city-wide rainfall predictions to hyper-local, street-level flood risk assessment for Hyderabad.

1. Project Context
Base Case: Monthly city-level rainfall models for Hyderabad.
Challenge: Urban flooding (e.g., the 2020 cloudbursts) requires area-specific "Nowcasting" and vulnerability mapping.
Solution: A 500m x 500m grid-based vulnerability system.
2. Key Components Implemented
utils/grid_logic.py
This is the core engine that:

Generates a spatial grid for any bounding box in Hyderabad.
Parses multiple KML layers (Nalas, Lakes, Historical Hotspots).
Calculates a Vulnerability Score (0-100) based on proximity to drainage and water bodies.
prototype_grid.py
A demonstration script targeting the Khairatabad/Banjara Hills area.

Correctly identifies high-risk zones near Nalas and Historical Hotspots.
Dashboard Integration
The resilience grid is now fully interactive via the web dashboard:

New UI Tab: A "Resilience Grid" button in the analysis models panel.
Dynamic Visualization: When activated, the dashboard map renders a color-coded grid (Safe = Green, High Risk = Red) over central Hyderabad.
Interactive Popup: Clicking any grid cell shows its Score and specific "Risk Factors" (e.g., "Drainage Proximity").
Phase 2: Live Nowcasting Integration
The framework now features real-time flood risk assessment:

Live Weather Service
: Connects to Open-Meteo to fetch current Hyderabad rainfall every time the grid is loaded.
Dynamic Scoring: The vulnerability_score is now a combination of static spatial factors and current rainfall intensity.
Flash Flood Alerts: If current rainfall exceeds 50mm, the system triggers a "CRITICAL" alert and marks all cells with higher risk weights.
Live Indicator: A new information box on the dashboard shows "Current Rainfall" and "Next 1h Prediction".
Phase 3: Route Safety Optimization
A citizen-focused navigation layer:

Safe Routing Service
: Implements an A* algorithm that treats high-vulnerability grid cells as "impassable" or high-cost barriers.
Interactive Selection: Users can click anywhere on the dashboard map to set a Start and Destination.
Visual Guidance: The safest path is rendered as a blue dashed polyline, with specific segments marked as Green (Safe) or Red (Warning) to justify the detour.
3. Resilience Benefits for End Users
Citizens: Direct navigation support during active floods. The system can divert a user away from a 500m cell that it knows has poor drainage and is currently receiving heavy rain.
City Admin (GHMC): Can identify "Disconnected Zones"—areas where no safe route exists—to prioritize emergency asset deployment.
4. Logical Validation
The scoring logic was tested against:

Hyd_Nalas.kml
: Cells close to major drains receive a proximity penalty.
Hyd_FloodingLocations.kml
: Historical hotspots are weighted with +20 to ensure they are always highlighted.
Next Steps for Development
Dashboard Integration: Add a Mapbox or Leaflet view to the existing FastAPI dashboard to visualize the resilience_grid_sample.csv.
Live Weather API: Feed hourly rainfall data into the grid to make the scores dynamic (Real-time Nowcasting).