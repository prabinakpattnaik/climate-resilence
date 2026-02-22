# Urban Flooding Resilience Framework Summary

This document summarizes the **Urban Flooding Resilience Framework** developed for Hyderabad, integrated into the Climate AI: ConnectBeacon project.

## 1. Project Objective
Transitioning the dashboard from monthly city-wide predictions to a **Hyper-Local Resilience Framework** that provides real-time, street-level risk intelligence for citizens and city administrators (GHMC).

## 2. Implemented Features

### Phase 1: The Resilience Grid
- **Spatial Intelligence:** Divided central Hyderabad into a **500m x 500m grid**.
- **Static Vulnerability Scoring:** Each cell receives a score (0-100) based on proximity to:
    - **Nalas & Drains:** High risk of overflow.
    - **Tanks & Lakes:** Potential inundation zones.
    - **Historical Hotspots:** Areas documented by city officials as flood-prone.
- **Interactive Map:** Visualized on the dashboard with clickable cells showing specific "Risk Factors."

### Phase 2: Live Nowcasting
- **Real-Time Integration:** Connected the backend to the **Open-Meteo API**.
- **Dynamic Risk Scaling:** Grid scores now change automatically based on current rainfall intensity.
- **Flash Flood Warnings:** Triggers critical dashboard alerts when rainfall exceeds safe drainage capacity (e.g., >50mm/hour).

### Phase 3: Route Safety Optimization
- **Flood-Safe Navigation:** A new tool that helps users find the safest path between two points.
- **Grid-Aware Routing:** Uses an A* algorithm that treats high-risk flood cells (Vulnerability > 70) as barriers, calculating an optimized path that avoids waterlogged areas.
- **Visual Guidance:** Renders the safest route directly on the map alongside the live resilience grid.

## 3. Core Technical Components
- **`utils/grid_logic.py`**: The spatial engine for KML parsing and vulnerability calculation.
- **`utils/weather_service.py`**: Logic for fetching real-time rainfall data.
- **`utils/routing_logic.py`**: A graph-based pathfinder tailored for flood avoidance.
- **FastAPI Backend**: Seamless integration of these services into the existing prediction API.
- **Leaflet Dashboard**: Enhanced frontend with real-time map overlays and point-to-point route selection.

## 4. Operational Benefits
- **For Citizens:** Real-time awareness and navigation support during heavy monsoons.
- **For GHMC/Emergency Services:** A dynamic "Heatmap" of infrastructure vulnerability to prioritize maintenance and rescue deployment.

---
*Developed for Hyderabad Climate Resilience - February 2026*
