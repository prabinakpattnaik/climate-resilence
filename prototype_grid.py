import os
from utils.grid_logic import ResilienceGrid

def main():
    # Define bounding box for a central part of Hyderabad (approx 5km x 5km)
    # Covering areas like Banjara Hills, Khairatabad, Panjagutta
    bbox = [17.40, 78.42, 17.45, 78.47] 
    
    print("Initializing Resilience Grid Prototype for Hyderabad...")
    rg = ResilienceGrid(bbox, cell_size=0.0045) # ~500m cells
    
    # Load project KML files
    kml_dir = "kml_files"
    
    # Mapping our logical features to the available KML files
    kml_mappings = {
        'nalas': 'Hyd_Nalas.kml',
        'drains': 'Hyd_Canals&Drains.kml',
        'lakes': 'Hyd_Tanks&Lakes.kml',
        'hotspots': 'Hyd_FloodingLocations.kml'
    }
    
    for feature, filename in kml_mappings.items():
        path = os.path.join(kml_dir, filename)
        if os.path.exists(path):
            rg.load_kml_features(path, feature)
        else:
            print(f"Warning: Missing expected KML file: {path}")

    print("Calculating vulnerability scores for the grid...")
    grid_results = rg.calculate_vulnerability()
    
    # Sort by vulnerability to see top hotspots
    hotspots = grid_results.sort_values(by='vulnerability_score', ascending=False).head(10)
    
    print("\n--- Top 10 High-Risk Grid Cells ---")
    print(hotspots[['lat', 'lon', 'vulnerability_score', 'risk_factors']])
    
    # Save to CSV for analysis
    output_file = "resilience_grid_sample.csv"
    grid_results.to_csv(output_file, index=False)
    print(f"\nFull grid results saved to {output_file}")

if __name__ == "__main__":
    main()
