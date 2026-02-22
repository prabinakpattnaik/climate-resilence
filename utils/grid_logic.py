import os
import numpy as np
import pandas as pd
from lxml import etree
from pykml import parser
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import nearest_points

class ResilienceGrid:
    def __init__(self, bbox, cell_size=0.0045):
        """
        bbox: [min_lat, min_lon, max_lat, max_lon]
        cell_size: degrees (~0.0045 is roughly 500m)
        """
        self.bbox = bbox
        self.cell_size = cell_size
        self.grid = self._generate_grid()
        self.spatial_features = {}

    def _generate_grid(self):
        lats = np.arange(self.bbox[0], self.bbox[2], self.cell_size)
        lons = np.arange(self.bbox[1], self.bbox[3], self.cell_size)
        
        cells = []
        for lat in lats:
            for lon in lons:
                cells.append({
                    'lat': lat + self.cell_size/2,
                    'lon': lon + self.cell_size/2,
                    'vulnerability_score': 0.0,
                    'risk_factors': []
                })
        return pd.DataFrame(cells)

    def load_kml_features(self, kml_path, feature_name):
        """Parses KML and extracts geometries."""
        if not os.path.exists(kml_path):
            print(f"Warning: {kml_path} not found.")
            return

        with open(kml_path, 'rb') as f:
            doc = parser.parse(f).getroot()

        geometries = []
        # Find all Placemarks
        namespaces = {'kml': 'http://www.opengis.net/kml/2.2'}
        placemarks = doc.xpath('//kml:Placemark', namespaces=namespaces)

        for pm in placemarks:
            # Handle LineString (Nalas/Drains)
            lines = pm.xpath('.//kml:LineString/kml:coordinates', namespaces=namespaces)
            for line in lines:
                text = "".join(line.xpath('.//text()')).strip()
                if not text: continue
                coords = [tuple(map(float, c.split(',')))[:2] for c in text.split()]
                if len(coords) >= 2:
                    geometries.append(LineString(coords))

            # Handle Polygon (Lakes/Tanks)
            polys = pm.xpath('.//kml:Polygon//kml:coordinates', namespaces=namespaces)
            for poly in polys:
                text = "".join(poly.xpath('.//text()')).strip()
                if not text: continue
                coords = [tuple(map(float, c.split(',')))[:2] for c in text.split()]
                if len(coords) >= 3:
                    geometries.append(Polygon(coords))

            # Handle Point (Flood Hotspots)
            points = pm.xpath('.//kml:Point/kml:coordinates', namespaces=namespaces)
            for pt in points:
                text = "".join(pt.xpath('.//text()')).strip()
                if not text: continue
                coords = tuple(map(float, text.split(',')))[:2]
                geometries.append(Point(coords))

        self.spatial_features[feature_name] = geometries
        print(f"Loaded {len(geometries)} features for {feature_name}")

    def calculate_vulnerability(self, live_rainfall=0.0):
        """
        Calculates vulnerability based on proximity to drainage, water bodies, 
        and optional live rainfall for Nowcasting.
        """
        for idx, row in self.grid.iterrows():
            cell_pt = Point(row['lon'], row['lat'])
            score = 10.0 # Base neutral score
            risk_factors = []

            # Factor 1: Proximity to Nalas/Drains
            if 'nalas' in self.spatial_features and self.spatial_features['nalas']:
                distances = [cell_pt.distance(g) for g in self.spatial_features['nalas']]
                min_dist = min(distances)
                if min_dist < 0.002:
                    inc = (0.002 - min_dist) * 5000 
                    score += inc
                    risk_factors.append(f"Drainage Proximity (+{round(inc, 1)})")

            # Factor 2: Proximity to Lakes
            if 'lakes' in self.spatial_features and self.spatial_features['lakes']:
                distances_lake = [cell_pt.distance(g) for g in self.spatial_features['lakes']]
                min_dist_lake = min(distances_lake)
                if min_dist_lake < 0.003:
                    inc = (0.003 - min_dist_lake) * 3000
                    score += inc
                    risk_factors.append(f"Lake Buffer Zone (+{round(inc, 1)})")

            # Factor 3: Historical Flood Locations
            if 'hotspots' in self.spatial_features and self.spatial_features['hotspots']:
                distances_hot = [cell_pt.distance(g) for g in self.spatial_features['hotspots']]
                min_dist_hot = min(distances_hot)
                if min_dist_hot < 0.001:
                    score += 20
                    risk_factors.append("Historical Hotspot (+20)")

            # --- DYNAMIC NOWCASTING FACTOR ---
            if live_rainfall > 0:
                # If current rainfall is high, it amplifies the existing spatial risk
                # Threshold: 20mm start showing risk, >50mm is critical
                rain_inc = live_rainfall * 1.5 
                score += rain_inc
                risk_factors.append(f"Live Rainfall {live_rainfall}mm (+{round(rain_inc, 1)})")
                
                # Critical Threshold Warning
                if live_rainfall > 50:
                    score += 30
                    risk_factors.append("FLASH FLOOD ALERT (Crit. Rain)")

            self.grid.at[idx, 'vulnerability_score'] = min(100, score)
            self.grid.at[idx, 'risk_factors'] = ", ".join(risk_factors)

        return self.grid

if __name__ == "__main__":
    # Sample BBox for Khairatabad / Banjara Hills area
    khairatabad_bbox = [17.40, 78.43, 17.45, 78.48]
    rg = ResilienceGrid(khairatabad_bbox)
    
    # In a real scenario, use paths to actual KMLs
    # rg.load_kml_features('kml_files/Hyd_Nalas.kml', 'nalas')
    # rg.calculate_vulnerability()
    print("ResilienceGrid class initialized.")
