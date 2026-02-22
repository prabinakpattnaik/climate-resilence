import heapq
import math
import requests
import time

# Simple cache to avoid redundant geocoding hits
LOCALITY_CACHE = {}

class SafeRouter:
    def __init__(self, grid_df):
        """
        Initializes the router with the resilience grid data.
        grid_df: pandas DataFrame with 'lat', 'lon', 'vulnerability_score'
        """
        self.grid = grid_df
        # Create a indexed lookup for faster neighbor finding
        # We'll treat the grid as a graph where each cell is a node
        self.nodes = {}
        for idx, row in self.grid.iterrows():
            # Round coordinates slightly to handle floating point issues in lookups
            key = (round(row['lat'], 4), round(row['lon'], 4))
            self.nodes[key] = {
                'score': row['vulnerability_score'],
                'lat': row['lat'],
                'lon': row['lon']
            }
            
    def _get_distance(self, p1, p2):
        """Euclidean distance between two lat/lon points."""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def _get_neighbors(self, node_key, cell_size=0.008):
        """Finds 8-neighbor adjacent cells in the grid."""
        lat, lon = node_key
        neighbors = []
        # Check surrounding cells with a small epsilon
        for dlat in [-cell_size, 0, cell_size]:
            for dlon in [-cell_size, 0, cell_size]:
                if dlat == 0 and dlon == 0:
                    continue
                
                n_lat, n_lon = round(lat + dlat, 4), round(lon + dlon, 4)
                if (n_lat, n_lon) in self.nodes:
                    neighbors.append((n_lat, n_lon))
        return neighbors

    def _get_locality(self, lat, lon):
        """Simple reverse geocoding using Nominatim (OSM)."""
        key = (round(lat, 3), round(lon, 3))
        if key in LOCALITY_CACHE:
            return LOCALITY_CACHE[key]
        
        try:
            # Note: In production, use a private geocoder or a more robust service
            url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json&addressdetails=1"
            headers = {'User-Agent': 'ClimateResilienceApp/1.0'}
            resp = requests.get(url, headers=headers, timeout=3)
            if resp.status_code == 200:
                data = resp.json()
                address = data.get('address', {})
                locality = address.get('suburb') or address.get('neighbourhood') or address.get('residential') or address.get('city_district') or "Hyderabad"
                LOCALITY_CACHE[key] = locality
                return locality
        except:
            pass
        return "Central Hyderabad"

    def find_safest_path(self, start_coords, end_coords, risk_threshold=75):
        """
        A* Algorithm that avoids high-risk cells.
        start_coords: (lat, lon)
        end_coords: (lat, lon)
        """
        # Find closest grid cells to start/end points
        start_node = min(self.nodes.keys(), key=lambda k: self._get_distance(k, start_coords))
        end_node = min(self.nodes.keys(), key=lambda k: self._get_distance(k, end_coords))

        queue = [(0, start_node)]
        came_from = {start_node: None}
        cost_so_far = {start_node: 0}

        while queue:
            current_priority, current = heapq.heappop(queue)

            if current == end_node:
                break

            for next_node in self._get_neighbors(current):
                node_data = self.nodes[next_node]
                
                risk_penalty = 1.0
                if node_data['score'] > risk_threshold:
                    risk_penalty = 100.0
                elif node_data['score'] > 50:
                    risk_penalty = 2.0

                new_cost = cost_so_far[current] + self._get_distance(current, next_node) * risk_penalty
                
                if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                    cost_so_far[next_node] = new_cost
                    priority = new_cost + self._get_distance(next_node, end_node)
                    heapq.heappush(queue, (priority, next_node))
                    came_from[next_node] = current

        # Reconstruct path
        if end_node not in came_from:
            return None, "No safe path found"

        raw_nodes = []
        curr = end_node
        while curr is not None:
            raw_nodes.append(curr)
            curr = came_from[curr]
        raw_nodes = raw_nodes[::-1]

        path_list = []
        # Optimization: Only geocode Start and End to avoid timeouts and rate limits
        # Intermediate points will be marked as "En route"
        start_locality = self._get_locality(raw_nodes[0][0], raw_nodes[0][1]) if raw_nodes else "Start"
        end_locality = self._get_locality(raw_nodes[-1][0], raw_nodes[-1][1]) if len(raw_nodes) > 1 else start_locality

        for i, node in enumerate(raw_nodes):
            locality = None
            if i == 0:
                locality = start_locality
            elif i == len(raw_nodes) - 1:
                locality = end_locality
            
            path_list.append({
                'lat': node[0], 
                'lon': node[1], 
                'score': self.nodes[node]['score'],
                'locality': locality
            })
        
        return path_list, "Success"
