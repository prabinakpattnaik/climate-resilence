// Climate AI Dashboard - JavaScript
const API_BASE = '';  // Same origin

// 1. INITIALIZE MAP & GLOBALS FIRST
// 1. INITIALIZE MAP & GLOBALS FIRST
const lightMap = L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
    attribution: '&copy; OpenStreetMap &copy; CARTO'
});

const satelliteMap = L.tileLayer('https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}', {
    attribution: '&copy; Google'
});

const map = L.map('map', {
    center: [17.385, 78.4867],
    zoom: 12,
    layers: [lightMap] // Default
});

// Layer Globals
let gridLayer = L.layerGroup();
let emergencyLayer = L.layerGroup().addTo(map);

// Add Layer Control
const baseMaps = {
    "Standard View": lightMap,
    "Satellite View": satelliteMap
};
const overlayMaps = {
    "Resilience Grid": gridLayer,
    "Emergency Hub": emergencyLayer
};
L.control.layers(baseMaps, overlayMaps).addTo(map);

// Add center indicator
L.marker([17.385, 78.4867]).addTo(map)
    .bindPopup('<b>City Center (Hyderabad)</b><br>Click nearby to start routing.')
    .openPopup();

// Layer Globals
let startPoint = null;
let endPoint = null;
let startMarker = null;
let endMarker = null;
let routePolyline = null;
let currentLayer = null;

// 2. UI LOGIC & EVENT LISTENERS

// Model button switching
document.querySelectorAll('.model-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.model-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.model-form').forEach(f => f.classList.remove('active'));

        btn.classList.add('active');
        const model = btn.dataset.model;
        const formId = model === 'crop' ? 'crop-form' : `${model}-form`;
        document.getElementById(formId).classList.add('active');

        const statusEl = document.getElementById('result-status');
        const detailsEl = document.getElementById('result-details');
        const valueEl = document.getElementById('result-value');

        if (model === 'route') {
            statusEl.textContent = '📍 Map Interaction Required';
            detailsEl.textContent = 'Please scroll down to the map and click to set your Start and Destination points.';
            valueEl.textContent = 'START/END';
        } else if (model === 'grid') {
            statusEl.textContent = '🟦 Ready to Load Grid';
            detailsEl.textContent = 'Click the button below to fetch live weather and calculate urban resilience.';
            valueEl.textContent = '--';
        } else {
            statusEl.textContent = 'Ready for Analysis';
            detailsEl.textContent = 'Enter parameters and click Generate';
            valueEl.textContent = '--';
        }
    });
});

// Map Click Handler for Point Selection
// Map Click Handler for Point Selection & Reporting
map.on('click', (e) => {
    // 1. Check for Community Reporting Mode
    if (isReportingFlooding) {
        const floodMarker = L.marker(e.latlng, {
            icon: L.icon({
                iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-gold.png',
                shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
                iconSize: [25, 41],
                iconAnchor: [12, 41]
            })
        }).addTo(map)
            .bindPopup(`<b>⚠️ Live Flood Report</b><br>Coordinates: ${e.latlng.lat.toFixed(4)}, ${e.latlng.lng.toFixed(4)}<br>Status: Citizen Reported`)
            .openPopup();

        isReportingFlooding = false;
        document.getElementById('map').style.cursor = '';
        return;
    }

    // 2. Check for Routing Mode Selection
    if (!document.getElementById('route-form').classList.contains('active')) return;

    const statusEl = document.getElementById('result-status');

    if (!startPoint) {
        startPoint = e.latlng;
        document.getElementById('route-start').value = `${startPoint.lat.toFixed(4)}, ${startPoint.lng.toFixed(4)}`;
        if (startMarker) map.removeLayer(startMarker);

        startMarker = L.marker(startPoint, {
            icon: L.icon({
                iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-green.png',
                shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
                iconSize: [25, 41],
                iconAnchor: [12, 41],
                popupAnchor: [1, -34],
                shadowSize: [41, 41]
            })
        }).addTo(map).bindPopup('<b>Start Point</b>').openPopup();

        statusEl.textContent = '📍 Start point set. Now click on the map for your Destination.';
    } else if (!endPoint) {
        endPoint = e.latlng;
        document.getElementById('route-end').value = `${endPoint.lat.toFixed(4)}, ${endPoint.lng.toFixed(4)}`;
        if (endMarker) map.removeLayer(endMarker);

        endMarker = L.marker(endPoint, {
            icon: L.icon({
                iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-red.png',
                shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
                iconSize: [25, 41],
                iconAnchor: [12, 41],
                popupAnchor: [1, -34],
                shadowSize: [41, 41]
            })
        }).addTo(map).bindPopup('<b>Destination Point</b>').openPopup();

        statusEl.textContent = '🏁 Destination set! Click "Find Safest Path" to calculate.';
    } else {
        startPoint = e.latlng;
        endPoint = null;
        document.getElementById('route-start').value = `${startPoint.lat.toFixed(4)}, ${startPoint.lng.toFixed(4)}`;
        document.getElementById('route-end').value = '';
        if (startMarker) map.removeLayer(startMarker);
        if (endMarker) map.removeLayer(endMarker);
        if (routePolyline) map.removeLayer(routePolyline);

        startMarker = L.marker(startPoint, {
            icon: L.icon({
                iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-green.png',
                shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
                iconSize: [25, 41],
                iconAnchor: [12, 41],
                popupAnchor: [1, -34],
                shadowSize: [41, 41]
            })
        }).addTo(map).bindPopup('<b>Start Point</b>').openPopup();

        statusEl.textContent = '📍 Start point reset. Click Destination.';
    }
});

// Scroll to Map
document.querySelectorAll('.scroll-to-map').forEach(btn => {
    btn.addEventListener('click', () => {
        document.getElementById('map').scrollIntoView({ behavior: 'smooth' });
    });
});

// Form Submissions
document.getElementById('rainfall-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const formData = new FormData(e.target);
    const data = {
        month: parseInt(formData.get('month')) || 6,
        lag_1: parseFloat(formData.get('lag_1')) || 0,
        lag_2: parseFloat(formData.get('lag_2')) || 0,
        lag_3: parseFloat(formData.get('lag_3')) || 0,
        lag_12: parseFloat(formData.get('lag_12')) || 0
    };
    try {
        const res = await fetch(`${API_BASE}/predict_rainfall`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        if (!res.ok) throw new Error(`API Error: ${res.status}`);
        const result = await res.json();
        displayResult('rainfall', result);
    } catch (err) {
        showError('Failed to get prediction');
    }
});

document.getElementById('drought-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const formData = new FormData(e.target);
    const data = {
        rolling_3mo_avg: parseFloat(formData.get('rolling_3mo')) || 0,
        rolling_6mo_avg: parseFloat(formData.get('rolling_6mo')) || 0,
        deficit_pct: parseFloat(formData.get('deficit_pct')) || 0,
        prev_year_drought: parseFloat(formData.get('prev_year_drought')) || 0,
        monsoon_strength: parseFloat(formData.get('monsoon_strength')) || 0
    };
    try {
        const res = await fetch(`${API_BASE}/predict_drought`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        const result = await res.json();
        displayResult('drought', result);
    } catch (err) {
        showError('Failed to get prediction');
    }
});

document.getElementById('heatwave-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const formData = new FormData(e.target);
    const data = {
        max_temp_lag1: parseFloat(formData.get('max_temp_lag1')) || 0,
        max_temp_lag2: parseFloat(formData.get('max_temp_lag2')) || 0,
        max_temp_lag3: parseFloat(formData.get('max_temp_lag3')) || 0,
        humidity: parseFloat(formData.get('humidity')) || 0,
        month: parseInt(formData.get('month')) || 5
    };
    try {
        const res = await fetch(`${API_BASE}/predict_heatwave`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        const result = await res.json();
        displayResult('heatwave', result);
    } catch (err) {
        showError('Failed to get prediction');
    }
});

document.getElementById('crop-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const formData = new FormData(e.target);
    const data = {
        crop_type: formData.get('crop_type'),
        state: formData.get('state'),
        season: formData.get('season'),
        rainfall: parseFloat(formData.get('rainfall')) || 0,
        rainfall_anomaly: parseFloat(formData.get('rainfall_anomaly')) || 0,
        fertilizer_per_area: parseFloat(formData.get('fertilizer_per_area')) || 0,
        pesticide_per_area: parseFloat(formData.get('pesticide_per_area')) || 0
    };
    try {
        const res = await fetch(`${API_BASE}/predict_crop_impact`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        const result = await res.json();
        displayResult('crop', result);
    } catch (err) {
        showError('Failed to get prediction');
    }
});

document.getElementById('grid-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const btn = e.target.querySelector('button');
    const weatherBox = document.getElementById('live-weather-box');
    const weatherText = document.getElementById('weather-text');

    btn.textContent = '⌛ Calculating Live Risk...';
    btn.disabled = true;

    try {
        const res = await fetch(`${API_BASE}/resilience_grid`);
        const data = await res.json();
        renderResilienceGrid(data.grid);

        weatherBox.style.display = 'block';
        const w = data.weather;
        weatherText.innerHTML = `
            Current Rainfall: <b>${w.current_rainfall_mm} mm</b><br>
            Next 1h Prediction: <b>${w.predicted_next_1h_mm} mm</b><br>
            <span style="font-size: 0.75rem; color: #64748b;">(Source: Open-Meteo @ ${w.timestamp})</span>
        `;

        document.getElementById('result-card').style.display = 'flex';
        document.getElementById('result-value').textContent = w.current_rainfall_mm > 50 ? 'CRITICAL' : 'LIVE GRID';
        document.getElementById('result-status').textContent = w.is_raining ? '⚠️ Active Rainfall Nowcasting' : '✓ Normal Conditions';
        document.getElementById('result-details').textContent = `Analyzed ${data.grid.length} cells. Live weather integrated successfully.`;

    } catch (err) {
        showError(`Failed: ${err.message}`);
    } finally {
        btn.textContent = '🚀 Load Resilience Grid Map';
        btn.disabled = false;
    }
});

document.getElementById('route-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!startPoint || !endPoint) {
        showError('Please select start and end points on the map');
        return;
    }
    const btn = e.target.querySelector('button');
    btn.textContent = '⌛ Searching Safe Path...';
    btn.disabled = true;
    try {
        const res = await fetch(`${API_BASE}/calculate_safe_route`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                start_lat: startPoint.lat, start_lon: startPoint.lng,
                end_lat: endPoint.lat, end_lon: endPoint.lng
            })
        });
        const data = await res.json();
        renderSafeRoute(data.path);
        document.getElementById('result-card').style.display = 'flex';
        document.getElementById('result-value').textContent = 'PATH FOUND';
        document.getElementById('result-status').textContent = 'Flood-Safe Navigation Active';
        document.getElementById('result-details').textContent = `Safest Corridor identified using A* Algorithm. Avoiding high-risk drainage zones (Vulnerability > 75).`;
    } catch (err) {
        showError(err.message);
    } finally {
        btn.textContent = '📍 Find Safest Path';
        btn.disabled = false;
    }
});

// KML Overlays
document.getElementById('load-kml').addEventListener('click', async () => {
    const filename = document.getElementById('kml-select').value;
    if (!filename) return;
    if (currentLayer) map.removeLayer(currentLayer);
    currentLayer = omnivore.kml(`${API_BASE}/kml_data/${filename}`)
        .on('ready', function () { map.fitBounds(this.getBounds()); })
        .addTo(map);
});

// 3. RENDERING FUNCTIONS

function renderResilienceGrid(data) {
    if (gridLayer) map.removeLayer(gridLayer);
    gridLayer = L.layerGroup().addTo(map);
    const cellSize = 0.008;
    data.forEach(cell => {
        const bounds = [[cell.lat - cellSize / 2, cell.lon - cellSize / 2], [cell.lat + cellSize / 2, cell.lon + cellSize / 2]];
        const score = cell.vulnerability_score;
        const color = score > 80 ? '#dc2626' : score > 60 ? '#f97316' : score > 40 ? '#facc15' : score > 20 ? '#a8a29e' : '#10b981';
        L.rectangle(bounds, { color: color, weight: 1, fillOpacity: 0.5, fillColor: color })
            .bindPopup(`Score: ${score.toFixed(1)}<br>${cell.risk_factors || ''}`)
            .addTo(gridLayer);
    });
    map.fitBounds(new L.featureGroup(gridLayer.getLayers()).getBounds());
}

function renderSafeRoute(path) {
    if (routePolyline) map.removeLayer(routePolyline);
    const latlngs = path.map(p => [p.lat, p.lon]);
    routePolyline = L.polyline(latlngs, {
        color: '#2563eb', // Blue
        weight: 8,
        opacity: 0.6,
        lineJoin: 'round'
    }).addTo(map);
    map.fitBounds(routePolyline.getBounds());

    // Populate Path Details Panel
    const panel = document.getElementById('route-details-panel');
    const stats = document.getElementById('route-summary-stats');
    const list = document.getElementById('route-steps-list');

    panel.style.display = 'block';
    list.innerHTML = '';

    // Calculate stats
    const avgRisk = path.reduce((acc, curr) => acc + curr.score, 0) / path.length;
    const highRiskNodes = path.filter(p => p.score > 75).length;

    stats.innerHTML = `
        <div class="route-stat-item">
            <span class="route-stat-label">Total Points</span>
            <span class="route-stat-value">${path.length}</span>
        </div>
        <div class="route-stat-item">
            <span class="route-stat-label">Avg Risk Score</span>
            <span class="route-stat-value">${avgRisk.toFixed(1)}</span>
        </div>
        <div class="route-stat-item">
            <span class="route-stat-label">Avoided Hotspots</span>
            <span class="route-stat-value">${highRiskNodes}</span>
        </div>
    `;

    // Add steps
    path.forEach((p, i) => {
        const li = document.createElement('li');
        li.style.cssText = "padding: 0.75rem; border-bottom: 1px solid #e2e8f0; display: flex; align-items: center; gap: 1rem; font-size: 0.9rem;";
        const riskLabel = p.score > 75 ? 'High Risk' : p.score > 50 ? 'Moderate' : 'Safe';
        const riskColor = p.score > 75 ? '#dc2626' : p.score > 50 ? '#ca8a04' : '#16a34a';

        const locality = p.locality;
        let stepDesc = locality ? `Locality: <b>${locality}</b>` : `Proceeding through area...`;

        li.innerHTML = `
            <span style="background: #f1f5f9; width: 24px; height: 24px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.75rem; font-weight: 600; flex-shrink: 0;">${i + 1}</span>
            <span>${stepDesc} <small style="display:block; color:#64748b; font-size: 0.7rem;">(${p.lat.toFixed(3)}, ${p.lon.toFixed(3)})</small></span>
            <span style="margin-left: auto; font-size: 0.75rem; font-weight: 700; color: ${riskColor};">${riskLabel}</span>
        `;
        list.appendChild(li);
    });
}

function displayResult(type, result) {
    const card = document.getElementById('result-card');
    card.style.display = 'flex';
    document.getElementById('result-value').textContent = type === 'heatwave' ? `${(result.heatwave_probability * 100).toFixed(0)}%` : `${result.predicted_rainfall_mm || result.yield_deviation_pct || 0}%`;
}

function showError(msg) {
    document.getElementById('result-status').textContent = msg;
    document.getElementById('result-value').textContent = 'Error';
}

// 4. DATA LOADING
async function loadKmlList() {
    try {
        const res = await fetch(`${API_BASE}/kml_files`);
        const data = await res.json();
        const select = document.getElementById('kml-select');
        data.files.forEach(file => {
            const opt = document.createElement('option');
            opt.value = file; opt.textContent = file;
            select.appendChild(opt);
        });
    } catch (e) { }
}

async function loadSummary() {
    try {
        const res = await fetch(`${API_BASE}/dashboard_summary`);
        const data = await res.json();

        if (data.live_weather) {
            const w = data.live_weather;
            document.getElementById('stat-rainfall').textContent = `${w.current_mm.toFixed(1)} mm`;
            document.getElementById('stat-drought').textContent = w.status === 'Clear' ? 'Low' : 'Active';

            if (data.baselines) {
                document.getElementById('stat-heatwave').textContent = `${data.baselines.heatwave_prob}%`;
                document.getElementById('stat-crop').textContent = data.baselines.crop_risk;
            }
        }
    } catch (e) { }
}

// 5. RESILIENCE ADDONS
let isReportingFlooding = false;

document.getElementById('opacity-slider').addEventListener('input', (e) => {
    const opacity = e.target.value / 100;
    if (gridLayer) {
        gridLayer.eachLayer(layer => {
            if (layer.setStyle) layer.setStyle({ fillOpacity: opacity, opacity: opacity });
        });
    }
});

document.getElementById('report-flooding').addEventListener('click', () => {
    isReportingFlooding = true;
    document.getElementById('map').style.cursor = 'crosshair';
    alert('📍 Click on the map to mark a location with active flooding.');
});

// NEW: Safe Haven Routing
document.getElementById('find-safe-haven').addEventListener('click', async () => {
    if (!startPoint) {
        showError('Please click on the map to set your current location (Start Point) first.');
        return;
    }

    const btn = document.getElementById('find-safe-haven');
    btn.textContent = '⌛ Finding Nearest Safe Haven...';
    btn.disabled = true;

    try {
        const res = await fetch(`${API_BASE}/find_nearest_safe_haven`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                start_lat: startPoint.lat, start_lon: startPoint.lng,
                end_lat: 0, end_lon: 0
            })
        });

        if (!res.ok) {
            const errorText = await res.text();
            throw new Error(`Server Error (${res.status}): ${errorText.substring(0, 100)}`);
        }

        const data = await res.json();

        // Update destination marker
        if (endMarker) map.removeLayer(endMarker);
        endPoint = L.latLng(data.destination.lat, data.destination.lon);
        document.getElementById('route-end').value = data.destination.name;

        endMarker = L.marker(endPoint, {
            icon: L.icon({
                iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-red.png',
                shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
                iconSize: [25, 41], iconAnchor: [12, 41]
            })
        }).addTo(map).bindPopup(`<b>Destination: ${data.destination.name}</b><br>Type: ${data.destination.type}`).openPopup();

        renderSafeRoute(data.path);

        document.getElementById('result-card').style.display = 'flex';
        document.getElementById('result-value').textContent = 'SAFE HAVEN FOUND';
        document.getElementById('result-status').textContent = `Route to ${data.destination.name} Active`;
        document.getElementById('result-details').textContent = `Targeting nearest ${data.destination.type}. Safe path avoids high-risk flood zones.`;

    } catch (err) {
        showError(`Failed: ${err.message}`);
    } finally {
        btn.textContent = 'Find Nearest Safe Haven';
        btn.disabled = false;
    }
});

// NEW: Emergency Resource Layer

async function loadEmergencyResources() {
    try {
        const res = await fetch(`${API_BASE}/emergency_resources`);
        const data = await res.json();

        emergencyLayer.clearLayers();
        data.resources.forEach(r => {
            const iconUrl = r.type === 'Hospital' ? 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-violet.png' :
                r.type === 'NDRF' ? 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-orange.png' :
                    'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-blue.png'; // Shelter

            L.marker([r.lat, r.lon], {
                icon: L.icon({
                    iconUrl: iconUrl,
                    shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
                    iconSize: [25, 41], iconAnchor: [12, 41]
                })
            }).bindPopup(`<b>${r.name}</b><br>Type: ${r.type}<br>📞 Contact: ${r.contact}`)
                .addTo(emergencyLayer);
        });
    } catch (e) { console.error('Failed to load emergency resources', e); }
}

// 6. STARTUP & SIMULATION
document.querySelectorAll('.simulate-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const formType = btn.dataset.form;
        const scenarios = { rainfall: { month: 6, lag_1: 150.5, lag_2: 80.2, lag_3: 35.0, lag_12: 180.5 } };
        const scenario = scenarios[formType];
        if (scenario) {
            const form = document.getElementById(`${formType}-form`);
            Object.keys(scenario).forEach(key => {
                const input = form.querySelector(`[name="${key}"]`);
                if (input) input.value = scenario[key];
            });
        }
    });
});

loadKmlList();
loadSummary();
loadEmergencyResources();
