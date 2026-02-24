// Climate AI Dashboard - JavaScript
const API_BASE = '';  // Same origin

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
let precisionFarmLayer = L.featureGroup();

// Add Layer Control
const baseMaps = {
    "Standard View": lightMap,
    "Satellite View": satelliteMap
};
const overlayMaps = {
    "Resilience Grid": gridLayer,
    "Emergency Hub": emergencyLayer,
    "Precision Farm Health": precisionFarmLayer
};
L.control.layers(baseMaps, overlayMaps).addTo(map);

// Add center indicator
L.marker([17.385, 78.4867]).addTo(map)
    .bindPopup('<b>City Center (Hyderabad)</b><br>Click nearby to start routing.')
    .openPopup();

// Interaction State
let startPoint = null;
let endPoint = null;
let startMarker = null;
let endMarker = null;
let routePolyline = null;
let currentLayer = null;
let isReportingFlooding = false;
// Single source of truth for selected farm/urban points
let selectedFarmPoint = null;
let selectedUrbanPoint = null;

async function fetchTourismSafety() {
    const list = document.getElementById('tourism-landmark-list');
    if (!list) return;

    try {
        const res = await fetch(`${API_BASE}/tourism_safety`);
        const data = await res.json();

        // Update Weather Context (NEW)
        const weatherCtx = document.getElementById('tourism-weather-context');
        if (weatherCtx && data.weather_summary) {
            const w = data.weather_summary;
            weatherCtx.innerHTML = `🌦️ <b>Live Weather Sync:</b> ${w.current_rainfall_mm}mm Rain | Next Hour: ${w.predicted_next_1h_mm}mm`;
        }

        list.innerHTML = '';
        data.reports.forEach(r => {
            const item = document.createElement('div');
            item.style.cssText = `
                padding: 1rem; 
                background: #f8fafc; 
                border: 1px solid ${r.risk_score > 70 ? '#fecaca' : '#e2e8f0'}; 
                border-radius: 8px;
            `;

            const riskColor = r.risk_score > 70 ? '#dc2626' : r.risk_score > 40 ? '#ca8a04' : '#16a34a';

            item.innerHTML = `
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <strong style="color: #1e293b;">${r.name}</strong>
                    <span style="font-size: 0.7rem; font-weight: 700; color: ${riskColor}; text-transform: uppercase;">${r.status}</span>
                </div>
                <div style="font-size: 0.8rem; color: #475569; line-height: 1.4;">
                    🛡️ <b>Advice:</b> ${r.advice}
                </div>
                <div style="margin-top: 0.5rem; display: flex; gap: 0.5rem;">
                    <span style="font-size: 0.65rem; background: #fff; padding: 2px 6px; border-radius: 4px; border: 1px solid #e2e8f0;">${r.type}</span>
                    <button onclick="map.flyTo([${r.lat}, ${r.lon}], 16)" style="font-size: 0.65rem; background: #2563eb; color: white; border: none; padding: 2px 8px; border-radius: 4px; cursor: pointer;">🔍 View on Map</button>
                </div>
            `;
            list.appendChild(item);

            // Add blue marker for tourist sites
            L.marker([r.lat, r.lon], {
                icon: L.icon({
                    iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-gold.png',
                    shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
                    iconSize: [25, 41], iconAnchor: [12, 41]
                })
            }).bindPopup(`<b>${r.name}</b><br>Risk: ${r.status}<br>${r.advice}`)
                .addTo(map);
        });
    } catch (err) {
        console.error('Tourism fetch failed', err);
    }
}

async function generateMediaPSA() {
    const ticker = document.getElementById('media-ticker');
    const social = document.getElementById('media-social');
    const action = document.getElementById('media-action');
    const btn = document.getElementById('generate-psa');

    if (!ticker) return;

    btn.textContent = '✨ Synthesizing...';
    btn.disabled = true;

    try {
        const res = await fetch(`${API_BASE}/generate_media_alert`);
        const data = await res.json();

        ticker.textContent = data.ticker_tape;
        social.textContent = data.social_media_brief;
        action.textContent = data.community_action;

    } catch (err) {
        console.error('Media Alert failed', err);
    } finally {
        btn.textContent = '✨ SYNTHESIZE NEW PSA';
        btn.disabled = false;
    }
}

document.getElementById('generate-psa').addEventListener('click', generateMediaPSA);

async function fetchIoTSensors() {
    const grid = document.getElementById('iot-sensor-grid');
    const indicator = document.getElementById('iot-refresh-indicator');
    if (!grid) return;

    try {
        indicator.textContent = '⌛ Syncing with Nodes...';
        const res = await fetch(`${API_BASE}/iot_sensor_data`);
        const data = await res.json();

        grid.innerHTML = '';

        // 1. ADD REAL CITY-WIDE AQI HEADER (NEW)
        if (data.real_city_aqi) {
            const aq = data.real_city_aqi;
            const aqCard = document.createElement('div');
            aqCard.className = 'real-data-source';
            aqCard.style.cssText = `
                grid-column: 1 / -1;
                padding: 0.75rem;
                background: #f0fdf4;
                border: 1px solid #bbf7d0;
                border-radius: 6px;
                margin-bottom: 0.5rem;
                display: flex;
                justify-content: space-between;
                align-items: center;
                font-size: 0.8rem;
            `;
            const aqColor = aq.status === 'Healthy' ? '#166534' : aq.status === 'Moderate' ? '#92400e' : '#991b1b';
            aqCard.innerHTML = `
                <span>🌍 <b>REAL DATA:</b> Hyderabad Air Quality (AQI)</span>
                <span style="font-weight: 700; color: ${aqColor};">${aq.aqi} - ${aq.status}</span>
                <span style="font-size: 0.65rem; color: #64748b;">PM2.5: ${aq.pm2_5} | O3: ${aq.ozone}</span>
            `;
            grid.appendChild(aqCard);
        }

        data.sensors.forEach(s => {
            const card = document.createElement('div');
            card.style.cssText = `
                padding: 0.75rem; 
                background: white; 
                border: 1px solid ${s.status === 'Critical' ? '#fee2e2' : '#e2e8f0'}; 
                border-radius: 6px; 
                box-shadow: 0 1px 2px rgba(0,0,0,0.05);
            `;

            const nalaColor = s.nala_fill_pct > 70 ? '#dc2626' : s.nala_fill_pct > 50 ? '#ca8a04' : '#16a34a';

            card.innerHTML = `
                <div style="font-weight: 600; font-size: 0.85rem; margin-bottom: 0.5rem; color: #1e293b;">📍 ${s.ward}</div>
                <div style="font-size: 0.75rem; color: #64748b; margin-bottom: 0.25rem;">Nala Fill: <b style="color: ${nalaColor}">${s.nala_fill_pct}%</b></div>
                <div style="font-size: 0.75rem; color: #64748b; margin-bottom: 0.25rem;">Surface: <b>${s.surface_temp_c}°C</b></div>
                <div style="font-size: 0.75rem; color: #64748b;">Water Table: <b>${s.water_depth_m}m</b></div>
                <div style="margin-top: 0.5rem; text-align: right;">
                    <span style="font-size: 0.6rem; padding: 2px 6px; border-radius: 10px; background: ${s.status === 'Critical' ? '#fee2e2' : '#f0fdf4'}; color: ${s.status === 'Critical' ? '#991b1b' : '#166534'};">● ${s.status}</span>
                </div>
            `;
            grid.appendChild(card);
        });

        indicator.textContent = `Live @ ${data.timestamp}`;
    } catch (err) {
        indicator.textContent = '❌ Sync Failed';
        console.error(err);
    }
}

document.getElementById('refresh-iot').addEventListener('click', fetchIoTSensors);

// 2. UI LOGIC & EVENT LISTENERS

// Model button switching
document.querySelectorAll('.model-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.model-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.model-form').forEach(f => f.classList.remove('active'));

        btn.classList.add('active');
        const model = btn.dataset.model;
        const formId = model === 'crop' ? 'crop-form' :
            model === 'iot' ? 'iot-panel' :
                model === 'tourism' ? 'tourism-panel' :
                    model === 'media' ? 'media-panel' :
                        `${model}-form`;
        document.getElementById(formId).classList.add('active');

        // Auto-show result card for data-driven panels
        const resultCard = document.getElementById('result-card');
        if (['iot', 'tourism', 'media'].includes(model)) {
            resultCard.style.display = 'flex';
            resultCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }

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
        } else if (model === 'agri') {
            statusEl.textContent = '👨‍🌾 Farmer Advisory Ready';
            detailsEl.textContent = 'Enter your crop details for advice. For an Advanced Satellite Scan, click your land on the map first.';
            valueEl.textContent = 'ADVISORY';
        } else if (model === 'iot') {
            statusEl.textContent = '📡 IoT Network Active';
            detailsEl.textContent = 'Real-time telemetry from ward-level sensors. Nala levels and heat monitoring active.';
            valueEl.textContent = 'LIVE';
            fetchIoTSensors();
        } else if (model === 'tourism') {
            statusEl.textContent = '🏛️ Resilient Tourism Active';
            detailsEl.textContent = 'Checking heritage sites & landmarks against climate risks. Safety advice loaded.';
            valueEl.textContent = 'TOURISM';
            fetchTourismSafety(); // Auto-load
        } else if (model === 'media') {
            statusEl.textContent = '📢 Media Intelligence Hub';
            detailsEl.textContent = 'Synthesizing live PSAs for newsTicker and Social Broadcasts. Media toolkit ready.';
            valueEl.textContent = 'MEDIA';
            generateMediaPSA(); // Auto-load
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

    // 2. Determine Mode from Active Button
    const activeBtn = document.querySelector('.model-btn.active');
    const activeModel = activeBtn ? activeBtn.dataset.model : null;

    if (activeModel !== 'route' && activeModel !== 'agri' && activeModel !== 'grid') return;

    const statusEl = document.getElementById('result-status');

    if (activeModel === 'agri') {
        selectedFarmPoint = e.latlng;
        startPoint = e.latlng;

        if (startMarker) map.removeLayer(startMarker);
        if (precisionFarmLayer) precisionFarmLayer.clearLayers();

        startMarker = L.marker(startPoint, {
            icon: L.icon({
                iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-green.png',
                shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
                iconSize: [25, 41],
                iconAnchor: [12, 41],
                popupAnchor: [1, -34],
                shadowSize: [41, 41]
            })
        }).addTo(map).bindPopup('<b>📍 Farm Center Selected</b><br>Coordinates captured and ready for scan.').openPopup();

        statusEl.innerHTML = '<span style="color: #16a34a; font-weight: bold;">✓ Farm location captured!</span> Now click "Scan Farm Health" in the sidebar.';
        return;
    }

    if (activeModel === 'grid') {
        selectedUrbanPoint = e.latlng;

        if (startMarker) map.removeLayer(startMarker);
        startMarker = L.marker(e.latlng, {
            icon: L.icon({
                iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-blue.png',
                shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
                iconSize: [25, 41],
                iconAnchor: [12, 41],
                popupAnchor: [1, -34],
                shadowSize: [41, 41]
            })
        }).addTo(map).bindPopup('<b>🏙️ Neighborhood Selected</b><br>Locality captured. Ready for Precision Urban Scan.').openPopup();

        return;
    }

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

document.getElementById('agri-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const formData = new FormData(e.target);
    const btn = e.target.querySelector('button');
    const container = document.getElementById('agri-results-container');
    const cardsList = document.getElementById('agri-cards-list');
    const logisticsBox = document.getElementById('agri-logistics-box');

    btn.textContent = '⌛ Compiling AI Advisory...';
    btn.disabled = true;

    try {
        const res = await fetch(`${API_BASE}/agri_advisory`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                crop: formData.get('crop'),
                state: 'Telangana',
                drought_prob: parseFloat(formData.get('drought_prob')) || 0
            })
        });
        const result = await res.json();

        container.style.display = 'block';
        cardsList.innerHTML = '';

        if (result.recommendations.length === 0) {
            cardsList.innerHTML = '<p style="font-size: 0.85rem; color: #64748b;">No high-priority climate risks detected for your selection. Continue normal operations.</p>';
        }

        result.recommendations.forEach(rec => {
            const card = document.createElement('div');
            const bgColor = rec.type === 'SWITCH' ? '#eff6ff' : rec.type === 'ACTION' ? '#f0fdf4' : '#fff7ed';
            const borderColor = rec.type === 'SWITCH' ? '#bfdbfe' : rec.type === 'ACTION' ? '#bbf7d0' : '#ffedd5';
            const titleColor = rec.type === 'SWITCH' ? '#1e40af' : rec.type === 'ACTION' ? '#166534' : '#9a3412';

            card.style.cssText = `padding: 1rem; border-radius: 8px; background: ${bgColor}; border: 1px solid ${borderColor};`;
            card.innerHTML = `
                <div style="font-weight: 700; color: ${titleColor}; margin-bottom: 0.25rem;">${rec.title}</div>
                <div style="font-size: 0.8rem; color: #4b5563; margin-bottom: 0.5rem;">${rec.reason}</div>
                <div style="font-size: 0.75rem; background: #ffffffa0; padding: 0.5rem; border-radius: 4px; border-left: 3px solid ${titleColor};">💡 ${rec.suggestion}</div>
            `;
            cardsList.appendChild(card);
        });

        // Logistics
        logisticsBox.style.background = result.logistics.safe_to_travel ? '#f0fdf4' : '#fef2f2';
        logisticsBox.style.color = result.logistics.safe_to_travel ? '#166534' : '#991b1b';
        logisticsBox.style.border = `1px solid ${result.logistics.safe_to_travel ? '#bbf7d0' : '#fecaca'}`;
        logisticsBox.innerHTML = `
            <strong>Status:</strong> ${result.logistics.safe_to_travel ? '✓ Clear for Transport' : '⚠️ Travel Warning'}<br>
            <p style="margin-top: 0.25rem;">${result.logistics.advice}</p>
        `;

        // Flood Risk Assessment
        renderAgriFloodRisk(result.flood_risk);

        // Update Summary Card
        document.getElementById('result-card').style.display = 'flex';
        document.getElementById('result-value').textContent = 'ADVISORY READY';
        document.getElementById('result-status').textContent = 'Climate-Smart Agri Intelligence';
        document.getElementById('result-details').textContent = `Generated ${result.recommendations.length} recommendations based on ${result.weather.current_rainfall_mm}mm live rainfall and ${formData.get('drought_prob')}% drought probability.`;

    } catch (err) {
        showError('Failed to generate advisory');
    } finally {
        btn.textContent = 'Generate Advisory Report';
        btn.disabled = false;
    }
});

// Phase 8 - Satellite Farm Scan logic
document.getElementById('scan-farm-health').addEventListener('click', async () => {
    const point = selectedFarmPoint || startPoint;

    if (!point) {
        showError('📍 Please click on your farm on the map first to set the scan center.');
        document.getElementById('map').scrollIntoView({ behavior: 'smooth' });
        return;
    }

    const btn = document.getElementById('scan-farm-health');
    const legend = document.getElementById('ndvi-legend');
    btn.textContent = '🛰️ Running AI Spectral Analysis...';
    btn.disabled = true;

    try {
        const res = await fetch(`${API_BASE}/farm_health_scan`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                start_lat: point.lat,
                start_lon: point.lng,
                end_lat: 0, end_lon: 0 // placeholders
            })
        });
        const data = await res.json();

        renderPrecisionFarmHealth(data.grid);
        legend.style.display = 'block';

        document.getElementById('result-card').style.display = 'flex';
        document.getElementById('result-value').textContent = 'SCAN COMPLETE';
        document.getElementById('result-status').textContent = 'Precision NDVI Health Map Active';
        document.getElementById('result-details').textContent = `Satellite scan identified crop health variance across a 1km plot. Green zones show optimal vigor; Yellow/Red zones indicate moisture stress or potential pest activity.`;

        // NEW: Also show flood resilience during satellite scan
        document.getElementById('agri-results-container').style.display = 'block';
        renderAgriFloodRisk(data.flood_risk);

    } catch (err) {
        showError('Satellite scan failed');
    } finally {
        btn.textContent = 'Scan Farm Health (Precision UI)';
        btn.disabled = false;
    }
});

document.getElementById('scan-urban-health').addEventListener('click', async () => {
    const point = selectedUrbanPoint;

    if (!point) {
        showError('🏙️ Please click on a specific neighborhood on the map first to set the scan center.');
        document.getElementById('map').scrollIntoView({ behavior: 'smooth' });
        return;
    }

    const btn = document.getElementById('scan-urban-health');
    const legend = document.getElementById('ndvi-legend');
    btn.textContent = '🛰️ Analyzing Locality Spectral Data...';
    btn.disabled = true;

    try {
        const res = await fetch(`${API_BASE}/urban_health_scan`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                start_lat: point.lat,
                start_lon: point.lng,
                end_lat: 0, end_lon: 0
            })
        });
        const data = await res.json();

        // Use a 2km grid for urban vs 1km for farm
        renderPrecisionFarmHealth(data.grid, 0.002);
        legend.style.display = 'block';

        // Auto-switch to Satellite View for ground-level context
        if (map.hasLayer(lightMap)) {
            map.removeLayer(lightMap);
            map.addLayer(satelliteMap);
        }

        const container = document.getElementById('urban-results-container');
        const content = document.getElementById('urban-insights-content');
        if (container) {
            container.style.display = 'block';
            container.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }

        const stats = data.stats;
        content.innerHTML = `
            <p><strong>Locality:</strong> Neighborhood Ground-Truth Analysis</p>
            <div style="margin: 0.75rem 0; padding: 0.5rem; background: #fff; border-left: 4px solid #2563eb; border-radius: 4px;">
                <span title="Greenery vs Concrete Ratio">🌿 <strong>Green Index:</strong> ${stats.avg_greenery_index}</span><br>
                <span title="Likelihood of rapid runoff">🏗️ <strong>Impervious Surface:</strong> ${stats.impervious_surface_pct}%</span>
            </div>
            <div style="display: flex; gap: 0.5rem; margin-top: 0.5rem;">
                <span style="flex: 1; padding: 4px; text-align: center; border-radius: 4px; background: ${stats.heat_island_risk === 'High' ? '#fee2e2' : '#f0fdf4'}; color: ${stats.heat_island_risk === 'High' ? '#991b1b' : '#166534'};">
                    🔥 Heat: ${stats.heat_island_risk}
                </span>
                <span style="flex: 1; padding: 4px; text-align: center; border-radius: 4px; background: ${stats.drainage_bottleneck_risk === 'Critical' ? '#fee2e2' : '#fffbeb'}; color: ${stats.drainage_bottleneck_risk === 'Critical' ? '#991b1b' : '#92400e'};">
                    🌊 Drain: ${stats.drainage_bottleneck_risk}
                </span>
            </div>
            <p style="margin-top: 0.75rem; font-size: 0.7rem; color: #64748b; font-style: italic;">
                *Analysis based on 2km locality clipping. High concrete (>70%) indicates flash flood sensitivity.
            </p>
        `;

        document.getElementById('result-card').style.display = 'flex';
        document.getElementById('result-value').textContent = 'WARD ANALYSIS COMPLETE';
        document.getElementById('result-status').textContent = 'Precision Neighborhood Intelligence';
        document.getElementById('result-details').textContent = `Localized scan identified ${stats.impervious_surface_pct}% concrete coverage. Surface runoff risk is ${stats.drainage_bottleneck_risk} due to urban build-up.`;

    } catch (err) {
        showError('Urban scan failed');
    } finally {
        btn.textContent = 'Scan Local Neighborhood';
        btn.disabled = false;
    }
});

function renderPrecisionFarmHealth(grid, cellSize = 0.001) {
    precisionFarmLayer.clearLayers();

    grid.forEach(cell => {
        const bounds = [
            [cell.lat - cellSize / 2, cell.lon - cellSize / 2],
            [cell.lat + cellSize / 2, cell.lon + cellSize / 2]
        ];

        // NDVI Color Map synced with backend: 
        // >0.75 Green, >0.55 Yellow, >0.35 Orange, <=0.35 Red
        const v = cell.ndvi;
        const color = v > 0.75 ? '#22c55e' : v > 0.55 ? '#facc15' : v > 0.35 ? '#f97316' : '#ef4444';

        // Check if we are in Urban Scan mode (based on cellSize)
        const isUrban = cellSize > 0.0015;
        const adviceLabel = isUrban ? 'Citizen Alert' : 'Farmer Advice';

        L.rectangle(bounds, {
            color: color,
            weight: 0.5,
            fillOpacity: 0.6,
            fillColor: color
        }).bindPopup(`
            <div style="font-family: 'Inter', sans-serif;">
                <strong>${cell.status}</strong><br>
                NDVI: ${cell.ndvi}<br>
                <div style="margin-top: 5px; padding: 5px; background: #f8fafc; border-left: 3px solid ${color}; font-size: 0.8rem;">
                    📢 <strong>${adviceLabel}:</strong><br>${cell.advice}
                </div>
            </div>
        `).addTo(precisionFarmLayer);
    });

    precisionFarmLayer.addTo(map);
    map.fitBounds(precisionFarmLayer.getBounds());
}

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
        L.rectangle(bounds, { color: color, weight: 1, fillOpacity: 0.5, fillColor: color, bubblingMouseEvents: true })
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
    const valueEl = document.getElementById('result-value');
    const statusEl = document.getElementById('result-status');
    const detailsEl = document.getElementById('result-details');

    card.style.display = 'flex';

    if (type === 'rainfall') {
        const mm = result.predicted_rainfall_mm;
        valueEl.textContent = `${mm} mm`;
        statusEl.textContent = result.risk_category;
        const riskColor = mm > 200 ? 'var(--danger)' : mm < 30 ? 'var(--warning)' : 'var(--success)';
        valueEl.style.color = riskColor;
        detailsEl.textContent = `Predicted monthly rainfall for Month ${result.input.month}. Based on lag features (${result.input.lag_1}mm, ${result.input.lag_2}mm, ${result.input.lag_3}mm).`;
    } else if (type === 'drought') {
        const score = result.drought_score;
        valueEl.textContent = `${score}`;
        statusEl.textContent = result.category;
        const riskColor = score > 60 ? 'var(--danger)' : score > 30 ? 'var(--warning)' : 'var(--success)';
        valueEl.style.color = riskColor;
        detailsEl.textContent = `Drought severity index (0-100). ${result.category} detected based on rainfall deficit of ${result.input.deficit_pct}% and monsoon strength ${result.input.monsoon_strength}.`;
    } else if (type === 'heatwave') {
        const prob = (result.heatwave_probability * 100).toFixed(1);
        valueEl.textContent = `${prob}%`;
        statusEl.textContent = result.is_heatwave ? 'HEATWAVE ALERT' : 'No Heatwave Detected';
        const riskColor = result.is_heatwave ? 'var(--danger)' : 'var(--success)';
        valueEl.style.color = riskColor;
        detailsEl.textContent = `Probability of heatwave conditions. Yesterday's max: ${result.input.max_temp_lag1}°C, Humidity: ${result.input.humidity}%. ${result.is_heatwave ? 'Take precautions - stay hydrated and avoid prolonged sun exposure.' : 'Conditions are within normal thresholds.'}`;
    } else if (type === 'crop') {
        const dev = result.yield_deviation_pct;
        valueEl.textContent = `${dev}%`;
        statusEl.textContent = result.impact_category;
        const riskColor = dev > 30 ? 'var(--danger)' : dev > 15 ? 'var(--warning)' : 'var(--success)';
        valueEl.style.color = riskColor;
        detailsEl.textContent = `Predicted yield deviation from historical average. ${result.impact_category} for the selected crop with ${result.input?.rainfall || 0}mm projected rainfall.`;
    } else {
        valueEl.textContent = '--';
        statusEl.textContent = 'Result';
        detailsEl.textContent = JSON.stringify(result);
    }
}

function showError(msg) {
    document.getElementById('result-status').textContent = msg;
    document.getElementById('result-value').textContent = 'Error';
    document.getElementById('result-card').style.display = 'flex';
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
loadHistoricalChart();

// 7. HISTORICAL RAINFALL TREND CHART (Chart.js)
let rainfallChart = null;

async function loadHistoricalChart() {
    const canvas = document.getElementById('rainfall-trend-chart');
    if (!canvas) return;

    try {
        const res = await fetch(`${API_BASE}/historical_rainfall`);
        const data = await res.json();

        const labels = data.long_term_averages.map(a => a.month);
        const avgData = data.long_term_averages.map(a => a.avg_mm);

        const colors = ['#2563eb', '#dc2626', '#16a34a', '#ca8a04', '#7c3aed'];
        const datasets = data.years.map((y, i) => ({
            label: `${y.year}`,
            data: y.monthly.map(m => m.rainfall_mm),
            borderColor: colors[i % colors.length],
            backgroundColor: 'transparent',
            borderWidth: 2,
            tension: 0.3,
            pointRadius: 2
        }));

        datasets.push({
            label: 'Long-term Average',
            data: avgData,
            borderColor: '#94a3b8',
            backgroundColor: 'rgba(148,163,184,0.1)',
            borderWidth: 2,
            borderDash: [6, 3],
            fill: true,
            tension: 0.3,
            pointRadius: 0
        });

        if (rainfallChart) rainfallChart.destroy();

        rainfallChart = new Chart(canvas, {
            type: 'line',
            data: { labels, datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: { usePointStyle: true, font: { size: 11 } }
                    },
                    title: {
                        display: true,
                        text: 'Monthly Rainfall Trend vs Long-term Average (Hyderabad)',
                        font: { size: 14, weight: '600' },
                        color: '#1e293b'
                    }
                },
                scales: {
                    y: {
                        title: { display: true, text: 'Rainfall (mm)' },
                        beginAtZero: true
                    },
                    x: {
                        title: { display: true, text: 'Month' }
                    }
                }
            }
        });
    } catch (err) {
        console.error('Failed to load historical chart', err);
    }
}

function renderAgriFloodRisk(fr) {
    const floodBox = document.getElementById('agri-flood-box');
    if (!floodBox || !fr) return;

    const floodColor = fr.level === 'High' ? '#fef2f2' : fr.level === 'Medium' ? '#fffbeb' : '#f0fdf4';
    const floodBorder = fr.level === 'High' ? '#fecaca' : fr.level === 'Medium' ? '#fde68a' : '#bbf7d0';
    const floodText = fr.level === 'High' ? '#991b1b' : fr.level === 'Medium' ? '#92400e' : '#166534';

    floodBox.style.background = floodColor;
    floodBox.style.border = `1px solid ${floodBorder}`;
    floodBox.style.color = floodText;
    floodBox.innerHTML = `
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
            <strong>Standing Water Risk:</strong>
            <span style="padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; background: ${floodText}; color: white;">${fr.level} Risk</span>
        </div>
        <div style="height: 6px; background: #e2e8f0; border-radius: 3px; margin-bottom: 1rem; overflow: hidden;">
            <div style="height: 100%; width: ${fr.risk_score}%; background: ${floodText}; transition: width 1s ease-in-out;"></div>
        </div>
        <ul style="padding-left: 1.25rem; margin-bottom: 0.5rem;">
            ${fr.drainage_advice.map(adv => `<li style="margin-bottom: 0.25rem;">${adv}</li>`).join('')}
        </ul>
        <p style="font-size: 0.75rem; font-style: italic; border-top: 1px solid ${floodBorder}; pt: 0.5rem; mt: 0.5rem;">
            🌱 <strong>Post-Flood Tip:</strong> ${fr.post_flood_tip}
        </p>
    `;
}

