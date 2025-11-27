// Brain Tumor Classifier - Frontend Logic

const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const results = document.getElementById('results');
const loading = document.getElementById('loading');
const error = document.getElementById('error');
const randomTestBtn = document.getElementById('randomTestBtn');
const reloadRandomBtn = document.getElementById('reloadRandomBtn');

// Feedback elements
const feedbackButtons = document.getElementById('feedbackButtons');
const correctionSection = document.getElementById('correctionSection');
const feedbackThankYou = document.getElementById('feedbackThankYou');
const correctionSelect = document.getElementById('correctionSelect');
const autoAcceptTimer = document.getElementById('autoAcceptTimer');
const timerSeconds = document.getElementById('timerSeconds');

let pieChart = null;
let currentPredictionData = null;
let autoAcceptCountdown = null;
let autoAcceptTimeoutId = null;

// Zoom/Pan State
let zoomState = {
    scale: 1,
    panning: false,
    pointX: 0,
    pointY: 0,
    startX: 0,
    startY: 0
};

// Stats Logic
let stats = {
    correct: 0,
    total: 0
};

// History Logic
let historyData = [];

// Load data from localStorage
if (localStorage.getItem('brainTumorStats')) {
    stats = JSON.parse(localStorage.getItem('brainTumorStats'));
    updateStatsDisplay();
}

if (localStorage.getItem('brainTumorHistory')) {
    historyData = JSON.parse(localStorage.getItem('brainTumorHistory'));
    updateHistoryUI();
}

function updateStats(isCorrect) {
    stats.total++;
    if (isCorrect) {
        stats.correct++;
    }
    
    // Save to localStorage
    localStorage.setItem('brainTumorStats', JSON.stringify(stats));
    updateStatsDisplay();
}

function updateStatsDisplay() {
    const accuracy = stats.total === 0 ? 0 : Math.round((stats.correct / stats.total) * 100);
    document.getElementById('accuracyDisplay').textContent = `${accuracy}%`;
    document.getElementById('countDisplay').textContent = `${stats.correct} / ${stats.total}`;
    
    // Update sidebar stats too
    const sidebarStats = document.getElementById('accuracyDisplaySidebar');
    if (sidebarStats) {
        sidebarStats.textContent = `${accuracy}%`;
    }
}

// Dropzone handlers
dropzone.addEventListener('click', () => fileInput.click());
dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
});
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) handleFile(files[0]);
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) handleFile(e.target.files[0]);
});

// Random Test Button (both buttons use same function)
async function loadRandomTest() {
    // Hide previous results and errors
    results.style.display = 'none';
    error.style.display = 'none';
    loading.style.display = 'block';
    resetFeedbackUI();
    cancelAutoAccept();

    try {
        const response = await fetch('/api/random-test');
        
        if (!response.ok) {
            throw new Error('Failed to fetch random test image');
        }

        const data = await response.json();
        displayResults(data);

    } catch (err) {
        showError('Fehler beim Laden des Testbildes: ' + err.message);
    } finally {
        loading.style.display = 'none';
    }
}

randomTestBtn.addEventListener('click', loadRandomTest);
if (reloadRandomBtn) {
    reloadRandomBtn.addEventListener('click', loadRandomTest);
}

// Handle file upload and prediction
async function handleFile(file) {
    // Validate file
    if (!file.type.startsWith('image/')) {
        showError('Bitte wähle ein Bild aus (PNG, JPG)');
        return;
    }

    if (file.size > 16 * 1024 * 1024) {
        showError('Datei zu groß (max 16MB)');
        return;
    }

    // Hide previous results and errors
    results.style.display = 'none';
    error.style.display = 'none';
    loading.style.display = 'block';
    resetFeedbackUI();
    cancelAutoAccept();

    try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch('/api/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Prediction failed');
        }

        const data = await response.json();
        // Add filename if available from file object
        data.filename = file.name;
        displayResults(data);

    } catch (err) {
        showError('Fehler bei der Analyse: ' + err.message);
    } finally {
        loading.style.display = 'none';
    }
}

// Display prediction results with progressive animation
// --- Display Results ---
// This function updates the UI with the prediction data received from the backend.
// It handles:
// 1. Showing the prediction text and probability.
// 2. Rendering the Pie Chart and Bar Charts (using Chart.js).
// 3. Updating the MRI image and Heatmap overlay.
// 4. Triggering the "Uncertainty Alert" if confidence is low.
function displayResults(data) {
    currentPredictionData = data;
    const { predictions, top_prediction, gradcam, original, filename, model_version, bbox } = data;

    // Top prediction
    const predictionHTML = `
        <div class="prediction-class">${top_prediction.class}</div>
        <div class="prediction-probability">${(top_prediction.probability * 100).toFixed(1)}%</div>
    `;
    document.getElementById('predictionResult').innerHTML = predictionHTML;

    // Filename and Model Version
    if (filename) {
        document.getElementById('filenameDisplay').textContent = filename;
    } else {
        document.getElementById('filenameDisplay').textContent = "Uploaded Image";
    }
    
    if (model_version) {
        document.getElementById('modelVersionDisplay').textContent = model_version;
    }

    // Pie chart
    createPieChart(predictions);

    // Original image
    document.getElementById('originalImage').src = 'data:image/png;base64,' + original;
    
    // Progressive heatmap loading effect
    const heatmapImg = document.getElementById('heatmapImage');
    const analysisOverlay = document.getElementById('analysisProgress');
    
    // Show analysis overlay
    heatmapImg.classList.remove('loaded');
    analysisOverlay.classList.add('active');
    
    // Simulate progressive analysis (1 second delay for effect)
    setTimeout(() => {
        analysisOverlay.classList.remove('active');
        heatmapImg.src = 'data:image/png;base64,' + gradcam;
        heatmapImg.classList.add('loaded');
        
        // Draw bounding box after heatmap loads
        if (bbox) {
            drawBoundingBox(bbox);
        }
    }, 800);

    // Probability bars
    createProbabilityBars(predictions);

    // Show results
    results.style.display = 'block';
    results.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    
    // Start auto-accept countdown
    startAutoAccept();

    // Check for uncertainty
    const uncertaintyAlert = document.getElementById('uncertaintyAlert');
    if (top_prediction.probability < 0.60) {
        uncertaintyAlert.style.display = 'block';
    } else {
        uncertaintyAlert.style.display = 'none';
    }

    // Render Consensus
    if (data.consensus) {
        const card = document.getElementById('consensusCard');
        const badge = document.getElementById('consensusBadge');
        const winner = document.getElementById('consensusWinner');
        const votesContainer = document.getElementById('modelVotes');
        
        card.style.display = 'block';
        badge.textContent = `${data.consensus.result.score} Stimmen`;
        winner.textContent = data.consensus.result.winner;
        
        votesContainer.innerHTML = data.consensus.models.map(m => `
            <div class="model-vote">
                <span class="vote-name">${m.name}</span>
                <div class="vote-bar-bg">
                    <div class="vote-bar-fill" style="width: ${m.confidence * 100}%"></div>
                </div>
                <span class="vote-val">${m.prediction}</span>
            </div>
        `).join('');
    }

    // Render Similar Cases
    if (data.similar_cases) {
        const card = document.getElementById('similarCasesCard');
        const grid = document.getElementById('similarCasesGrid');
        
        card.style.display = 'block';
        grid.innerHTML = data.similar_cases.map(c => `
            <div class="similar-case-item">
                <div class="similar-case-placeholder">
                    ${c.id}
                </div>
                <span class="similar-case-label">${c.label}</span>
                <span class="similar-case-sim">${c.similarity} Match</span>
            </div>
        `).join('');
    }

    // Save to history
    addToHistory(data);
}

// History Functions
function addToHistory(data) {
    // Avoid duplicates if reloading same image
    if (historyData.length > 0 && historyData[0].filename === data.filename) return;

    const historyItem = {
        id: Date.now(),
        filename: data.filename || 'Upload',
        prediction: data.top_prediction.class,
        probability: data.top_prediction.probability,
        timestamp: new Date().toISOString(),
        data: data // Store full data to restore view
    };

    historyData.unshift(historyItem); // Add to top
    if (historyData.length > 50) historyData.pop(); // Limit to 50

    localStorage.setItem('brainTumorHistory', JSON.stringify(historyData));
    updateHistoryUI();
}

function updateHistoryUI() {
    const list = document.getElementById('historyList');
    if (!list) return;

    if (historyData.length === 0) {
        list.innerHTML = '<p class="text-muted">Noch keine Analysen in dieser Sitzung.</p>';
        return;
    }

    list.innerHTML = historyData.map(item => `
        <div class="history-item" onclick="loadFromHistory(${item.id})">
            <div class="history-info">
                <strong>${item.filename}</strong>
                <small>${new Date(item.timestamp).toLocaleTimeString()}</small>
            </div>
            <div class="history-badge">
                ${item.prediction} ${(item.probability * 100).toFixed(0)}%
            </div>
        </div>
    `).join('');
}

function loadFromHistory(id) {
    const item = historyData.find(i => i.id === id);
    if (item) {
        switchView('analyze');
        displayResults(item.data);
    }
}

// Navigation
function switchView(viewName) {
    // Hide all views
    document.querySelectorAll('.view-section').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));

    // Show selected
    document.getElementById(`view-${viewName}`).classList.add('active');
    
    // Update nav button
    const navBtn = document.querySelector(`.nav-item[onclick="switchView('${viewName}')"]`);
    if (navBtn) navBtn.classList.add('active');
}

// Zoom & Pan Logic
const containerOriginal = document.getElementById('containerOriginal');
const containerHeatmap = document.getElementById('containerHeatmap');
const wrapperOriginal = document.getElementById('wrapperOriginal');
const wrapperHeatmap = document.getElementById('wrapperHeatmap');

function setupZoomPan(container, wrapper) {
    if (!container || !wrapper) return;

    container.onwheel = function (e) {
        e.preventDefault();
        const xs = (e.offsetX - zoomState.pointX) / zoomState.scale;
        const ys = (e.offsetY - zoomState.pointY) / zoomState.scale;
        const delta = -e.deltaY;
        
        (delta > 0) ? (zoomState.scale *= 1.2) : (zoomState.scale /= 1.2);
        
        // Limit zoom
        if (zoomState.scale < 1) zoomState.scale = 1;
        if (zoomState.scale > 5) zoomState.scale = 5;

        zoomState.pointX = e.offsetX - xs * zoomState.scale;
        zoomState.pointY = e.offsetY - ys * zoomState.scale;

        applyTransform();
    };

    container.onmousedown = function (e) {
        e.preventDefault();
        zoomState.startX = e.clientX - zoomState.pointX;
        zoomState.startY = e.clientY - zoomState.pointY;
        zoomState.panning = true;
    };

    container.onmouseup = function (e) {
        zoomState.panning = false;
    };

    container.onmousemove = function (e) {
        e.preventDefault();
        if (!zoomState.panning) return;
        zoomState.pointX = e.clientX - zoomState.startX;
        zoomState.pointY = e.clientY - zoomState.startY;
        applyTransform();
    };
}

function applyTransform() {
    const transform = `translate(${zoomState.pointX}px, ${zoomState.pointY}px) scale(${zoomState.scale})`;
    if (wrapperOriginal) wrapperOriginal.style.transform = transform;
    if (wrapperHeatmap) wrapperHeatmap.style.transform = transform;
}

function resetZoom() {
    zoomState = { scale: 1, panning: false, pointX: 0, pointY: 0, startX: 0, startY: 0 };
    applyTransform();
}

// Initialize Zoom (sync both images)
// We attach listeners to both, but they share state so they move together
setupZoomPan(containerOriginal, wrapperOriginal);
// setupZoomPan(containerHeatmap, wrapperHeatmap); // Only need one controller really if they sync, but let's keep it simple. 
// Actually, better to control both with one set of events or just let them be independent? 
// User request: "Zoom & Pan". Usually better if synced.
// Let's attach events to BOTH containers, but they update the SAME state and apply to BOTH wrappers.
if (containerHeatmap) {
    containerHeatmap.onwheel = containerOriginal.onwheel;
    containerHeatmap.onmousedown = containerOriginal.onmousedown;
    containerHeatmap.onmouseup = containerOriginal.onmouseup;
    containerHeatmap.onmousemove = containerOriginal.onmousemove;
}

// Active Learning Simulation (SSE)
function startTraining() {
    const status = document.getElementById('trainingStatus');
    const btn = document.getElementById('startTrainingBtn');
    const progressContainer = document.getElementById('trainingProgressContainer');
    const progressBar = document.getElementById('trainingProgressBar');
    const statusText = document.getElementById('trainingStatusText');
    const alertBox = document.getElementById('trainingAlert');
    
    // Reset UI
    status.textContent = "Training startet...";
    alertBox.className = "alert alert-warning";
    btn.disabled = true;
    progressContainer.style.display = 'block';
    progressBar.style.width = '0%';
    statusText.textContent = "Verbinde...";
    
    // Start SSE connection
    const eventSource = new EventSource('/api/train');
    
    eventSource.onmessage = function(e) {
        const data = JSON.parse(e.data);
        
        // Update Progress
        progressBar.style.width = `${data.progress}%`;
        statusText.textContent = data.message;
        
        // Check for completion
        if (data.progress >= 100) {
            eventSource.close();
            status.textContent = "Training abgeschlossen!";
            alertBox.className = "alert alert-success";
            btn.disabled = false;
            
            // Update model version display
            const versionDisplay = document.getElementById('modelVersionDisplay');
            if (versionDisplay) versionDisplay.textContent = "v2.1-active";
            
            // Hide progress after delay
            setTimeout(() => {
                progressContainer.style.display = 'none';
            }, 3000);
        }
    };
    
    eventSource.onerror = function(e) {
        console.error("EventSource failed:", e);
        eventSource.close();
        status.textContent = "Fehler beim Training.";
        alertBox.className = "alert alert-danger";
        btn.disabled = false;
        progressContainer.style.display = 'none';
    };
}

// Theme Toggle
function toggleTheme() {
    const html = document.documentElement;
    const currentTheme = html.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    
    html.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    
    updateThemeIcon(newTheme);
}

function updateThemeIcon(theme) {
    const btn = document.getElementById('themeToggleBtn');
    if (btn) {
        btn.textContent = theme === 'dark' ? '☀️ Light Mode' : '🌙 Dark Mode';
    }
}

// Initialize Theme
const savedTheme = localStorage.getItem('theme') || 'light';
document.documentElement.setAttribute('data-theme', savedTheme);
document.addEventListener('DOMContentLoaded', () => {
    updateThemeIcon(savedTheme);
});

/**
 * Main Frontend Logic
 * ===================
 * 
 * This file handles all user interactions and API communication.
 * 
 * Key Features:
 * - Image Upload & Drag-and-Drop
 * - API Calls to /api/predict and /api/train
 * - Visualization (Pie Chart, Probability Bars, Heatmap Overlay)
 * - Advanced UI (Dark Mode, Zoom/Pan, History, 3D Viewer Simulation)
 * 
 * How to Modify:
 * - Change API Endpoint: Look for `fetch('/api/predict', ...)` to change where images are sent.
 * - Adjust Thresholds: The uncertainty threshold is set at `0.60` (60%). Search for `uncertaintyAlert` to change it.
 * - Add New Views: Add a new `switchView` case or navigation button logic.
 * - 3D Viewer: The `drawSlice` function currently simulates a brain. Replace this with a real DICOM loader (e.g., using cornerstone.js) for real 3D data.
 */

// Configuration
const API_ENDPOINT = '/api/predict';
const FEEDBACK_ENDPOINT = '/api/feedback';
const CLASSES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary'];

// State Variables

// 3D Viewer Logic (Simulation)
const sliceCanvas = document.getElementById('sliceCanvas');
const sliceSlider = document.getElementById('sliceSlider');
const sliceValue = document.getElementById('sliceValue');
let sliceCtx = sliceCanvas ? sliceCanvas.getContext('2d') : null;

function drawSlice(sliceIndex) {
    if (!sliceCtx) return;
    
    // Clear
    sliceCtx.fillStyle = '#000';
    sliceCtx.fillRect(0, 0, 300, 300);
    
    // Simulate a brain shape that changes size with slice index
    // Middle slices (15) are largest, edges (1, 30) are smallest
    const center = 15;
    const dist = Math.abs(sliceIndex - center);
    const radius = 100 - (dist * 5); // Max radius 100, shrinks by 5 per slice
    
    if (radius > 0) {
        sliceCtx.beginPath();
        sliceCtx.arc(150, 150, radius, 0, Math.PI * 2);
        sliceCtx.fillStyle = '#334155'; // Grey brain matter
        sliceCtx.fill();
        
        // Add some "structure" inside
        sliceCtx.beginPath();
        sliceCtx.arc(150, 150, radius * 0.4, 0, Math.PI * 2);
        sliceCtx.fillStyle = '#1e293b'; // Ventricles
        sliceCtx.fill();
        
        // Simulate a tumor on some slices
        if (sliceIndex >= 12 && sliceIndex <= 18) {
            sliceCtx.beginPath();
            sliceCtx.arc(180, 120, 15, 0, Math.PI * 2);
            sliceCtx.fillStyle = 'rgba(239, 68, 68, 0.8)'; // Red tumor
            sliceCtx.fill();
        }
    }
}

if (sliceSlider) {
    sliceSlider.addEventListener('input', (e) => {
        const val = parseInt(e.target.value);
        sliceValue.textContent = val;
        drawSlice(val);
    });
    
    // Initial draw
    drawSlice(15);
}

// Expose for HTML
window.switchView = switchView;
window.resetZoom = resetZoom;
window.loadFromHistory = loadFromHistory;
window.startTraining = startTraining;
window.toggleTheme = toggleTheme;

// Draw bounding box on images
function drawBoundingBox(bbox) {
    if (!bbox) return;
    
    // Draw on original image
    const bboxRectOriginal = document.getElementById('bboxRectOriginal');
    const bboxLabelOriginal = document.getElementById('bboxLabelOriginal');
    
    bboxRectOriginal.setAttribute('x', `${bbox.x}%`);
    bboxRectOriginal.setAttribute('y', `${bbox.y}%`);
    bboxRectOriginal.setAttribute('width', `${bbox.width}%`);
    bboxRectOriginal.setAttribute('height', `${bbox.height}%`);
    
    // Position label at top-left of bbox
    bboxLabelOriginal.setAttribute('x', `${bbox.x + 1}%`);
    bboxLabelOriginal.setAttribute('y', `${bbox.y + 4}%`);
    bboxLabelOriginal.textContent = `Focus ${(bbox.confidence * 100).toFixed(0)}%`;
    
    // Draw on heatmap
    const bboxRectHeatmap = document.getElementById('bboxRectHeatmap');
    
    bboxRectHeatmap.setAttribute('x', `${bbox.x}%`);
    bboxRectHeatmap.setAttribute('y', `${bbox.y}%`);
    bboxRectHeatmap.setAttribute('width', `${bbox.width}%`);
    bboxRectHeatmap.setAttribute('height', `${bbox.height}%`);
}

// Auto-accept feedback timer
function startAutoAccept() {
    cancelAutoAccept(); // Clear any existing timer
    
    autoAcceptCountdown = 5;
    timerSeconds.textContent = autoAcceptCountdown;
    autoAcceptTimer.style.display = 'block';
    
    const countdownInterval = setInterval(() => {
        autoAcceptCountdown--;
        timerSeconds.textContent = autoAcceptCountdown;
        
        if (autoAcceptCountdown <= 0) {
            clearInterval(countdownInterval);
            // Auto-submit as correct
            submitFeedback(true);
        }
    }, 1000);
    
    // Store interval ID for cleanup
    autoAcceptTimeoutId = countdownInterval;
}

function cancelAutoAccept() {
    if (autoAcceptTimeoutId) {
        clearInterval(autoAcceptTimeoutId);
        autoAcceptTimeoutId = null;
    }
    autoAcceptTimer.style.display = 'none';
}

// Create pie chart
function createPieChart(predictions) {
    const ctx = document.getElementById('pieChart').getContext('2d');

    // Destroy previous chart
    if (pieChart) {
        pieChart.destroy();
    }

    const colors = [
        'rgba(239, 68, 68, 0.8)',   // Glioma - Red
        'rgba(245, 158, 11, 0.8)',  // Meningioma - Orange
        'rgba(16, 185, 129, 0.8)',  // No Tumor - Green
        'rgba(99, 102, 241, 0.8)'   // Pituitary - Blue
    ];

    pieChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: predictions.map(p => p.class),
            datasets: [{
                data: predictions.map(p => p.probability * 100),
                backgroundColor: colors,
                borderWidth: 2,
                borderColor: '#ffffff'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        color: '#0f172a',
                        padding: 15,
                        font: { size: 12, weight: 600 }
                    }
                },
                tooltip: {
                    callbacks: {
                        label: (context) => {
                            return `${context.label}: ${context.parsed.toFixed(1)}%`;
                        }
                    }
                }
            }
        }
    });
}

// Create probability bars
function createProbabilityBars(predictions) {
    const container = document.getElementById('probabilityBars');
    
    // Sort by probability
    const sorted = [...predictions].sort((a, b) => b.probability - a.probability);

    container.innerHTML = sorted.map(pred => `
        <div class="probability-item">
            <div class="probability-label">${pred.class}</div>
            <div class="probability-bar">
                <div class="probability-fill" style="width: ${pred.probability * 100}%">
                    ${(pred.probability * 100).toFixed(1)}%
                </div>
            </div>
        </div>
    `).join('');
}

// Feedback Logic
function showCorrectionDropdown() {
    cancelAutoAccept(); // Stop auto-accept timer
    feedbackButtons.style.display = 'none';
    correctionSection.style.display = 'block';
}

function resetFeedbackUI() {
    feedbackButtons.style.display = 'flex';
    correctionSection.style.display = 'none';
    feedbackThankYou.style.display = 'none';
    cancelAutoAccept();
}

async function submitFeedback(isCorrect) {
    if (!currentPredictionData) return;
    
    cancelAutoAccept(); // Stop auto-accept timer

    const predictedLabel = currentPredictionData.top_prediction.class;
    let trueLabel = predictedLabel;

    if (!isCorrect) {
        trueLabel = correctionSelect.value;
    }

    const feedbackData = {
        filename: currentPredictionData.filename || 'uploaded_image',
        predicted_label: predictedLabel,
        true_label: trueLabel,
        confidence: currentPredictionData.top_prediction.probability,
        timestamp: new Date().toISOString(),
        model_version: currentPredictionData.model_version || 'unknown'
    };

    try {
        const response = await fetch('/api/feedback', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(feedbackData)
        });

        if (response.ok) {
            feedbackButtons.style.display = 'none';
            correctionSection.style.display = 'none';
            feedbackThankYou.style.display = 'block';
            
            // Update stats
            updateStats(isCorrect);
        } else {
            showError('Fehler beim Speichern des Feedbacks');
        }
    } catch (err) {
        showError('Fehler beim Senden des Feedbacks: ' + err.message);
    }
}

// Show error message
function showError(message) {
    error.textContent = message;
    error.style.display = 'block';
    setTimeout(() => {
        error.style.display = 'none';
    }, 5000);
}

// Load device info
async function loadDeviceInfo() {
    try {
        const response = await fetch('/api/health');
        const data = await response.json();
        document.getElementById('deviceInfo').textContent = `Device: ${data.device}`;
        if (data.version) {
             // Initial load might not have a prediction, but we can store it or show it somewhere if needed
             // For now, we update it when results are shown
        }
    } catch (err) {
        console.error('Failed to load device info');
    }
}

// Initialize
loadDeviceInfo();
// Expose functions to global scope for HTML onclick handlers
window.submitFeedback = submitFeedback;
window.showCorrectionDropdown = showCorrectionDropdown;

// Keyboard Shortcuts
document.addEventListener('keydown', (e) => {
    // Only if results are visible
    if (results.style.display === 'none') return;
    
    // Ignore if typing in an input (though we don't have many)
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;

    const key = e.key.toLowerCase();

    // Feedback shortcuts (only if feedback buttons are visible)
    if (feedbackButtons.style.display !== 'none') {
        if (key === 'y') {
            submitFeedback(true);
        } else if (key === 'n') {
            showCorrectionDropdown();
        }
    }
    
    // Correction dropdown shortcuts (only if correction section is visible)
    if (correctionSection.style.display !== 'none') {
        if (key === 'enter') {
             submitFeedback(false);
        }
    }

    // Reload shortcut (Enter) - only if not in correction mode
    if (key === 'enter' && correctionSection.style.display === 'none') {
        // Check if reload button is visible/clickable
        if (reloadRandomBtn && reloadRandomBtn.offsetParent !== null) {
            loadRandomTest();
        } else if (randomTestBtn && randomTestBtn.offsetParent !== null) {
            // Also allow top button if results not shown, but we checked results.display above.
            // Actually, if results are shown, reloadRandomBtn is the one to click.
            loadRandomTest();
        }
    }
});
