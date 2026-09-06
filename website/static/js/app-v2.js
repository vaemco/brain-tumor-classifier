/**
 * Brain Tumor Classifier v3 - Main Application
 * =============================================
 *
 * Features:
 * - Modular drag-and-drop grid layout
 * - uPlot graphs for model confidence visualization
 * - Real-time layer progress animation
 * - Collapsible panels
 * - Image upload & random test
 */

// ========================================
// STATE MANAGEMENT
// ========================================
const AppState = {
    isEditMode: false,
    isAnalyzing: false,
    currentData: null,
    panelOrder: ['upload', 'analysis', 'visualization', 'result'],
    charts: {},
    animationFrames: {},
    animationTimeouts: {}
};

// ========================================
// DOM ELEMENTS
// ========================================
const DOM = {
    // Will be populated on init
};

// ========================================
// INITIALIZATION
// ========================================
document.addEventListener('DOMContentLoaded', () => {
    initDOM();
    initDropzone();
    initPanelCollapse();
    initDragAndDrop();
    loadPanelOrder();

    console.log('Brain Tumor Classifier v3 initialized');
});

function initDOM() {
    DOM.editModeBtn = document.getElementById('editModeBtn');
    DOM.dropzone = document.getElementById('dropzone');
    DOM.fileInput = document.getElementById('fileInput');
    DOM.randomTestBtn = document.getElementById('randomTestBtn');
    DOM.nextImageBtn = document.getElementById('nextImageBtn');
    DOM.loadingOverlay = document.getElementById('loadingOverlay');
    DOM.panelGrid = document.getElementById('panelGrid');
    DOM.resultsContent = document.getElementById('resultsContent');
    DOM.analysisPanel = document.getElementById('analysisPanel');
    DOM.visualizationPanel = document.getElementById('visualizationPanel');
    DOM.resultPanel = document.getElementById('resultPanel');

    // Event listeners
    if (DOM.editModeBtn) {
        DOM.editModeBtn.addEventListener('click', toggleEditMode);
    }
    if (DOM.randomTestBtn) {
        DOM.randomTestBtn.addEventListener('click', loadRandomTest);
    }
    if (DOM.nextImageBtn) {
        DOM.nextImageBtn.addEventListener('click', loadRandomTest);
    }
}

// ========================================
// DROPZONE / FILE UPLOAD
// ========================================
function initDropzone() {
    const dropzone = DOM.dropzone;
    const fileInput = DOM.fileInput;

    if (!dropzone || !fileInput) return;

    dropzone.addEventListener('click', () => fileInput.click());

    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
    });

    dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('dragover');
    });

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileUpload(files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });
}

let activeAbortController = null;

async function handleFileUpload(file) {
    // Validate
    if (!file.type.startsWith('image/')) {
        showError('Please select an image file (PNG, JPG)');
        return;
    }

    if (file.size > 16 * 1024 * 1024) {
        showError('File too large (max 16MB)');
        return;
    }

    if (activeAbortController) {
        activeAbortController.abort();
    }
    activeAbortController = new AbortController();

    showLoading(true);

    try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch('/api/analyze-detailed', {
            method: 'POST',
            body: formData,
            signal: activeAbortController.signal
        });

        const data = await response.json().catch(() => null);
        if (!response.ok || !data) {
            const message = (data && data.error) ? data.error : 'Analysis failed';
            throw new Error(message);
        }
        if (data.error) throw new Error(data.error);

        AppState.currentData = data;
        displayResults(data);

    } catch (err) {
        if (err.name === 'AbortError') {
            console.log('Analysis request aborted');
            return;
        }
        showError('Error during analysis: ' + err.message);
    } finally {
        showLoading(false);
    }
}

// ========================================
// RANDOM TEST
// ========================================
async function loadRandomTest() {
    if (activeAbortController) {
        activeAbortController.abort();
    }
    activeAbortController = new AbortController();

    showLoading(true);

    try {
        const response = await fetch('/api/random-test-detailed', {
            signal: activeAbortController.signal
        });

        const data = await response.json().catch(() => null);
        if (!response.ok || !data) {
            const message = (data && data.error) ? data.error : 'Failed to load test image';
            throw new Error(message);
        }
        if (data.error) throw new Error(data.error);

        AppState.currentData = data;
        displayResults(data);

    } catch (err) {
        if (err.name === 'AbortError') {
            console.log('Random test request aborted');
            return;
        }
        showError('Error loading test image: ' + err.message);
    } finally {
        showLoading(false);
    }
}

// ========================================
// DISPLAY RESULTS
// ========================================
function displayResults(data) {
    if (!data || !data.models) {
        showError('No model results returned from server');
        return;
    }

    // Show result panels
    if (DOM.analysisPanel) DOM.analysisPanel.classList.remove('hidden');
    if (DOM.visualizationPanel) DOM.visualizationPanel.classList.remove('hidden');
    if (DOM.resultPanel) DOM.resultPanel.classList.remove('hidden');

    // Display images
    displayImages(data);

    // Animate model graphs
    animateModelGraphs(data.models);

    // Display pie chart and result
    displayPieChart(data.averaged_predictions);
    displayFinalResult(data.final_result);
    displayProbabilityBars(data.averaged_predictions);

    // Show filename
    const filenameEl = document.getElementById('filenameDisplay');
    if (filenameEl && data.filename) {
        filenameEl.textContent = data.filename;
    }

    // Display auto-evaluation if available (for random test images)
    displayAutoEval(data.auto_eval);

    // Reset feedback UI
    resetFeedbackUI();

    // Ensure loading is hidden (backup)
    showLoading(false);
}

function displayImages(data) {
    const originalImg = document.getElementById('originalImage');
    const heatmapImg = document.getElementById('heatmapImage');

    if (originalImg && data.original_b64) {
        originalImg.src = 'data:image/png;base64,' + data.original_b64;
    }

    if (heatmapImg && data.heatmap_b64) {
        heatmapImg.src = 'data:image/png;base64,' + data.heatmap_b64;
    }

    // Draw bounding box
    if (data.bbox) {
        drawBoundingBox(data.bbox);
    } else {
        clearBoundingBox();
    }
}

function drawBoundingBox(bbox) {
    const rect = document.getElementById('bboxRect');
    if (!rect) return;

    rect.setAttribute('x', `${bbox.x}%`);
    rect.setAttribute('y', `${bbox.y}%`);
    rect.setAttribute('width', `${bbox.width}%`);
    rect.setAttribute('height', `${bbox.height}%`);
}

function clearBoundingBox() {
    const rect = document.getElementById('bboxRect');
    if (!rect) return;

    rect.setAttribute('x', '0');
    rect.setAttribute('y', '0');
    rect.setAttribute('width', '0');
    rect.setAttribute('height', '0');
}

// ========================================
// MODEL GRAPHS WITH UPLOT
// ========================================
function animateModelGraphs(models) {
    const modelOrder = ['resnet18', 'efficientnet', 'densenet'];

    modelOrder.forEach((modelName, index) => {
        const modelData = models[modelName];
        if (!modelData) return;

        const container = document.getElementById(`graph-${modelName}`);
        const confidenceEl = document.getElementById(`confidence-${modelName}`);
        const layerEl = document.getElementById(`layer-${modelName}`);

        if (!container) return;

        // Reset previous state
        if (AppState.charts[modelName]) {
            AppState.charts[modelName].destroy();
        }
        if (AppState.animationFrames[modelName]) {
            cancelAnimationFrame(AppState.animationFrames[modelName]);
        }
        if (AppState.animationTimeouts[modelName]) {
            clearTimeout(AppState.animationTimeouts[modelName]);
        }
        container.innerHTML = '';

        // Animate the confidence build-up
        const layerProgress = modelData.layer_progress;
        const layerNames = modelData.layer_names;
        const finalConfidence = modelData.confidence;
        const color = modelData.color;

        let currentStep = 0;
        const totalSteps = layerProgress.length;
        const stepDuration = 150; // ms per step

        // Create uPlot chart
        const chartWidth = container.clientWidth || (container.parentElement ? container.parentElement.clientWidth : 0) || 320;
        const chartData = [
            [0],  // x-axis (layer index)
            [0]   // y-axis (confidence)
        ];

        const opts = {
            width: chartWidth,
            height: 80,
            cursor: { show: false },
            legend: { show: false },
            scales: {
                x: { time: false, min: 0, max: totalSteps - 1 },
                y: { min: 0, max: 1 }
            },
            axes: [
                { show: false },
                { show: false }
            ],
            series: [
                {},
                {
                    stroke: color,
                    width: 2,
                    fill: color + '33', // 20% opacity
                }
            ]
        };

        const chart = new uPlot(opts, chartData, container);
        AppState.charts[modelName] = chart;

        // Animation function
        function animateStep() {
            if (currentStep >= totalSteps) {
                // Update final confidence
                if (confidenceEl) {
                    confidenceEl.textContent = (finalConfidence * 100).toFixed(1) + '%';
                }
                if (layerEl) {
                    layerEl.textContent = 'Complete';
                }
                return;
            }

            // Build data arrays
            const xData = [];
            const yData = [];
            for (let i = 0; i <= currentStep; i++) {
                xData.push(i);
                yData.push(layerProgress[i]);
            }

            chart.setData([xData, yData]);

            // Update layer name
            if (layerEl && layerNames[currentStep]) {
                layerEl.textContent = layerNames[currentStep];
            }

            // Update confidence display
            if (confidenceEl) {
                confidenceEl.textContent = (layerProgress[currentStep] * 100).toFixed(1) + '%';
            }

            currentStep++;

            AppState.animationTimeouts[modelName] = setTimeout(animateStep, stepDuration);
        }

        // Start animation with staggered delay
        AppState.animationTimeouts[modelName] = setTimeout(animateStep, index * 100);
    });
}

// ========================================
// PIE CHART (using uPlot or Canvas)
// ========================================
function displayPieChart(predictions) {
    const container = document.getElementById('pieChart');
    if (!container) return;

    // Sort by probability
    const sorted = [...predictions].sort((a, b) => b.probability - a.probability);

    // Create pie chart using Canvas
    const canvas = container;
    const ctx = canvas.getContext('2d');
    const size = Math.min(canvas.width, canvas.height);
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const radius = size * 0.4;

    // Colors for each class
    const colors = {
        'Glioma': '#EF4444',
        'Meningioma': '#F59E0B',
        'No Tumor': '#10B981',
        'Pituitary': '#8B5CF6'
    };

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw pie slices
    let startAngle = -Math.PI / 2; // Start from top

    sorted.forEach(pred => {
        const sliceAngle = pred.probability * 2 * Math.PI;

        ctx.beginPath();
        ctx.moveTo(centerX, centerY);
        ctx.arc(centerX, centerY, radius, startAngle, startAngle + sliceAngle);
        ctx.closePath();

        ctx.fillStyle = colors[pred.class] || '#666666';
        ctx.fill();

        // Draw slice border
        ctx.strokeStyle = '#1a1a1a';
        ctx.lineWidth = 2;
        ctx.stroke();

        startAngle += sliceAngle;
    });

    // Draw center circle (donut hole)
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius * 0.6, 0, 2 * Math.PI);
    ctx.fillStyle = '#1a1a1a';
    ctx.fill();
}

function displayFinalResult(result) {
    const classEl = document.getElementById('resultClass');
    const confEl = document.getElementById('resultConfidence');

    if (classEl) {
        classEl.textContent = result.class;
    }

    if (confEl) {
        confEl.textContent = (result.confidence * 100).toFixed(1) + '%';
    }
}

function displayProbabilityBars(predictions) {
    const container = document.getElementById('probabilityBars');
    if (!container) return;

    // Sort by probability
    const sorted = [...predictions].sort((a, b) => b.probability - a.probability);

    container.innerHTML = sorted.map(pred => `
        <div class="probability-item">
            <span class="probability-label">${pred.class}</span>
            <div class="probability-bar">
                <div class="probability-fill" style="width: ${pred.probability * 100}%"></div>
            </div>
            <span class="probability-value">${(pred.probability * 100).toFixed(1)}%</span>
        </div>
    `).join('');
}

// ========================================
// PANEL COLLAPSE
// ========================================
function initPanelCollapse() {
    document.querySelectorAll('.panel-header').forEach(header => {
        header.addEventListener('click', (e) => {
            // Don't collapse when clicking drag handle
            if (e.target.closest('.drag-handle')) return;

            const panel = header.closest('.panel');
            panel.classList.toggle('collapsed');

            // Save state
            savePanelState();
        });
    });
}

function savePanelState() {
    const panelStates = {};
    document.querySelectorAll('.panel').forEach(panel => {
        panelStates[panel.id] = panel.classList.contains('collapsed');
    });
    localStorage.setItem('brainTumorPanelStates', JSON.stringify(panelStates));
}

// ========================================
// DRAG AND DROP GRID
// ========================================
function initDragAndDrop() {
    const panels = document.querySelectorAll('.panel[draggable]');

    panels.forEach(panel => {
        panel.addEventListener('dragstart', handleDragStart);
        panel.addEventListener('dragend', handleDragEnd);
        panel.addEventListener('dragover', handleDragOver);
        panel.addEventListener('dragenter', handleDragEnter);
        panel.addEventListener('dragleave', handleDragLeave);
        panel.addEventListener('drop', handleDrop);
    });
}

let draggedPanel = null;

function handleDragStart(e) {
    if (!AppState.isEditMode) {
        e.preventDefault();
        return;
    }

    draggedPanel = this;
    this.classList.add('dragging');
    e.dataTransfer.effectAllowed = 'move';
    e.dataTransfer.setData('text/html', this.innerHTML);
}

function handleDragEnd(e) {
    this.classList.remove('dragging');
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('drag-over'));
    draggedPanel = null;
}

function handleDragOver(e) {
    if (!AppState.isEditMode) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
}

function handleDragEnter(e) {
    if (!AppState.isEditMode) return;
    this.classList.add('drag-over');
}

function handleDragLeave(e) {
    this.classList.remove('drag-over');
}

function handleDrop(e) {
    if (!AppState.isEditMode) return;
    e.stopPropagation();

    if (draggedPanel !== this) {
        // Swap positions
        const grid = DOM.panelGrid;
        const panels = Array.from(grid.children);
        const draggedIndex = panels.indexOf(draggedPanel);
        const targetIndex = panels.indexOf(this);

        if (draggedIndex < targetIndex) {
            grid.insertBefore(draggedPanel, this.nextSibling);
        } else {
            grid.insertBefore(draggedPanel, this);
        }

        savePanelOrder();
    }

    this.classList.remove('drag-over');
    return false;
}

function savePanelOrder() {
    const order = Array.from(DOM.panelGrid.children).map(p => p.id);
    localStorage.setItem('brainTumorPanelOrder', JSON.stringify(order));
}

function loadPanelOrder() {
    const savedOrder = localStorage.getItem('brainTumorPanelOrder');
    if (!savedOrder) return;

    try {
        const order = JSON.parse(savedOrder);
        const grid = DOM.panelGrid;

        order.forEach(id => {
            const panel = document.getElementById(id);
            if (panel) {
                grid.appendChild(panel);
            }
        });
    } catch (e) {
        console.warn('Failed to load panel order:', e);
    }
}

// ========================================
// EDIT MODE
// ========================================
function toggleEditMode() {
    AppState.isEditMode = !AppState.isEditMode;

    document.body.classList.toggle('edit-mode', AppState.isEditMode);

    if (DOM.editModeBtn) {
        DOM.editModeBtn.classList.toggle('active', AppState.isEditMode);
    }

    // Enable/disable dragging
    document.querySelectorAll('.panel').forEach(panel => {
        panel.setAttribute('draggable', AppState.isEditMode);
    });
}

// ========================================
// FEEDBACK
// ========================================
async function submitFeedback(isCorrect) {
    if (!AppState.currentData) return;

    const data = AppState.currentData;
    let trueLabel = data.final_result.class;

    if (!isCorrect) {
        const select = document.getElementById('correctionSelect');
        if (select) {
            trueLabel = select.value;
        }
    }

    try {
        await fetch('/api/feedback', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                filename: data.filename,
                predicted_label: data.final_result.class,
                true_label: trueLabel,
                confidence: data.final_result.confidence,
                timestamp: new Date().toISOString(),
                model_version: data.model_version
            })
        });

        showFeedbackSuccess();

    } catch (err) {
        console.error('Feedback error:', err);
    }
}

function showCorrectionDropdown() {
    const feedbackBtns = document.getElementById('feedbackButtons');
    const correctionSection = document.getElementById('correctionSection');

    if (feedbackBtns) feedbackBtns.classList.add('hidden');
    if (correctionSection) correctionSection.classList.remove('hidden');
}

function showFeedbackSuccess() {
    const feedbackBtns = document.getElementById('feedbackButtons');
    const correctionSection = document.getElementById('correctionSection');
    const successMsg = document.getElementById('feedbackSuccess');

    if (feedbackBtns) feedbackBtns.classList.add('hidden');
    if (correctionSection) correctionSection.classList.add('hidden');
    if (successMsg) successMsg.classList.remove('hidden');
}

function resetFeedbackUI() {
    const feedbackBtns = document.getElementById('feedbackButtons');
    const correctionSection = document.getElementById('correctionSection');
    const successMsg = document.getElementById('feedbackSuccess');
    const autoEvalEl = document.getElementById('autoEvalResult');

    if (feedbackBtns) feedbackBtns.classList.remove('hidden');
    if (correctionSection) correctionSection.classList.add('hidden');
    if (successMsg) successMsg.classList.add('hidden');
    if (autoEvalEl) autoEvalEl.classList.add('hidden');
}

// ========================================
// AUTO-EVALUATION (for random test images)
// ========================================
function displayAutoEval(autoEval) {
    const container = document.getElementById('autoEvalResult');
    if (!container) return;

    if (!autoEval) {
        container.classList.add('hidden');
        return;
    }

    const isCorrect = autoEval.is_correct;
    const icon = isCorrect ? '[PASS]' : '[FAIL]';
    const statusClass = isCorrect ? 'correct' : 'incorrect';
    const statusText = isCorrect ? 'Correct!' : 'Incorrect';

    container.innerHTML = `
        <div class="auto-eval-badge ${statusClass}">
            <span class="auto-eval-icon">${icon}</span>
            <span class="auto-eval-status">${statusText}</span>
        </div>
        <div class="auto-eval-details">
            <span class="auto-eval-label">True Label:</span>
            <span class="auto-eval-value">${autoEval.true_label}</span>
        </div>
        ${!isCorrect ? `
        <div class="auto-eval-details">
            <span class="auto-eval-label">Predicted:</span>
            <span class="auto-eval-value">${autoEval.predicted_label}</span>
        </div>
        ` : ''}
    `;
    container.classList.remove('hidden');

    // Update running statistics
    updateAutoEvalStats(isCorrect);
}

// Track running statistics for auto-eval
const AutoEvalStats = {
    total: 0,
    correct: 0
};

function updateAutoEvalStats(isCorrect) {
    AutoEvalStats.total++;
    if (isCorrect) AutoEvalStats.correct++;

    const statsEl = document.getElementById('autoEvalStats');
    if (statsEl) {
        const accuracy = AutoEvalStats.total > 0
            ? ((AutoEvalStats.correct / AutoEvalStats.total) * 100).toFixed(1)
            : 0;
        statsEl.innerHTML = `
            <span class="stats-label">Session Stats:</span>
            <span class="stats-value">${AutoEvalStats.correct}/${AutoEvalStats.total} correct (${accuracy}%)</span>
        `;
        statsEl.classList.remove('hidden');
    }
}

// ========================================
// UTILITIES
// ========================================
function showLoading(show) {
    AppState.isAnalyzing = show;
    // Use cached reference or get directly
    const overlay = DOM.loadingOverlay || document.getElementById('loadingOverlay');
    if (overlay) {
        if (show) {
            overlay.classList.add('active');
        } else {
            overlay.classList.remove('active');
        }
    }
}

function showError(message) {
    const errorEl = document.getElementById('errorMessage');
    if (errorEl) {
        errorEl.textContent = message;
        errorEl.classList.remove('hidden');

        setTimeout(() => {
            errorEl.classList.add('hidden');
        }, 5000);
    }
    console.error(message);
}

// ========================================
// GRADCAM COMPARISON
// ========================================
async function toggleGradcamCompare() {
    const standardView = document.getElementById('standardView');
    const compareView = document.getElementById('gradcamCompareView');
    const compareBtn = document.getElementById('compareGradcamsBtn');

    if (!standardView || !compareView) return;

    const isCompareMode = compareView.classList.contains('hidden');

    if (isCompareMode) {
        // Switch to compare mode
        standardView.classList.add('hidden');
        compareView.classList.remove('hidden');
        if (compareBtn) compareBtn.textContent = 'Standard View';

        // Fetch comparison data if we have current data
        if (AppState.currentData && AppState.currentData.filename) {
            await fetchGradcamComparison();
        }
    } else {
        // Switch back to standard view
        standardView.classList.remove('hidden');
        compareView.classList.add('hidden');
        if (compareBtn) compareBtn.textContent = 'Compare Models';
    }
}

async function fetchGradcamComparison() {
    if (!AppState.currentData) return;

    const grid = document.querySelector('.gradcam-grid');
    const consistencyEl = document.getElementById('attentionConsistency');

    if (grid) {
        grid.innerHTML = '<div class="loading-text">Generating GradCAM comparisons...</div>';
    }

    try {
        const formData = new FormData();

        // If we have the original image data, send it
        if (AppState.currentData.original_b64) {
            // Convert base64 to blob
            const byteString = atob(AppState.currentData.original_b64);
            const ab = new ArrayBuffer(byteString.length);
            const ia = new Uint8Array(ab);
            for (let i = 0; i < byteString.length; i++) {
                ia[i] = byteString.charCodeAt(i);
            }
            const blob = new Blob([ab], { type: 'image/png' });
            formData.append('file', blob, AppState.currentData.filename || 'image.png');
        }

        const response = await fetch('/api/compare-gradcams', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        if (!response.ok || data.error) {
            throw new Error(data.error || 'Failed to compare GradCAMs');
        }

        displayGradcamComparison(data);

    } catch (err) {
        if (grid) {
            grid.innerHTML = `<div class="error-text">Error: ${err.message}</div>`;
        }
    }
}

function displayGradcamComparison(data) {
    const models = ['resnet18', 'efficientnet', 'densenet'];

    models.forEach(modelName => {
        const container = document.getElementById(`gradcam-${modelName}`);
        if (!container) return;

        const modelData = data.gradcams[modelName];
        if (!modelData) {
            container.innerHTML = '<span class="error-text">N/A</span>';
            return;
        }

        container.innerHTML = `
            <img src="data:image/png;base64,${modelData.heatmap_b64}" alt="${modelName} GradCAM">
            <div class="gradcam-label">
                <span class="model-name">${modelName.charAt(0).toUpperCase() + modelName.slice(1)}</span>
                <span class="model-pred">${modelData.prediction} (${(modelData.confidence * 100).toFixed(1)}%)</span>
            </div>
        `;
    });

    // Display consistency warning
    const consistencyEl = document.getElementById('attentionConsistency');
    if (consistencyEl && data.attention_analysis) {
        const analysis = data.attention_analysis;

        if (analysis.is_consistent) {
            consistencyEl.innerHTML = `
                <span class="consistency-good">[OK] Models agree on attention regions</span>
                <span class="consistency-score">Overlap: ${(analysis.overlap_score * 100).toFixed(0)}%</span>
            `;
            consistencyEl.className = 'attention-consistency good';
        } else {
            consistencyEl.innerHTML = `
                <span class="consistency-warning">[!] Models focus on different regions</span>
                <span class="consistency-score">Overlap: ${(analysis.overlap_score * 100).toFixed(0)}%</span>
                <span class="consistency-hint">Lower confidence recommended</span>
            `;
            consistencyEl.className = 'attention-consistency warning';
        }
        consistencyEl.classList.remove('hidden');
    }
}

// ========================================
// CONFIDENCE CALIBRATION
// ========================================
async function fetchConfidenceCalibration() {
    const content = document.getElementById('calibrationContent');
    if (!content || !AppState.currentData) return;

    content.innerHTML = '<div class="loading-text">Analyzing confidence calibration...</div>';

    try {
        const formData = new FormData();

        if (AppState.currentData.original_b64) {
            const byteString = atob(AppState.currentData.original_b64);
            const ab = new ArrayBuffer(byteString.length);
            const ia = new Uint8Array(ab);
            for (let i = 0; i < byteString.length; i++) {
                ia[i] = byteString.charCodeAt(i);
            }
            const blob = new Blob([ab], { type: 'image/png' });
            formData.append('file', blob, AppState.currentData.filename || 'image.png');
        }

        const response = await fetch('/api/confidence-calibration', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        if (!response.ok || data.error) {
            throw new Error(data.error || 'Failed to get calibration');
        }

        displayCalibration(data);

    } catch (err) {
        content.innerHTML = `<div class="error-text">Error: ${err.message}</div>`;
    }
}

function displayCalibration(data) {
    const content = document.getElementById('calibrationContent');
    if (!content) return;

    const entropyPercent = Math.min(data.entropy / 2 * 100, 100); // Normalize entropy (max ~2 for 4 classes)
    const agreementPercent = data.model_agreement * 100;

    let html = `
        <div class="calibration-grid">
            <div class="calibration-item">
                <span class="calibration-label">Prediction Entropy</span>
                <div class="calibration-bar">
                    <div class="calibration-fill entropy" style="width: ${entropyPercent}%"></div>
                </div>
                <span class="calibration-value">${data.entropy.toFixed(3)}</span>
                <span class="calibration-hint">${entropyPercent < 30 ? 'Low uncertainty' : entropyPercent < 60 ? 'Moderate uncertainty' : 'High uncertainty'}</span>
            </div>
            <div class="calibration-item">
                <span class="calibration-label">Model Agreement</span>
                <div class="calibration-bar">
                    <div class="calibration-fill agreement" style="width: ${agreementPercent}%"></div>
                </div>
                <span class="calibration-value">${agreementPercent.toFixed(0)}%</span>
                <span class="calibration-hint">${agreementPercent > 80 ? 'Strong consensus' : agreementPercent > 50 ? 'Partial agreement' : 'Models disagree'}</span>
            </div>
        </div>
    `;

    // Overconfidence warning
    if (data.overconfidence_risk) {
        html += `
            <div class="calibration-warning">
                <span class="warning-icon">[!]</span>
                <span class="warning-text">Potential overconfidence detected. Consider this prediction with caution.</span>
            </div>
        `;
    }

    // Recommendation
    html += `
        <div class="calibration-recommendation">
            <strong>Recommendation:</strong> ${data.recommendation}
        </div>
    `;

    content.innerHTML = html;
}

// Update displayResults to also fetch calibration
const originalDisplayResults = displayResults;
function displayResultsWithCalibration(data) {
    originalDisplayResults(data);

    // Auto-fetch calibration after results
    const calibrationPanel = document.getElementById('calibrationPanel');
    if (calibrationPanel && !calibrationPanel.classList.contains('collapsed')) {
        setTimeout(fetchConfidenceCalibration, 500);
    }
}
// Override
displayResults = displayResultsWithCalibration;

// ========================================
// EXPOSE FUNCTIONS TO HTML
// ========================================
window.submitFeedback = submitFeedback;
window.showCorrectionDropdown = showCorrectionDropdown;
window.loadRandomTest = loadRandomTest;
window.toggleEditMode = toggleEditMode;
window.toggleGradcamCompare = toggleGradcamCompare;
window.fetchConfidenceCalibration = fetchConfidenceCalibration;
