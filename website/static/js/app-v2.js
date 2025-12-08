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
    animationFrames: {}
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

    showLoading(true);

    try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch('/api/analyze-detailed', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Analysis failed');
        }

        const data = await response.json();
        if (data.error) {
            throw new Error(data.error);
        }

        AppState.currentData = data;
        displayResults(data);

    } catch (err) {
        showError('Error during analysis: ' + err.message);
    } finally {
        showLoading(false);
    }
}

// ========================================
// RANDOM TEST
// ========================================
async function loadRandomTest() {
    showLoading(true);

    try {
        const response = await fetch('/api/random-test-detailed');

        if (!response.ok) {
            throw new Error('Failed to load test image');
        }

        const data = await response.json();
        if (data.error) {
            throw new Error(data.error);
        }

        AppState.currentData = data;
        displayResults(data);

    } catch (err) {
        showError('Error loading test image: ' + err.message);
    } finally {
        showLoading(false);
    }
}

// ========================================
// DISPLAY RESULTS
// ========================================
function displayResults(data) {
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

        // Clear previous chart
        container.innerHTML = '';

        // Destroy previous animation
        if (AppState.animationFrames[modelName]) {
            cancelAnimationFrame(AppState.animationFrames[modelName]);
        }

        // Animate the confidence build-up
        const layerProgress = modelData.layer_progress;
        const layerNames = modelData.layer_names;
        const finalConfidence = modelData.confidence;
        const color = modelData.color;

        let currentStep = 0;
        const totalSteps = layerProgress.length;
        const stepDuration = 150; // ms per step

        // Create uPlot chart
        const chartData = [
            [0],  // x-axis (layer index)
            [0]   // y-axis (confidence)
        ];

        const opts = {
            width: container.clientWidth,
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

            AppState.animationFrames[modelName] = requestAnimationFrame(() => {
                setTimeout(animateStep, stepDuration);
            });
        }

        // Start animation with staggered delay
        setTimeout(animateStep, index * 100);
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

    if (feedbackBtns) feedbackBtns.classList.remove('hidden');
    if (correctionSection) correctionSection.classList.add('hidden');
    if (successMsg) successMsg.classList.add('hidden');
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
// EXPOSE FUNCTIONS TO HTML
// ========================================
window.submitFeedback = submitFeedback;
window.showCorrectionDropdown = showCorrectionDropdown;
window.loadRandomTest = loadRandomTest;
window.toggleEditMode = toggleEditMode;
