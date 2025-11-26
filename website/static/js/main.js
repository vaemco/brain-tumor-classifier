// Brain Tumor Classifier - Frontend Logic

const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const results = document.getElementById('results');
const loading = document.getElementById('loading');
const error = document.getElementById('error');

let pieChart = null;

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
        displayResults(data);

    } catch (err) {
        showError('Fehler bei der Analyse: ' + err.message);
    } finally {
        loading.style.display = 'none';
    }
}

// Display prediction results
function displayResults(data) {
    const { predictions, top_prediction, gradcam, original } = data;

    // Top prediction
    const predictionHTML = `
        <div class="prediction-class">${top_prediction.class}</div>
        <div class="prediction-probability">${(top_prediction.probability * 100).toFixed(1)}%</div>
    `;
    document.getElementById('predictionResult').innerHTML = predictionHTML;

    // Pie chart
    createPieChart(predictions);

    // Grad-CAM images
    document.getElementById('originalImage').src = 'data:image/png;base64,' + original;
    document.getElementById('heatmapImage').src = 'data:image/png;base64,' + gradcam;

    // Probability bars
    createProbabilityBars(predictions);

    // Show results
    results.style.display = 'block';
    results.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
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
                borderColor: '#1e293b'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        color: '#f1f5f9',
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
    } catch (err) {
        console.error('Failed to load device info');
    }
}

// Initialize
loadDeviceInfo();
