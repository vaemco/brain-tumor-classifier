/* Monitor Logic - Brain Tumor AI */

const REFRESH_RATE = 2000;
let charts = {}; // Store uPlot instances
let currentData = null;

document.addEventListener('DOMContentLoaded', () => {
    initCharts();
    setupTheme();
    setupDropzone();
    startMonitoring();
});

// --- Chart Initialization (uPlot) ---
function initCharts() {
    // 1. Latency Chart (Sparkline)
    charts.latency = createSparkline("chartLatency", "#58a6ff");
    // 2. Confidence Chart (Sparkline)
    charts.confidence = createSparkline("chartConfidence", "#238636");
    // 3. Scans Chart (Sparkline)
    charts.scans = createSparkline("chartScans", "#a371f7");

    // 4. Probability Chart (Bar chart simulation using uPlot or just DOM for simplicity? Plan said uPlot)
    // uPlot is better for time series. For categorical bars, let's use a simple HTML approach or uPlot bars if needed.
    // For simplicity and look, I will use DOM-based bars inside the 'probChart' container for now, as uPlot for 4 static bars is overkill/tricky.
    // However, I will adhere to "use uPlot" by making the history charts uPlot.
    // Let's stick to Sparklines for the bottom stats for now.
}

function createSparkline(id, color) {
    const el = document.getElementById(id);
    if(!el) return null;

    // Initial Data
    const data = [
        Array.from({length: 30}, (_, i) => i), // x
        Array(30).fill(0) // y
    ];

    const opts = {
        width: el.clientWidth,
        height: 40,
        class: "spark",
        cursor: { show: false },
        select: { show: false },
        legend: { show: false },
        scales: { x: { time: false }, y: { auto: true } },
        axes: [ { show: false }, { show: false } ],
        series: [
            {},
            {
                stroke: color,
                fill: color + "33", // hex alpha
                width: 2
            }
        ]
    };

    return new uPlot(opts, data, el);
}

// --- Main Interactions ---

async function runRandomTest() {
    setLoading(true);
    try {
        const res = await fetch('/api/random-test');
        const data = await res.json();
        handlePredictionResult(data);
    } catch (e) {
        console.error(e);
        alert("Error running test");
    } finally {
        setLoading(false);
    }
}

// File Upload
function setupDropzone() {
    const dz = document.getElementById('dropzone');
    const input = document.getElementById('fileInput');

    dz.onclick = () => input.click();

    input.onchange = (e) => {
        if (e.target.files[0]) uploadFile(e.target.files[0]);
    };

    dz.ondragover = (e) => { e.preventDefault(); dz.style.borderColor = "var(--primary)"; };
    dz.ondragleave = (e) => { e.preventDefault(); dz.style.borderColor = "var(--border)"; };
    dz.ondrop = (e) => {
        e.preventDefault();
        dz.style.borderColor = "var(--border)";
        if (e.dataTransfer.files[0]) uploadFile(e.dataTransfer.files[0]);
    };
}

async function uploadFile(file) {
    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
        const res = await fetch('/api/predict', { method: 'POST', body: formData });
        const data = await res.json();
        handlePredictionResult(data);
    } catch (e) {
        alert("Upload failed");
    } finally {
        setLoading(false);
    }
}

// --- Rendering Results ---

function handlePredictionResult(data) {
    currentData = data;

    // Images
    const imgOriginal = document.getElementById('imgOriginal');
    const imgHeatmap = document.getElementById('imgHeatmap');

    imgOriginal.src = `data:image/png;base64,${data.original}`;
    imgOriginal.classList.remove('placeholder-img');

    imgHeatmap.src = `data:image/png;base64,${data.gradcam}`;
    imgHeatmap.classList.remove('placeholder-img');

    // Bounding Box (if available)
    const bboxEl = document.getElementById('bboxOriginal');
    if (data.bbox) {
        bboxEl.style.display = 'block';
        bboxEl.style.left = data.bbox.x + '%';
        bboxEl.style.top = data.bbox.y + '%';
        bboxEl.style.width = data.bbox.width + '%';
        bboxEl.style.height = data.bbox.height + '%';
        bboxEl.style.border = '2px solid var(--danger)';
        bboxEl.style.position = 'absolute';
    } else {
        bboxEl.style.display = 'none';
    }

    // Text Stats
    const predEl = document.getElementById('primaryPred');
    predEl.querySelector('.pred-label').innerText = data.top_prediction.class;
    predEl.querySelector('.pred-conf').innerText = (data.top_prediction.probability * 100).toFixed(1) + '%';

    // Render Pie Chart
    renderProbPie(data.predictions);

    // Consensus
    if (data.consensus) {
        document.getElementById('consensusBox').style.display = 'block';
        document.getElementById('consensusScore').innerText = data.consensus.result.score;
        renderConsensusList(data.consensus.models);
    }

    // Similar Cases
    if (data.similar_cases) {
        renderSimilarCases(data.similar_cases);
    }

    // Add to history list
    addToMiniHistory(data.filename || "Upload");
}

function renderProbPie(preds) {
    const container = document.getElementById('probChart');
    if (!container) return;
    container.innerHTML = '';
    container.style.height = 'auto'; // Allow growing

    // Sort
    preds.sort((a,b) => b.probability - a.probability);

    // Prepare Colors
    // CSS Variables aren't directly available in JS calculation string easily without getComputedStyle
    // So we use hardcoded hex matches or getComputedStyle.
    // Let's use hex for simplicity that matches the theme.
    const colors = ['#58a6ff', '#238636', '#d29922', '#da3633', '#8b949e'];

    let gradient = [];
    let start = 0;

    preds.forEach((p, i) => {
        const pct = p.probability * 100;
        const end = start + pct;
        gradient.push(`${colors[i % colors.length]} ${start}% ${end}%`);
        start = end;
    });

    const wrapper = document.createElement('div');
    wrapper.style.display = 'flex';
    wrapper.style.alignItems = 'center';
    wrapper.style.justifyContent = 'center';
    wrapper.style.gap = '20px';
    wrapper.style.marginTop = '10px';

    // Pie Circle
    const pie = document.createElement('div');
    pie.style.width = '100px';
    pie.style.height = '100px';
    pie.style.borderRadius = '50%';
    pie.style.background = `conic-gradient(${gradient.join(', ')})`;
    pie.style.flexShrink = '0';
    pie.style.boxShadow = '0 0 10px rgba(0,0,0,0.2)';

    // Legend
    const legend = document.createElement('div');
    legend.style.display = 'flex';
    legend.style.flexDirection = 'column';
    legend.style.gap = '6px';
    legend.style.fontSize = '12px';

    preds.forEach((p, i) => {
        const row = document.createElement('div');
        row.style.display = 'flex';
        row.style.alignItems = 'center';
        row.style.gap = '8px';
        const color = colors[i % colors.length];

        row.innerHTML = `
            <span style="width:8px; height:8px; border-radius:50%; background:${color}"></span>
            <span style="color:var(--text);">${p.class}</span>
            <span style="opacity:0.7; font-weight:600;">${(p.probability * 100).toFixed(1)}%</span>
        `;
        legend.appendChild(row);
    });

    wrapper.appendChild(pie);
    wrapper.appendChild(legend);
    container.appendChild(wrapper);
}

function renderConsensusList(models) {
    const con = document.getElementById('modelVotesMini');
    if (!con) return;
    con.innerHTML = '';
    models.forEach(m => {
        const div = document.createElement('div');
        div.className = 'cons-item';
        div.innerHTML = `
            <div style="display:flex; justify-content:space-between;">
                <b>${m.name}</b>
                <span style="opacity:0.7">${(m.confidence * 100).toFixed(0)}% • ${m.prediction}</span>
            </div>
            <div class="cons-bar-bg">
                <div class="cons-bar-fill" style="width: ${m.confidence * 100}%; background-color: ${getConfColor(m.confidence)}"></div>
            </div>
        `;
        con.appendChild(div);
    });
}

function getConfColor(conf) {
    if (conf > 0.8) return 'var(--success)';
    if (conf > 0.6) return 'var(--warning)';
    return 'var(--danger)';
}

function renderSimilarCases(cases) {
    const container = document.getElementById('similarCasesContainer');
    if (!container) return;
    container.innerHTML = '';

    cases.forEach(c => {
        const div = document.createElement('div');
        div.className = 'sim-card';
        div.innerHTML = `
            <img src="${c.image || '/static/img/placeholder_brain.png'}" class="sim-img" alt="Case">
            <div class="sim-info">
                <span class="sim-match">${(c.similarity * 100).toFixed(0)}% Match</span>
                <span class="sim-meta">${c.label} • ${c.id}</span>
            </div>
            <i data-feather="arrow-right" style="width:14px; opacity:0.5"></i>
        `;
        container.appendChild(div);
    });
    if (window.feather) feather.replace();
}

// --- Status Monitoring ---

function startMonitoring() {
    setInterval(updateStats, REFRESH_RATE);
}

async function updateStats() {
    try {
        const res = await fetch('/api/metrics');
        const m = await res.json();

        // Safety checks for new server / empty history
        const latVal = (m.inference_times && m.inference_times.resnet18 && m.inference_times.resnet18.length > 0)
            ? m.inference_times.resnet18.slice(-1)[0]
            : 0;

        // Update values
        updateSparkline(charts.latency, parseFloat(latVal));
        updateSparkline(charts.confidence, m.average_confidence * 100);
        updateSparkline(charts.scans, m.predictions.total);

        document.getElementById('valLatency').innerText = parseFloat(latVal).toFixed(0) + 'ms';
        document.getElementById('valConfidence').innerText = (m.average_confidence * 100).toFixed(1) + '%';
        document.getElementById('valScans').innerText = m.predictions.total;

        const cStatus = (m.consensus && m.consensus.low_consensus > 0) ? 'Low' : 'High';
        document.getElementById('valConsensus').innerText = cStatus;
        document.getElementById('consensusRing').style.borderColor = cStatus === 'High' ? 'var(--success)' : 'var(--warning)';

    } catch (e) {
        console.error("Stats update failed", e);
    }
}

function updateSparkline(chart, newVal) {
    if (!chart) return;
    const data = chart.data; // [x[], y[]]

    // Shift
    const x = data[0].map(v => v);
    const y = data[1].map(v => v);

    y.shift();
    y.push(newVal);

    chart.setData([x, y]);
}

// --- UI Helpers ---

function setLoading(active) {
    const loader = document.getElementById('mainLoader');
    if (active) loader.classList.add('active');
    else loader.classList.remove('active');
}

function addToMiniHistory(name) {
    const list = document.getElementById('miniHistoryList');
    const el = document.createElement('div');
    el.innerText = name;
    el.style.fontSize = '12px';
    el.style.padding = '4px 0';
    el.style.borderBottom = '1px solid var(--border-light)';
    list.prepend(el);
    if (list.children.length > 5) list.lastChild.remove();
}

function toggleTheme() {
    const html = document.documentElement;
    const current = html.getAttribute('data-theme');
    const next = current === 'dark' ? 'light' : 'dark';
    html.setAttribute('data-theme', next);

    const icon = document.querySelector('#themeBtn i');
    if(next==='light') feather.replace({ 'moon': 'sun' }); // Re-render needed if changing icon name, but feather replaces <i> so slightly tricky.
    // Simplest: just keep moon icon for now or reload feather.
}
