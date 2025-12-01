import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import gradio as gr
import requests
import time
from datetime import datetime
import numpy as np
from PIL import Image
import io
import threading

FLASK_API_URL = "http://localhost:3000/api/metrics"

def fetch_metrics():
    try:
        response = requests.get(FLASK_API_URL, timeout=2)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def create_simple_chart(data, chart_type):
    fig, ax = plt.subplots(figsize=(6, 3.5), facecolor='white')

    if chart_type == "predictions":
        models = ["ResNet18", "EfficientNet", "DenseNet"]
        counts = [
            data["predictions"].get("resnet18", 0),
            data["predictions"].get("efficientnet", 0),
            data["predictions"].get("densenet", 0),
        ]
        bars = ax.bar(models, counts, color='#cccccc', edgecolor='black', linewidth=1)
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height, f'{int(count)}',
                    ha='center', va='bottom', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title('Predictions per Model', fontsize=11)

    elif chart_type == "confidence":
        if data["confidence_scores"]:
            scores = data["confidence_scores"][-20:]
            ax.plot(range(len(scores)), scores, color='black', linewidth=1.5, marker='.')
            mean_score = np.mean(scores)
            ax.axhline(y=mean_score, color='#666666', linestyle='--', linewidth=1,
                       label=f'Mean: {mean_score:.2%}')
            ax.set_xlabel('Prediction #', fontsize=10)
            ax.set_ylabel('Confidence', fontsize=10)
            ax.set_ylim(0, 1)
            ax.legend(fontsize=9)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=12,
                    transform=ax.transAxes)
        ax.set_title('Confidence Timeline', fontsize=11)

    ax.grid(alpha=0.2, color='#cccccc')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=90, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    chart = Image.open(buf)
    plt.close(fig)
    return chart

def update_all():
    data = fetch_metrics()

    if data is None:
        empty = Image.new('RGB', (400, 200), color='white')
        return (
            "Connection Error", "N/A", "N/A", "N/A", "N/A",
            empty, empty,
            "API not reachable on port 3000"
        )

    # Stats
    uptime = format_time(data["uptime_seconds"])
    total = data["predictions"]["total"]
    avg_conf = data["average_confidence"]

    # Consensus
    c = data["consensus"]
    c_total = sum(c.values())
    full = c["full_agreement"]
    majority = c["majority"]
    low = c["low_consensus"]

    # Recent predictions
    recent_text = "Recent Predictions:\n\n"
    if data["recent_predictions"]:
        recent_text += "Time       | Prediction  | Conf  | Status\n"
        recent_text += "-----------|-------------|-------|----------------\n"
        for pred in reversed(data["recent_predictions"][-6:]):
            ts = datetime.fromtimestamp(pred["timestamp"]).strftime("%H:%M:%S")
            recent_text += f"{ts}    | {pred['prediction'][:11]:<11} | {pred['confidence']:.0%}  | {pred['consensus_status']}\n"
    else:
        recent_text += "No predictions yet"

    recent_text += f"\n\nLast update: {datetime.now().strftime('%H:%M:%S')}"

    # Charts
    chart_pred = create_simple_chart(data, "predictions")
    chart_conf = create_simple_chart(data, "confidence")

    return (
        uptime,
        str(total),
        f"{avg_conf:.1%}",
        f"{full} / {c_total}",
        f"{majority} / {low}",
        chart_pred,
        chart_conf,
        recent_text
    )

# Minimal Dashboard
with gr.Blocks(title="Model Monitor", theme=gr.themes.Monochrome()) as demo:
    gr.Markdown("# Model Monitoring Dashboard")
    gr.Markdown("Real-time vital statistics from Flask API - Click 'Refresh' to update")

    # Refresh button
    refresh_btn = gr.Button("Refresh Data", variant="primary")

    # Stats Grid
    with gr.Row():
        with gr.Column():
            gr.Markdown("**Uptime**")
            stat_uptime = gr.Textbox(value="00:00:00", interactive=False, show_label=False)
        with gr.Column():
            gr.Markdown("**Total Predictions**")
            stat_total = gr.Textbox(value="0", interactive=False, show_label=False)
        with gr.Column():
            gr.Markdown("**Avg Confidence**")
            stat_conf = gr.Textbox(value="0%", interactive=False, show_label=False)
        with gr.Column():
            gr.Markdown("**Full Agreement**")
            stat_full = gr.Textbox(value="0 / 0", interactive=False, show_label=False)
        with gr.Column():
            gr.Markdown("**Majority / Low**")
            stat_consensus = gr.Textbox(value="0 / 0", interactive=False, show_label=False)

    # Charts
    gr.Markdown("---")
    with gr.Row():
        chart_predictions = gr.Image(show_label=False)
        chart_confidence = gr.Image(show_label=False)

    # Recent predictions
    gr.Markdown("---")
    recent_box = gr.Textbox(lines=10, interactive=False, show_label=False)

    # Load initial data
    demo.load(
        fn=update_all,
        inputs=None,
        outputs=[stat_uptime, stat_total, stat_conf, stat_full, stat_consensus,
                 chart_predictions, chart_confidence, recent_box]
    )

    # Refresh button click
    refresh_btn.click(
        fn=update_all,
        inputs=None,
        outputs=[stat_uptime, stat_total, stat_conf, stat_full, stat_consensus,
                 chart_predictions, chart_confidence, recent_box]
    )

if __name__ == "__main__":
    print("Starting Model Monitor Dashboard...")
    print(f"API: {FLASK_API_URL}")
    print("Dashboard: http://localhost:7860")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
