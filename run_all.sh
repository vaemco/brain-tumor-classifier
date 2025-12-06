#!/bin/bash
# Unified launcher for Brain Tumor Classifier
# Starts both Flask Website and Gradio Dashboard

echo "🚀 Brain Tumor Classifier - Unified Launcher"
echo "============================================"
echo ""
echo "Services:"
echo "  🌐 Main Website      → http://localhost:3000"
echo "  📊 Monitor Dashboard → http://localhost:3000/monitor"
echo ""
echo "Starting application..."
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping all services..."
    kill $FLASK_PID $GRADIO_PID 2>/dev/null
    wait $FLASK_PID $GRADIO_PID 2>/dev/null
    echo "✓ All services stopped"
    exit 0
}

# Set trap to catch Ctrl+C
trap cleanup INT TERM

# Start Flask app in background
echo "▶ Starting Flask Website..."
python3 -m website.app &
FLASK_PID=$!
sleep 2

echo ""
echo "✅ Application is running!"
echo ""
echo "Services:"
echo "  Flask:  PID $FLASK_PID"
echo ""
echo "URLs:"
echo "  🌐 Website:  http://localhost:3000"
echo "  📊 Dashboard: http://localhost:3000/monitor"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for process
wait $FLASK_PID
