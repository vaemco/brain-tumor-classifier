#!/bin/bash
# Brain Tumor Demo - Setup Script (Linux/macOS)
# Run: chmod +x setup.sh && ./setup.sh

set -e

echo "================================================"
echo "Brain Tumor Classifier Demo - Setup"
echo "================================================"

DEMO_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$DEMO_DIR")"

# Create directories
echo -e "\n[1/3] Creating directories..."
mkdir -p "$DEMO_DIR/model"
mkdir -p "$DEMO_DIR/samples/glioma"
mkdir -p "$DEMO_DIR/samples/meningioma"
mkdir -p "$DEMO_DIR/samples/notumor"
mkdir -p "$DEMO_DIR/samples/pituitary"
echo "  Done"

# Copy model
echo -e "\n[2/3] Copying model..."
MODEL_SRC="$PROJECT_DIR/models/brain_tumor_efficientnet_b0_v2_trained.pt"
MODEL_DST="$DEMO_DIR/model/brain_tumor_efficientnet_b0.pt"

if [ -f "$MODEL_SRC" ]; then
    cp "$MODEL_SRC" "$MODEL_DST"
    SIZE=$(du -h "$MODEL_DST" | cut -f1)
    echo "  Copied: $SIZE"
else
    echo "  WARNING: Model not found at $MODEL_SRC"
    echo "  Train the model first or copy it manually."
fi

# Copy sample images
echo -e "\n[3/3] Copying sample images..."
TESTING_DIR="$PROJECT_DIR/data/Brain_Tumor_Dataset/Testing"
SAMPLES_PER_CLASS=5

for class in glioma meningioma notumor pituitary; do
    SRC_DIR="$TESTING_DIR/$class"
    DST_DIR="$DEMO_DIR/samples/$class"

    if [ -d "$SRC_DIR" ]; then
        count=$(ls "$SRC_DIR"/*.jpg 2>/dev/null | head -n $SAMPLES_PER_CLASS | wc -l)
        ls "$SRC_DIR"/*.jpg 2>/dev/null | head -n $SAMPLES_PER_CLASS | xargs -I {} cp {} "$DST_DIR/"
        echo "  $class: $count images"
    else
        echo "  $class: Source not found"
    fi
done

# Summary
echo -e "\n================================================"
echo "Setup Complete!"
echo "================================================"

echo -e "\nNext steps:"
echo "  1. cd $DEMO_DIR"
echo "  2. docker-compose up -d"
echo "  3. Open http://localhost:5000"

echo -e "\nOr run without Docker:"
echo "  1. pip install -r requirements.txt"
echo "  2. python app.py"
