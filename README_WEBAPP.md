# Brain Tumor Classifier - Web App

## 🌐 Web Application Setup

Diese Web-App bietet eine moderne Oberfläche für die Brain Tumor Klassifizierung mit:
- **Drag & Drop** Upload
- **Pie Chart** Visualisierung
- **Grad-CAM Heatmaps** (zeigt wo das Modell "hinschaut")
- **Echtzeit-Analyse**

## 🚀 Installation

1. **Dependencies installieren:**
```bash
mamba env update -f environment.yml
mamba activate data_brain
```

2. **Web-Server starten:**
```bash
cd /Users/valentinemser/dev_projects/03_data_projects/data_brain_tumor
python -m website.app
```

3. **Browser öffnen:**
```
http://localhost:5000
```

## 💡 Nutzung

1. Ziehe ein MRI-Bild in die Dropzone ODER klicke zum Auswählen
2. Warte auf die Analyse (~2-3 Sekunden)
3. Siehe Ergebnisse:
   - **Top Prediction** mit Wahrscheinlichkeit
   - **Pie Chart** mit allen Klassen
   - **Grad-CAM Heatmap** - zeigt relevante Bildbereiche
   - **Detaillierte Wahrscheinlichkeiten** als Balken

## 🎨 Features

### Grad-CAM Visualisierung
Grad-CAM (Gradient-weighted Class Activation Mapping) zeigt, welche Bereiche des MRI-Bildes am wichtigsten für die Entscheidung des Modells waren.

- **Original**: Das hochgeladene Bild
- **Heatmap**: Überlagerung mit wichtigsten Bereichen (rot = wichtig)

### API Endpoints

- `GET /` - Hauptseite
- `POST /api/predict` - Bild hochladen & analysieren
- `GET /api/health` - System Status

## 📊 Maximale Genauigkeit erreichen

### 1. Data Augmentation (bereits implementiert)
- Random Horizontal Flip
- Random Rotation
- Color Jitter
- Random Resized Crop

### 2. Test-Time Augmentation (TTA)
Für noch bessere Ergebnisse kann man TTA verwenden:
```python
# Mehrere Augmentierungen, dann Durchschnitt
predictions = []
for _ in range(5):
    aug_image = augment(image)
    pred = model(aug_image)
    predictions.append(pred)
average_pred = torch.stack(predictions).mean(0)
```

### 3. Ensemble Methods
Mehrere Modelle trainieren und kombinieren:
- ResNet18 (schnell, aktuell)
- ResNet50 (genauer, langsamer)
- EfficientNet
- → Durchschnitt der Predictions

### 4. Längeres Training
- Mehr Epochen (50-100)
- Learning Rate Scheduling
- Cross-Validation

### 5. Größeres Modell
```python
# Statt ResNet18 → ResNet50
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
```

## ❓ Zusätzliches Neuronales Netzwerk?

**NICHT empfohlen.** Stattdessen:

### ✅ Besser: Model Improvement
1. **Bessere Architektur**: ResNet50/101, EfficientNet
2. **Transfer Learning**: Von medizinischen Datasets (z.B. RadImageNet)
3. **Data Augmentation**: Mehr Variationen
4. **Ensembles**: Mehrere Modelle kombinieren

### ❌ Schlechter: Zusätzliches Netzwerk
- Komplexität steigt massiv
- Training wird schwieriger
- Marginal bessere Ergebnisse
- Nicht wartbar

## 🎯 Empfohlene Optimierungen

### Priorisierung (1 = höchste):

1. **Test-Time Augmentation** (TTA)
   - Einfach zu implementieren
   - 2-5% Verbesserung
   - Implementierung siehe oben

2. **Längeres Training + Early Stopping**
   - 50-100 Epochen
   - Learning Rate Scheduler
   - 3-7% Verbesserung

3. **Ensemble (3 Modelle)**
   - ResNet18, ResNet34, ResNet50
   - 5-10% Verbesserung
   - Mehr Rechenzeit

4. **Größeres Base Model**
   - ResNet50 oder EfficientNet-B3
   - 5-8% Verbesserung
   - Längeres Training

## 🔧 Troubleshooting

### Port bereits in Benutzung
```bash
# Port ändern in app.py:
app.run(debug=True, host='0.0.0.0', port=5001)
```

### Modell nicht gefunden
```bash
# Erst trainieren:
jupyter notebook
# → train_m2_macbook.ipynb ausführen
```

### Grad-CAM Installation
```bash
pip install grad-cam
```

## 📈 Performance

- **Inference Zeit**: ~300ms auf M2
- **Grad-CAM**: ~400ms zusätzlich
- **Total**: ~700ms pro Bild

## 🎨 UI Customization

CSS in `website/static/css/style.css` anpassen für:
- Farben (:root Variablen)
- Layout
- Animationen

---

**Ready to deploy!** 🚀 Starte den Server und analysiere MRI-Bilder.
