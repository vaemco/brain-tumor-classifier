# 🧠 Brain Tumor Classifier - Demo Walkthrough

> **Portfolio Demo Guide** - Schritt-für-Schritt Erklärung aller Features

---

## 📋 Übersicht

Diese Demo zeigt einen **KI-gestützten Hirntumor-Klassifikator** basierend auf MRT-Bildern.
Das System unterscheidet zwischen 4 Klassen:

| Klasse         | Beschreibung         | Schweregrad |
| -------------- | -------------------- | ----------- |
| **Glioma**     | Tumor aus Gliazellen | Hoch        |
| **Meningioma** | Tumor der Hirnhäute  | Mittel      |
| **Pituitary**  | Tumor der Hypophyse  | Mittel      |
| **No Tumor**   | Normales MRT         | Keiner      |

---

## 🚀 Demo-Ablauf

### Phase 1: Bild-Upload

```
┌─────────────────────────────────────────┐
│  👤 User Action                          │
│  ─────────────────                       │
│  - Klickt auf Upload-Bereich             │
│  - Wählt MRT-Bild aus                    │
│  - ODER: Drag & Drop                     │
│  - ODER: Nutzt Sample-Bild aus API       │
└─────────────────────────────────────────┘
```

**Was passiert technisch:**

1. Bild wird als `multipart/form-data` an `/api/predict/explain` gesendet
2. Server empfängt und validiert das Bild
3. Konvertierung zu RGB-Format mit PIL

---

### Phase 2: Preprocessing Pipeline (Bildvorverarbeitung)

> **Warum zeigen wir das?** Um zu demonstrieren, dass ML nicht "magisch" ist,
> sondern systematische Datenaufbereitung benötigt.

```
┌──────────────────────────────────────────────────────────────────────┐
│                      PREPROCESSING PIPELINE                          │
│                                                                      │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌───────────┐        │
│   │ ORIGINAL│ -> │ RESIZE  │ -> │  CROP   │ -> │ GRAYSCALE │ ->     │
│   │         │    │ 256x256 │    │ 224x224 │    │  3-Kanal  │        │
│   └─────────┘    └─────────┘    └─────────┘    └───────────┘        │
│                                                       │              │
│                                                       v              │
│                                              ┌────────────┐          │
│                                              │ NORMALIZE  │          │
│                                              │ ImageNet   │          │
│                                              │ Statistik  │          │
│                                              └────────────┘          │
└──────────────────────────────────────────────────────────────────────┘
```

#### Schritt 1: Resize (256×256)

```python
transforms.Resize((256, 256))
```

**Erklärung für Portfolio:**

- Neuronale Netze brauchen einheitliche Eingabegrößen
- Größere Bilder = mehr Speicher & Rechenzeit
- 256px ist guter Kompromiss zwischen Detail und Effizienz

#### Schritt 2: Center Crop (224×224)

```python
transforms.CenterCrop(224)
```

**Erklärung für Portfolio:**

- Fokussiert auf Bildmitte (wo der Tumor typischerweise liegt)
- **Anti-Clever-Hans Maßnahme**: Verhindert Lernen von Rand-Artefakten
- 224px = Standard für ImageNet-vortrainierte Modelle

#### Schritt 3: Grayscale Conversion

```python
transforms.Grayscale(num_output_channels=3)
```

**Erklärung für Portfolio:**

- MRT-Bilder sind von Natur aus Graustufen
- Farben wären nur Rauschen/Artefakte
- 3 Kanäle für Kompatibilität mit vortrainiertem Netz

#### Schritt 4: Normalize

```python
transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
```

**Erklärung für Portfolio:**

- Verwendet ImageNet-Statistiken (Pretrained Model)
- Hilft bei schnellerer Konvergenz
- Werte zwischen -2 und +2 statt 0-255

---

### Phase 3: Model Inference

```
┌─────────────────────────────────────────────────────────────────┐
│                    EFFICIENTNET-B0 ARCHITEKTUR                   │
│                                                                  │
│   ┌───────────┐     ┌───────────────────┐     ┌──────────────┐  │
│   │  Input    │     │   Feature         │     │  Classifier  │  │
│   │ 224×224×3 │ --> │   Extraction      │ --> │  4 Klassen   │  │
│   │           │     │   (ConvBlocks)    │     │  Softmax     │  │
│   └───────────┘     └───────────────────┘     └──────────────┘  │
│                                                                  │
│   Parameter: 5.3M                                                │
│   Pretrained: ImageNet                                          │
│   Finetuned: Brain Tumor Dataset                                │
└─────────────────────────────────────────────────────────────────┘
```

**Was passiert:**

```python
with torch.no_grad():
    logits = model(tensor)           # Raw Scores
    probs = F.softmax(logits, dim=1) # Wahrscheinlichkeiten
```

**Output:**

```json
{
  "glioma": 0.05,
  "meningioma": 0.02,
  "notumor": 0.9,
  "pituitary": 0.03
}
```

---

### Phase 4: GradCAM Visualisierung (Explainable AI)

> **Highlight für Portfolio:** Zeigt Transparenz & Interpretierbarkeit

```
┌─────────────────────────────────────────────────────────────────┐
│                         GRAD-CAM                                 │
│         Gradient-weighted Class Activation Mapping               │
│                                                                  │
│   ┌─────────────┐                      ┌─────────────────────┐  │
│   │   Original  │                      │    Heatmap Overlay  │  │
│   │    Image    │    +    Gradients =  │    🔴 Wichtig       │  │
│   │             │                      │    🔵 Unwichtig     │  │
│   └─────────────┘                      └─────────────────────┘  │
│                                                                  │
│   Zeigt: "Wo schaut das Modell hin?"                            │
└─────────────────────────────────────────────────────────────────┘
```

**Erklärung für Portfolio:**

- **Warme Farben (Rot/Orange)**: Hohe Aufmerksamkeit des Modells
- **Kalte Farben (Blau)**: Niedrige Aufmerksamkeit
- Zeigt, ob das Modell auf relevante Bereiche (Tumor) schaut
- **Vertrauensbildend**: Man kann sehen, dass das Modell nicht auf Artefakte reagiert

---

### Phase 5: Confidence Calibration

```
┌─────────────────────────────────────────────────────────────────┐
│                    KONFIDENZ-ANALYSE                             │
│                                                                  │
│   Konfidenz: 92%                                                 │
│   ████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░      │
│                                                                  │
│   Entropie-Meter:                                                │
│   [Sicher]──────────●──────────[Unsicher]                       │
│            ↑                                                     │
│         Niedrige Entropie = Modell ist sich sicher               │
│                                                                  │
│   Interpretation:                                                │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  > 90%  │ HOCH   │ Modell ist sehr sicher               │   │
│   │  70-90% │ MITTEL │ Zusätzliche Verifikation empfohlen   │   │
│   │  < 70%  │ NIEDRIG│ Experten-Review empfohlen            │   │
│   └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

**Erklärung für Portfolio:**

- Nicht nur "was" das Modell vorhersagt, sondern "wie sicher" es ist
- **Entropy-basierte Unsicherheitsquantifizierung**
- Wichtig für medizinische Anwendungen: Low-Confidence Cases brauchen Review

---

## 🔧 API Endpoints (für React-Integration)

### GET `/api/info`

Liefert Modell-Metadaten für "About this Model" Section:

```json
{
  "model": {"name": "EfficientNet-B0", "parameters": "5.3M"},
  "training": {"techniques": ["Transfer Learning", "Anti-Clever-Hans"]},
  "anti_clever_hans": {"methods": [...]}
}
```

### GET `/api/preprocessing`

Liefert Preprocessing-Schritte für Educational Walkthrough:

```json
{
  "steps": [
    {"id": "resize", "name": "Resize", "why": "..."},
    {"id": "center_crop", "name": "Center Crop", "why": "..."},
    ...
  ]
}
```

### POST `/api/predict/explain`

**Hauptendpoint** - Liefert vollständige Analyse:

```json
{
  "prediction": {"class": "glioma", "confidence": 0.85},
  "probabilities": {...},
  "preprocessing": {"images": {"original": "base64...", "resize": "base64..."}},
  "gradcam": {"image": "base64...", "explanation": "..."},
  "calibration": {"entropy": 0.2, "recommendation": "..."}
}
```

---

## 🎓 Anti-Clever-Hans Maßnahmen

> **Was ist der Clever-Hans-Effekt?** Das Modell lernt Abkürzungen/Artefakte
> statt die echten Features (z.B. Schädel-Rand statt Tumor).

| Maßnahme           | Implementierung      | Wirkung                      |
| ------------------ | -------------------- | ---------------------------- |
| **Center Crop**    | Nur 224px Zentrum    | Schädel-Rand wird ignoriert  |
| **Grayscale**      | Farben entfernen     | Keine Scanner-Farb-Artefakte |
| **Mixup**          | Bilder mischen       | Verhindert Auswendiglernen   |
| **Random Erasing** | Zufällige Maskierung | Robustere Features           |

---

## 📱 UI-Flow für Portfolio-Präsentation

### Empfohlener Demo-Ablauf (3-5 Minuten)

1. **Intro (30 Sek)**
   - "Dies ist ein KI-System zur Hirntumor-Klassifikation"
   - "4 Klassen: Glioma, Meningioma, Pituitary, No Tumor"

2. **Sample laden (20 Sek)**
   - Random Sample über API laden
   - Zeigen, dass echte MRT-Bilder verwendet werden

3. **Preprocessing zeigen (60 Sek)**
   - Jeden Schritt durchgehen
   - Erklären WARUM jeder Schritt nötig ist
   - **Key Point:** "Das Modell sieht nie das Original-Bild"

4. **Prediction (30 Sek)**
   - Wahrscheinlichkeitsverteilung zeigen
   - Confidence Level erklären

5. **GradCAM (60 Sek)**
   - Heatmap zeigen
   - **Key Point:** "Wir können sehen, WO das Modell hinschaut"
   - "Das Modell fokussiert auf den Tumor, nicht auf Artefakte"

6. **Calibration (30 Sek)**
   - Entropy-Meter zeigen
   - **Key Point:** "In der Medizin reicht 'wahrscheinlich' nicht"

7. **Zusammenfassung (30 Sek)**
   - Transfer Learning + Anti-Clever-Hans
   - Explainable AI für Vertrauen
   - Production-Ready API

---

## 🐳 Docker Deployment

```bash
# Setup (kopiert Model + Samples)
./setup.ps1   # Windows
./setup.sh    # Linux/macOS

# Container starten
docker-compose up -d

# API verfügbar unter
http://localhost:5000
```

---

## 📊 Technologie-Stack

| Komponente         | Technologie               |
| ------------------ | ------------------------- |
| **Model**          | PyTorch + EfficientNet-B0 |
| **API**            | Flask + Flask-CORS        |
| **Explainability** | pytorch-grad-cam          |
| **Container**      | Docker + docker-compose   |
| **Frontend**       | (React-ready API)         |

---

## 💡 Portfolio-Highlights zum Betonen

1. ✅ **Transfer Learning** - Vortrainiertes Modell adaptiert
2. ✅ **Explainable AI** - GradCAM für Transparenz
3. ✅ **Anti-Clever-Hans** - Robuste Features statt Artefakte
4. ✅ **Confidence Calibration** - Weiß, wenn es unsicher ist
5. ✅ **Production API** - Docker-ready, CORS-enabled
6. ✅ **Educational Design** - Preprocessing-Schritte visualisiert

---

> 📌 **Für Fragen in Interviews:**
>
> - "Warum EfficientNet?" → Beste Accuracy/Parameter-Ratio
> - "Warum Grayscale?" → MRT ist inherent grayscale, Farben = Noise
> - "Was ist GradCAM?" → Zeigt welche Pixel zur Entscheidung beitragen
> - "Was ist Clever-Hans?" → Modell lernt Shortcuts statt echte Features
