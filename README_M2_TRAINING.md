# Brain Tumor Classification Training - M2 MacBook

## Übersicht

Dieses Projekt trainiert ein ResNet18-Modell zur Klassifizierung von Hirntumor-MRT-Bildern auf einem M2 MacBook mit MPS (Metal Performance Shaders) Unterstützung.

## Neue Dateien

- **`notebooks/train_m2_macbook.ipynb`**: Neues Training-Notebook optimiert für M2 MacBook
- **`models/`**: Verzeichnis für trainierte Modelle

## Änderungen gegenüber dem Original

### 1. Device-Unterstützung
- **Neu**: Automatische Erkennung von MPS (Apple Silicon GPU)
- Falls MPS verfügbar: nutzt Apple Silicon GPU
- Fallback auf CUDA (NVIDIA) oder CPU

### 2. Datenpfad
- **Neu**: `/Users/valentinemser/dev_projects/03_data_projects/data_brain_tumor/data/Brain_Tumor_Dataset/Training`
- **Alt**: Windows-Pfad (OneDrive)

### 3. Modell-Speicherpfad
- **Neu**: `../models/brain_tumor_resnet18_final.pt`
- Relativer Pfad zum Projektverzeichnis

### 4. Aktualisierte Dateien
- **`website/dataset.py`**: Aktualisierte Transforms (Grayscale mit 3 Kanälen, 224x224)
- **`website/gui_braintumor.py`**: 
  - ResNet18 statt SimpleCNN
  - MPS-Unterstützung
  - Neue Modellpfade

## Training starten

1. **Umgebung aktivieren:**
   ```bash
   mamba activate data_brain
   ```

2. **Jupyter starten:**
   ```bash
   jupyter notebook
   ```

3. **Notebook öffnen:**
   - Öffne `notebooks/train_m2_macbook.ipynb`
   - Führe alle Zellen nacheinander aus

## Nach dem Training

Das trainierte Modell wird gespeichert unter:
- `models/brain_tumor_resnet18_final.pt` - Finales Modell
- `models/brain_tumor_resnet18_scripted.pt` - TorchScript Version
- `runs/metrics.json` - Training-Metriken

## GUI testen

Nach dem Training kannst du die GUI starten:

```bash
python -m website.gui_braintumor
```

**Wichtig**: Die GUI benötigt das trainierte Modell unter `models/brain_tumor_resnet18_final.pt`

## Klassen

Das Modell klassifiziert 4 Arten von Hirntumoren:
- `glioma` - Gliom
- `meningioma` - Meningiom
- `no_tumor` - Kein Tumor
- `pituitary` - Hypophysenadenom

## Training-Parameter

- **Modell**: ResNet18 (vortrainiert auf ImageNet)
- **Optimierer**: Adam mit differenzierten Learning Rates
- **Batch Size**: 32
- **Epochs**: 30 (mit Early Stopping)
- **Patience**: 5 Epochen
- **Validation Split**: 80/20

## Technische Details

### Transforms

**Training:**
- Grayscale → 3 Kanäle
- Resize: 256
- RandomResizedCrop: 224
- RandomHorizontalFlip: 0.5
- RandomRotation: ±15°
- ColorJitter
- Normalization: ImageNet

**Validation:**
- Grayscale → 3 Kanäle
- Resize: 256
- CenterCrop: 224
- Normalization: ImageNet

### Architektur

- **Base**: ResNet18
- **Frozen**: layer1, layer2
- **Trainable**: layer3, layer4, fc
- **Output**: 4 Klassen

## Fehlerbehebung

### MPS nicht verfügbar
Falls MPS nicht erkannt wird:
```python
import torch
print(torch.backends.mps.is_available())  # Sollte True sein
print(torch.backends.mps.is_built())      # Sollte True sein
```

### CUDA Warnung
Falls du CUDA-Warnungen siehst, ignoriere sie - das Notebook nutzt automatisch MPS auf M2.

### Modell nicht gefunden
Stelle sicher, dass das Verzeichnis `models/` existiert und das Training abgeschlossen ist.

## Nächste Schritte

1. Training durchführen
2. Metriken in `runs/metrics.json` überprüfen
3. GUI testen
4. Optional: Modell optimieren (z.B. mehr Datenaugmentation)
