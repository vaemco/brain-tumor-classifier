# Brain Tumor Classifier - Demo Container

Self-contained Docker container for portfolio integration.
Provides educational API with explainability features for React frontend.

## Quick Start

```bash
# 1. Run setup script to copy model + samples
./setup.ps1   # Windows
./setup.sh    # Linux/macOS

# 2. Build and run
docker-compose up -d

# API available at
http://localhost:5000
```

## Container Structure

```
demo/
├── app.py              # Flask API (CORS enabled for React)
├── Dockerfile          # Container definition
├── docker-compose.yml  # Orchestration
├── requirements.txt    # Python dependencies
├── setup.ps1           # Windows setup script
├── setup.sh            # Linux/macOS setup script
├── model/
│   └── brain_tumor_efficientnet_b0.pt  # Trained model (~20MB)
└── samples/
    ├── glioma/         # Sample MRI images
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

## API Endpoints

### Information

| Endpoint             | Method | Description                           |
| -------------------- | ------ | ------------------------------------- |
| `/`                  | GET    | API documentation                     |
| `/api/info`          | GET    | Model architecture & training details |
| `/api/classes`       | GET    | Tumor class descriptions              |
| `/api/preprocessing` | GET    | Pipeline step explanations            |
| `/health`            | GET    | Container health check                |

### Sample Images

| Endpoint                      | Method | Description                |
| ----------------------------- | ------ | -------------------------- |
| `/api/samples`                | GET    | List all available samples |
| `/api/samples/{class}/{file}` | GET    | Get specific sample image  |
| `/api/samples/random`         | GET    | Random sample with label   |

### Prediction

| Endpoint               | Method | Description                   |
| ---------------------- | ------ | ----------------------------- |
| `/api/predict`         | POST   | Quick classification          |
| `/api/predict/explain` | POST   | Full explanation with GradCAM |

## React Integration Examples

### 1. Fetch Model Info

```typescript
interface ModelInfo {
  model: { name: string; parameters: string; pretrained: string };
  training: { techniques: string[] };
  anti_clever_hans: { methods: { name: string; description: string }[] };
}

const response = await fetch("http://localhost:5000/api/info");
const modelInfo: ModelInfo = await response.json();
```

### 2. Get Preprocessing Steps (Educational)

```typescript
interface PreprocessingStep {
  id: string;
  name: string;
  description: string;
  why: string; // Educational explanation
  params: Record<string, unknown>;
}

const response = await fetch("http://localhost:5000/api/preprocessing");
const { steps }: { steps: PreprocessingStep[] } = await response.json();

// Display each step with explanation
steps.forEach((step) => {
  console.log(`${step.name}: ${step.why}`);
});
```

### 3. Classify with Full Explanation

```typescript
interface PredictionResult {
  prediction: {
    class: string;
    confidence: number;
    confidence_level: "high" | "medium" | "low";
    confidence_explanation: string;
    class_info: { name: string; description: string; severity: string };
  };
  probabilities: Record<string, { value: number; rank: number }>;
  calibration: {
    entropy: number;
    interpretation: string;
    recommendation: string;
  };
  preprocessing: {
    steps: PreprocessingStep[];
    images: {
      original: string; // base64
      resize: string;
      center_crop: string;
      grayscale: string;
      normalize: string;
    };
  };
  gradcam?: {
    image: string; // base64
    explanation: string;
    interpretation: string;
  };
}

const formData = new FormData();
formData.append("file", imageFile);

const response = await fetch("http://localhost:5000/api/predict/explain", {
  method: "POST",
  body: formData,
});

const result: PredictionResult = await response.json();
```

### 4. Interactive Sample Browser

```typescript
// Get random sample for demo
const response = await fetch("http://localhost:5000/api/samples/random");
const sample = await response.json();

// sample = {
//   class: "meningioma",
//   filename: "Te-me_0001.jpg",
//   image: "base64...",  // Ready to display
//   info: { name: "Meningioma", description: "...", severity: "medium" }
// }
```

## Example React Component

```tsx
import { useState, useEffect } from "react";

const API_URL = "http://localhost:5000";

export function BrainTumorDemo() {
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [currentStep, setCurrentStep] = useState(0);

  const handleUpload = async (file: File) => {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch(`${API_URL}/api/predict/explain`, {
      method: "POST",
      body: formData,
    });

    setResult(await response.json());
  };

  // Interactive preprocessing visualization
  if (result) {
    const steps = result.preprocessing.steps;
    const images = result.preprocessing.images;

    return (
      <div>
        <h2>
          Step {currentStep + 1}: {steps[currentStep].name}
        </h2>
        <p>{steps[currentStep].why}</p>
        <img src={`data:image/png;base64,${images[steps[currentStep].id]}`} />
        <button
          onClick={() =>
            setCurrentStep((s) => Math.min(s + 1, steps.length - 1))
          }
        >
          Next Step
        </button>
      </div>
    );
  }

  return <DropZone onUpload={handleUpload} />;
}
```

## Resource Usage

| Resource       | Usage  |
| -------------- | ------ |
| Container Size | ~1.5GB |
| Runtime RAM    | ~600MB |
| CPU            | 1 core |
| Model Size     | ~20MB  |

## Production Deployment

### With Docker Compose

```bash
docker-compose up -d
```

### Manual with Gunicorn

```bash
pip install -r requirements.txt
gunicorn -w 1 -b 0.0.0.0:5000 --timeout 60 app:app
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: brain-tumor-demo
spec:
  replicas: 1
  template:
    spec:
      containers:
        - name: demo
          image: brain-tumor-demo:latest
          ports:
            - containerPort: 5000
          resources:
            limits:
              memory: "1Gi"
```

---

For educational/portfolio purposes only. Not for medical diagnosis.
