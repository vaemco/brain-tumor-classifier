# 🧠 Brain Tumor Classification with Deep Learning

> **AI-Assisted Project Notice**: This project was developed with assistance from AI tools to accelerate development and explore modern AI-powered workflows. All architecture decisions, optimizations, and implementations were reviewed and understood before integration.

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end deep learning solution for classifying brain tumors from MRI scans, featuring a modern web interface with explainable AI visualizations.

## 🎯 Project Overview

This project demonstrates how modern deep learning can assist in medical image classification. Using transfer learning with ResNet18 and Grad-CAM visualization, it classifies brain MRI scans into four categories:

- **Glioma** - Malignant brain tumor
- **Meningioma** - Usually benign tumor of brain membranes
- **Pituitary** - Tumor of the pituitary gland
- **No Tumor** - Healthy brain scan

### ✨ Key Features

- 🎨 **Modern Web Interface** with drag & drop functionality
- 🔍 **Grad-CAM Visualization** - See exactly where the model focuses
- 📊 **Real-time Analysis** with probability distributions
- 🚀 **M2 MacBook Optimized** using Metal Performance Shaders (MPS)
- 📱 **Responsive Design** works on desktop and mobile
- 🧪 **Interactive Testing** via web UI or Python API

## 🏗️ Architecture

### Model
- **Base**: ResNet18 (pretrained on ImageNet)
- **Transfer Learning**: Fine-tuned on brain tumor MRI dataset
- **Input**: 224x224 RGB images (grayscale converted)
- **Output**: 4-class softmax classification
- **Optimization**: AdamW optimizer with differential learning rates

### Tech Stack
- **Backend**: Flask, PyTorch, Grad-CAM
- **Frontend**: Vanilla JS, Chart.js, Modern CSS
- **Development**: Jupyter Notebooks, Conda/Mamba
- **Deployment Ready**: Docker-compatible (optional)

## 📊 Dataset

**Source**: [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) on Kaggle

- **Total Images**: ~7,000 MRI scans
- **Classes**: 4 (balanced distribution)
- **Format**: JPEG/PNG images
- **Split**: 80% Training / 20% Validation

**Note**: The dataset is## 📂 Project Structure

```
brain-tumor-classifier/
├── data/                   # Dataset (Training/Testing)
├── models/                 # Saved PyTorch models
├── notebooks/              # Jupyter Notebooks for experiments
├── scripts/                # Utility scripts (Train, Eval, Data Prep)
│   ├── train.py            # Training script
│   ├── evaluate.py         # Evaluation script
│   └── prepare_data.py     # Data split script
├── website/                # Flask Web Application
│   ├── static/             # CSS, JS, Images
│   ├── templates/          # HTML Templates
│   └── app.py              # Main App Entry Point
├── environment.yml         # Conda Environment
└── README.md               # Project Documentation
```

## 🚀 Getting Started

### 1. Installation

```bash
# Clone the repo
git clone https://github.com/your-username/brain-tumor-classifier.git
cd brain-tumor-classifier

# Create Conda Environment
conda env create -f environment.yml
conda activate data_brain
```

### 2. Run the Web App

```bash
python3 -m website.app
```

Visit `http://localhost:3000` in your browser.

### 3. Training & Evaluation

Scripts are located in the `scripts/` folder:

```bash
# Train the model
python3 scripts/train.py

# Evaluate on external data
python3 scripts/evaluate.py
```
- Download and extract to `data/Brain_Tumor_Dataset/`

4. **Train the model** (optional - pretrained weights available)
```bash
jupyter notebook
# Open notebooks/train_m2_macbook.ipynb and run all cells
```

5. **Start the web application**
```bash
python -m website.app
```

6. **Open browser**
```
http://localhost:5000
```

## 📚 Project Structure

```
brain-tumor-classifier/
├── data/                          # Dataset (not in repo)
│   └── Brain_Tumor_Dataset/
│       └── Training/
├── models/                        # Trained model weights
│   └── brain_tumor_resnet18_final.pt
├── notebooks/                     # Jupyter notebooks
│   ├── 01_exploration.ipynb       # EDA & data analysis
│   └── train_m2_macbook.ipynb     # Model training
├── website/                       # Web application
│   ├── app.py                     # Flask backend
│   ├── dataset.py                 # Data preprocessing
│   ├── static/                    # CSS, JS, uploads
│   └── templates/                 # HTML templates
├── runs/                          # Training metrics
│   ├── metrics.json
│   └── training_history.png
├── docs/                          # Documentation
│   └── PROJECT_BRIEFING.pdf       # Detailed project brief
├── environment.yml                # Conda environment
├── .gitignore
├── LICENSE
└── README.md
```

## 🎓 Learning Outcomes

Through this project, I gained hands-on experience with:

1. **Transfer Learning**: Fine-tuning pretrained CNNs for medical imaging
2. **Explainable AI**: Implementing Grad-CAM for model interpretability
3. **Full-Stack ML**: From data preprocessing to web deployment
4. **Model Optimization**: Achieving high accuracy through systematic improvements
5. **Modern ML Ops**: Structured experiments, metrics tracking, reproducibility
6. **AI-Assisted Development**: Leveraging AI tools effectively while maintaining code quality

## 🔬 Model Performance

| Metric | Value |
|--------|-------|
| Validation Accuracy | ~98% |
| Training Time | ~15-20 min (M2 Mac) |
| Inference Time | ~300ms per image |
| Model Size | ~45 MB |

## 🎨 Features in Detail

### Grad-CAM Visualization
Gradient-weighted Class Activation Mapping (Grad-CAM) shows which regions of the MRI scan were most important for the model's decision. This explainability feature is crucial for medical AI applications.

### Web Interface
- **Drag & Drop**: Intuitive file upload
- **Real-time Analysis**: See results in seconds
- **Probability Distribution**: Understand model confidence
- **Visual Comparison**: Original vs heatmap view

### Desktop GUI (Alternative)
For offline use, a Tkinter-based desktop application is also available:
```bash
python -m website.gui_braintumor
```

## 📈 Future Improvements

- [ ] Ensemble models (ResNet50 + EfficientNet)
- [ ] Test-Time Augmentation for higher accuracy
- [ ] DICOM format support
- [ ] API authentication
- [ ] Model versioning & A/B testing
- [ ] Deployment to cloud (AWS/GCP)

## 🤝 Transparency & AI Usage

This project was developed with assistance from modern AI coding tools (GitHub Copilot, ChatGPT, Claude) to:
- Accelerate development cycles
- Explore best practices in ML engineering
- Learn modern full-stack development patterns
- Generate boilerplate code efficiently

**My contributions:**
- Architecture design and model selection
- Hyperparameter tuning and optimization strategies
- UI/UX design decisions
- Code review and understanding of all implementations
- Testing and validation
- Documentation and project structure

This approach allowed me to focus on high-level problem-solving and learning while AI handled repetitive coding tasks - a workflow increasingly common in modern software development.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: [Masoud Nickparvar](https://www.kaggle.com/masoudnickparvar) for the MRI dataset on Kaggle
- **ResNet Architecture**: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) by He et al.
- **Grad-CAM**: [Grad-CAM: Visual Explanations from Deep Networks](https://arxiv.org/abs/1610.02391) by Selvaraju et al.
- **AI Tools**: Claude (Anthropic), ChatGPT (OpenAI) for development assistance

## 📧 Contact

For questions or collaboration opportunities, feel free to reach out!

---

**Note**: This is a educational/portfolio project and is not intended for medical diagnosis. Always consult qualified healthcare professionals for medical decisions.
