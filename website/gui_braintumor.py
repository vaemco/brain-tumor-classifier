"""
Brain Tumor Classification GUI
Optimized for M2 MacBook with MPS support
"""

import tkinter as tk
from pathlib import Path
from tkinter import Button, Label, filedialog, messagebox

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
from torchvision import models

# Import validation transforms
from website.dataset import val_tf

# Configuration
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR.parent / "models" / "brain_tumor_resnet18_v2_trained.pt"
CLASSES = ["glioma", "meningioma", "no_tumor", "pituitary"]

# Device setup - M2 optimized
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✓ Using MPS (Apple Silicon GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("✓ Using CUDA GPU")
else:
    device = torch.device("cpu")
    print("⚠ Using CPU")


# Load model
def load_model():
    """Load ResNet18 model with trained weights"""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at: {MODEL_PATH}\\n"
            "Please train the model first using train_m2_macbook.ipynb"
        )

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device, weights_only=True)
    )
    model = model.to(device)
    model.eval()
    return model


try:
    model = load_model()
except FileNotFoundError as e:
    print(f"Error: {e}")
    model = None


@torch.no_grad()
def predict_image(img_path):
    """Predict brain tumor class from image"""
    if model is None:
        raise RuntimeError("Model not loaded")

    img = Image.open(img_path).convert("RGB")
    x = val_tf(img).unsqueeze(0).to(device)

    out = model(x)
    probs = F.softmax(out, dim=1).cpu().numpy()[0]

    result = {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}
    return result, img


def open_file():
    """Open file dialog and make prediction"""
    if model is None:
        messagebox.showerror("Error", "Model not loaded. Please train the model first.")
        return

    filepath = filedialog.askopenfilename(
        title="Select MRI Image", filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )

    if not filepath:
        return

    try:
        result, pil_img = predict_image(filepath)

        # Display image
        img_resized = pil_img.resize((200, 200))
        img_tk = ImageTk.PhotoImage(img_resized)
        img_label.config(image=img_tk)
        img_label.image = img_tk

        # Create pie chart
        fig, ax = plt.subplots(figsize=(3, 3))
        values = list(result.values())
        labels = list(result.keys())
        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]
        explode = [0.1 if v == max(values) else 0 for v in values]

        ax.pie(
            values,
            labels=labels,
            autopct="%1.1f%%",
            startangle=90,
            explode=explode,
            colors=colors,
        )
        ax.set_title("Probabilities", fontsize=12, fontweight="bold")

        # Clear previous chart
        for widget in chart_frame.winfo_children():
            widget.destroy()

        canvas = FigureCanvasTkAgg(fig, master=chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

        # Display top prediction
        top_class = max(result, key=result.get)
        top_prob = result[top_class] * 100
        result_text.set(f"{top_prob:.1f}% → {top_class.upper()}")

        # Update color based on prediction
        if top_class == "no_tumor":
            pred_label.config(fg="green")
        else:
            pred_label.config(fg="red")

    except Exception as e:
        messagebox.showerror("Error", f"Failed to process image:\\n{str(e)}")


# GUI Setup
root = tk.Tk()
root.title("Brain Tumor Classifier")
root.geometry("600x400")
root.configure(bg="#f0f0f0")

# Main frame
main_frame = tk.Frame(root, bg="#f0f0f0")
main_frame.pack(pady=20)

# Image display (left)
img_label = Label(
    main_frame,
    bg="#f0f0f0",
    text="No image loaded",
    width=25,
    height=12,
    relief="sunken",
)
img_label.grid(row=0, column=0, padx=20)

# Chart display (right)
chart_frame = tk.Frame(main_frame, bg="#f0f0f0")
chart_frame.grid(row=0, column=1, padx=20)

# Result display
result_text = tk.StringVar()
result_text.set("Select an image to classify")
pred_label = Label(
    root,
    textvariable=result_text,
    font=("Helvetica", 18, "bold"),
    bg="#f0f0f0",
    fg="#333",
)
pred_label.pack(pady=20)

# Button
btn = Button(
    root,
    text="📁 Select MRI Image",
    command=open_file,
    font=("Helvetica", 14),
    bg="#4ECDC4",
    fg="white",
    padx=20,
    pady=10,
    relief="raised",
    cursor="hand2",
)
btn.pack(pady=10)

# Status bar
status = Label(
    root,
    text=f"Device: {device}  |  Model: ResNet18  |  Classes: {len(CLASSES)}",
    font=("Courier", 9),
    bg="#ddd",
    anchor="w",
    relief="sunken",
)
status.pack(side="bottom", fill="x")

if __name__ == "__main__":
    root.mainloop()
