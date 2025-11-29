import json
import os

nb_path = '../../notebooks/05_model_explainability_education.ipynb'

if not os.path.exists(nb_path):
    print(f"File not found: {nb_path}")
    exit(1)

with open(nb_path, 'r') as f:
    nb = json.load(f)

found = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        source_text = "".join(source)
        if "def generate_gradcam(model, input_tensor):" in source_text:
            print("Found Grad-CAM cell")
            new_source = []
            for line in source:
                if "def generate_gradcam(model, input_tensor):" in line:
                    new_source.append("def generate_gradcam(model, input_tensor, target_size=(224, 224)):\n")
                elif "cam = cv2.resize(cam, (224, 224))" in line:
                    new_source.append("    cam = cv2.resize(cam, target_size)\n")
                elif "heatmap = generate_gradcam(model, processed_tensor)" in line:
                    new_source.append("# Generate heatmap\n")
                    new_source.append("target_size = (original_image.shape[1], original_image.shape[0])\n")
                    new_source.append("heatmap = generate_gradcam(model, processed_tensor, target_size=target_size)\n")
                else:
                    new_source.append(line)

            cell['source'] = new_source
            found = True
            break

if found:
    with open(nb_path, 'w') as f:
        json.dump(nb, f, indent=1)
    print("Notebook updated")
else:
    print("Grad-CAM cell not found")
