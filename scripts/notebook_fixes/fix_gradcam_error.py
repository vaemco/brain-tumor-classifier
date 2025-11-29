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
        if "target_size = (original_image.shape[1], original_image.shape[0])" in source_text:
            print("Found error line")
            new_source = []
            for line in source:
                if "target_size = (original_image.shape[1], original_image.shape[0])" in line:
                    new_source.append("target_size = original_image.size\n")
                else:
                    new_source.append(line)

            cell['source'] = new_source
            found = True
            break

if found:
    with open(nb_path, 'w') as f:
        json.dump(nb, f, indent=1)
    print("Notebook updated to fix AttributeError")
else:
    print("Error line not found")
