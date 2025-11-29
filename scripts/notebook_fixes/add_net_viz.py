import json
import os

nb_path = '../../notebooks/05_model_explainability_education.ipynb'

if not os.path.exists(nb_path):
    print(f"File not found: {nb_path}")
    exit(1)

with open(nb_path, 'r') as f:
    nb = json.load(f)

# Create new cell
new_cell = {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Visualize Network Architecture\n",
    "print(\"Model Architecture:\")\n",
    "print(model)\n",
    "\n",
    "print(\"\\nModel Graph (torch.fx):\")\n",
    "try:\n",
    "    import torch.fx\n",
    "    gm = torch.fx.symbolic_trace(model)\n",
    "    gm.graph.print_tabular()\n",
    "except Exception as e:\n",
    "    print(f\"Could not generate torch.fx graph: {e}\")\n"
   ]
}

nb['cells'].append(new_cell)

with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=1)
print("Notebook updated with network visualization")
