import json
import os

nb_path = '../../notebooks/05_model_explainability_education.ipynb'

if not os.path.exists(nb_path):
    print(f"File not found: {nb_path}")
    exit(1)

with open(nb_path, 'r') as f:
    nb = json.load(f)

# Create new cell for Mermaid visualization
mermaid_cell = {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Generate Mermaid.js Graph\n",
    "def get_mermaid_graph(model, input_shape=(1, 3, 224, 224)):\n",
    "    graph = [\"graph TD\"]\n",
    "    graph.append(f\"    Input(Input {input_shape})\")\n",
    "    previous_node = \"Input\"\n",
    "    \n",
    "    for name, layer in model.named_children():\n",
    "        node_name = name\n",
    "        layer_type = layer.__class__.__name__\n",
    "        # Clean up layer type for display\n",
    "        graph.append(f\"    {node_name}[{layer_type}]\")\n",
    "        graph.append(f\"    {previous_node} --> {node_name}\")\n",
    "        previous_node = node_name\n",
    "    \n",
    "    graph.append(f\"    {previous_node} --> Output\")\n",
    "    return \"\\n\".join(graph)\n",
    "\n",
    "print(\"Mermaid Graph (Copy and paste into a Markdown cell to view):\")\n",
    "print(\"```mermaid\")\n",
    "print(get_mermaid_graph(model))\n",
    "print(\"```\")\n"
   ]
}

# Append the new cell
nb['cells'].append(mermaid_cell)

with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=1)
print("Notebook updated with Mermaid visualization")
