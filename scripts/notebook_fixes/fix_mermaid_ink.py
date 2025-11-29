import json
import os

nb_path = '../../notebooks/05_model_explainability_education.ipynb'

if not os.path.exists(nb_path):
    print(f"File not found: {nb_path}")
    exit(1)

with open(nb_path, 'r') as f:
    nb = json.load(f)

# Find the Mermaid cell and update it to render using mermaid.ink image
found = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "def get_mermaid_graph(model" in source:
            print("Found Mermaid cell, updating to use mermaid.ink...")
            cell['source'] = [
                "# Generate Mermaid.js Graph and Render as Image\n",
                "import base64\n",
                "from IPython.display import Image, display\n",
                "\n",
                "def get_mermaid_graph(model, input_shape=(1, 3, 224, 224)):\n",
                "    graph = [\"graph TD\"]\n",
                "    graph.append(f\"    Input(Input {input_shape})\")\n",
                "    previous_node = \"Input\"\n",
                "    \n",
                "    for name, layer in model.named_children():\n",
                "        node_name = name\n",
                "        layer_type = layer.__class__.__name__\n",
                "        graph.append(f\"    {node_name}[{layer_type}]\")\n",
                "        graph.append(f\"    {previous_node} --> {node_name}\")\n",
                "        previous_node = node_name\n",
                "    \n",
                "    graph.append(f\"    {previous_node} --> Output\")\n",
                "    return \"\\n\".join(graph)\n",
                "\n",
                "mermaid_code = get_mermaid_graph(model)\n",
                "\n",
                "# Encode for mermaid.ink\n",
                "graphbytes = mermaid_code.encode(\"utf8\")\n",
                "base64_bytes = base64.b64encode(graphbytes)\n",
                "base64_string = base64_bytes.decode(\"ascii\")\n",
                "url = \"https://mermaid.ink/img/\" + base64_string\n",
                "\n",
                "# Display the image\n",
                "display(Image(url=url))\n"
            ]
            found = True
            break

if found:
    with open(nb_path, 'w') as f:
        json.dump(nb, f, indent=1)
    print("Notebook updated with mermaid.ink image rendering")
else:
    print("Mermaid cell not found")
