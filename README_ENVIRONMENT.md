# Mamba Environment: data_brain

This environment has been created for the brain tumor classification project.

## Activation

To activate the environment, run:
```bash
mamba activate data_brain
```

or

```bash
conda activate data_brain
```

## Deactivation

To deactivate the environment, run:
```bash
conda deactivate
```

## Installed Packages

The environment includes:
- **Python 3.10**
- **Deep Learning**: PyTorch, Torchvision, Torchaudio (CPU version)
- **Data Science**: NumPy, Pandas, Scikit-learn
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Image Processing**: Pillow, OpenCV
- **Jupyter**: Jupyter Notebook, IPython, IPykernel, IPywidgets
- **GUI**: Tkinter
- **Utilities**: tqdm, PyYAML, tensorboard

## GPU Support

The current environment is configured for **CPU-only** operation. If you need GPU support:

1. Edit `environment.yml`
2. Remove or comment out the line: `- cpuonly`
3. Recreate the environment or update it:
   ```bash
   mamba env update -f environment.yml
   ```

## Running the GUI Application

To run the brain tumor classifier GUI:
```bash
mamba activate data_brain
cd /Users/valentinemser/dev_projects/03_data_projects/data_brain_tumor
python -m website.gui_braintumor
```

## Running Jupyter Notebooks

To start Jupyter Notebook:
```bash
mamba activate data_brain
jupyter notebook
```

## Updating the Environment

If you need to add more packages:
1. Edit `environment.yml`
2. Run: `mamba env update -f environment.yml`

Or install packages directly:
```bash
mamba install package_name
# or for pip packages
pip install package_name
```
