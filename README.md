# Benchmarking machine learning models for predicting aerofoil performance

## 💾 Data

The data used in this repository is the 'turb_model' data from the '2k Shapes Data Sets' which can be found here: https://registry.opendata.aws/nrel-pds-windai/

For a breakdown of the available data, see the NREL GitHub repository here: https://github.com/NREL/windAI_bench

## 🚀 Overview

This repository addresses the computational challenge of aerodynamic flow simulation by replacing expensive CFD computations with fast neural network predictions. The framework supports multiple state-of-the-art architectures optimized for geometric deep learning on irregular mesh data.

This repository is the code supplemetary for the paper published to EWTEC 2025, "Benchmarking machine learning models for predicting aerofoil performance", which can be found here: https://arxiv.org/abs/2504.15993

### Key Features

- **Multiple Model Architectures**: MLP, PointNet, GraphSAGE, and GUNet (Graph U-Net) implementations
- **Comprehensive Benchmarking**: Systematic evaluation across different training set sizes and model configurations
- **Aerodynamic Metrics**: Calculation of lift coefficients and flow parameters, including momentum, energy, and vorticity. 
- **Data Processing Pipeline**: Complete preprocessing, normalization, and post-processing workflows
- **Visualization Tools**: Error analysis, performance metrics, and result visualization capabilities

## 📊 Supported Models

### 1. **MLP (Multi-Layer Perceptron)**
Traditional fully-connected neural network baseline for comparison with geometric deep learning approaches.

### 2. **PointNet**
Point cloud-based architecture that processes mesh nodes as permutation-invariant point sets with global feature aggregation.

### 3. **GraphSAGE** 
Graph-based neural network using neighborhood sampling and aggregation for flow field prediction on unstructured meshes.

### 4. **GUNet (Graph U-Net)**
Multi-scale graph neural network with encoder-decoder architecture featuring graph pooling and upsampling operations.


## 🛠️ Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.8+
- PyTorch Geometric
- PyVista
- NumPy, Matplotlib, Seaborn
- h5py, tqdm, PyYAML

### Setup
```bash
git clone https://github.com/OllieS-PhD/Benchmark_windAI.git
cd Benchmark_windAI
pip install -r requirements.txt
```

## 🎯 Quick Start

### Basic Training
```bash
# General training formula
python main.py 'model' -f 'n_foils' -e 'n_epochs'  

# E.g. Train GraphSAGE model on 20 airfoils for 400 epochs
python main.py GraphSAGE -f 20 -e 400
```

### Model Evaluation
```bash
# Custom validation run
python validation.py --model GraphSAGE --foils 55
```


### All encompassing run
```bash
# Run the batch file: runner.bat
@echo off
set mod= "MLP" "PointNet" "GUNet" "GraphSAGE"
set task="-t full"
set foils= 5 20 55 150
set epochs=50 100 200 400

for %%m in (%mod%) do (
for %%f in (%foils%) do (
python C:\\Users\\olive\\Documents\\Code\\eXFoil\\eX-Foil\\main.py %%m -f %%f -e 400
)
)
python C:\\Users\\olive\\Documents\\Code\\eXFoil\\eX-Foil\\validation.py
```

## 📁 Project Structure

```
eX-Foil/
├── main.py                 # Main training script
├── train.py                # Training utilities and loops
├── validation.py           # Validation and testing procedures
├── dataset.py              # Dataset loading and preprocessing
├── params.yaml             # Model hyperparameters
├── normalise.py            # Data normalization utilities
├── metrics.py              # Performance evaluation metrics
├── metrics_NACA.py         # NACA-specific aerodynamic metrics
├── naca_generator.py       # NACA airfoil generation utilities
├── models/                 # Neural network architectures
│   ├── GraphSAGE.py
│   ├── PointNet.py
│   ├── GUNet.py
│   ├── MLP.py
│   └── NN.py
├── data/                   # Data processing modules
│   ├── data_loader.py
│   ├── data_processor.py
│   ├── post_process.py
│   └── visualise_error.py
├── post_proc/              # Post-processing utilities
└── metrics/                # Results and evaluation outputs
```

## ⚙️ Configuration

Model hyperparameters are defined in `params.yaml`:

```yaml
GraphSAGE:
  encoder: [8, 64, 64, 8]
  decoder: [8, 64, 64, 5]
  nb_hidden_layers: 3
  size_hidden_layers: 64
  batch_size: 1
  nb_epochs: 400
  lr: 0.001
  max_neighbors: 6
  subsampling: 32000
  r: 0.05
```

## 📈 Performance Metrics

The framework evaluates models using:

- **Flow Field Accuracy**: RMSE on density, momentum, energy, and vorticity
- **Aerodynamic Coefficients**: Lift (CL) coefficient prediction


## 📊 Benchmarking Results

Training and validation results are organized by:
- Number of training airfoils (5, 20, 55, 150)
- Model architecture comparisons
- Epoch-wise convergence analysis
- Foil-by-foil performance breakdown

Results are saved in the `metrics/` directory with the visualization and statistical analysis performed by validation.py

## 📚 Citation

If you use this repository in your research, please cite:

```bibtex
@article{Summerell25,
   author = {Oliver Summerell and Gerardo Aragon-Camarasa and Stephanie Ordonez Sanchez},
   doi = {10.36688/ewtec-2025-879},
   journal = {PROCEEDINGS OF THE 16TH EUROPEAN WAVE AND TIDAL ENERGY CONFERENCE},
   title = {Benchmarking machine learning models for predicting aerofoil performance},
   year = {2025},
   url = {https://arxiv.org/abs/2504.15993}
}
```

## 🙋‍♂️ Support

For questions and support:
- Open an issue on GitHub
- Email the authour at o.summerell.1@research.gla.ac.uk

---

**Note**: This framework is actively under development. Please check the latest releases for stable versions and updated features.
