# EvoActiv

Evolutionary discovery and optimization of activation functions for neural networks.

## Overview

EvoActiv is a research tool that uses evolutionary search to discover novel activation formulas for neural networks. It builds expression trees, mutates and crosses them, and evaluates candidate formulas by training a simple model on a dataset (MNIST by default). Beyond standard functions like ReLU, sigmoid, or tanh, EvoActiv explores composite and parametric formulas that can be trained alongside the model.

## Key Features

- Evolutionary search of activation formulas using genetic programming.
- Parametric constants inside formulas (e.g., `a`, `b`) trained via PyTorch optimizers.
- Safe numerical implementations (clamping, eps additions) to avoid NaNs and infs.
- Configurable operator set via `config/default.yaml` (enable/disable unary and binary ops).
- Joint training of model weights and formula parameters during evaluation.
- Early stopping, logging of best formula and fitness, saving results to JSON.
- CPU/GPU support via PyTorch (automatically selects `cuda` if available).
- Result analysis script to visualize and compare top formulas.

## Project Structure

```
EvoActiv/
├── data/                     # Datasets (MNIST, examples)
├── results/                  # JSON, plots, models
├── evo_core/                 # Core logic
│   ├── evolution.py          # Evolution engine: mutation, crossover
│   ├── formula_generator.py  # Expression tree generator
│   ├── activation_builder.py # Compile formulas into Torch activations
│   ├── train_evaluator.py    # Model training and validation loop
│   └── utils.py              # Logging, timers, helpers
├── datasets/
│   └── dataset_loader.py     # Unified dataset loader (Torchvision/custom CSV/JSON/NumPy)
├── config/
│   └── default.yaml          # Experiment settings (generations, population, etc.)
├── evolve.py                 # Main entry point for experiments
├── analyze_results.py        # Visualization and analysis of top formulas
└── requirements.txt          # Project dependencies
```

## Installation

Requirements: Python 3.10+ and PyTorch.

```
git clone https://github.com/yourusername/EvoActiv.git
cd EvoActiv
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: . .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Notes:
- If you encounter NumPy compatibility issues, pin `numpy<2` in `requirements.txt`.
- MNIST will be downloaded automatically to `data/` on first run.

## Quickstart

Run a small evolutionary search with default settings:

```
python evolve.py --config config/default.yaml --output results
```

Run a faster smoke test:

```
python evolve.py --config config/default.yaml --generations 2 --population 5
```

Increase search quality (more exploration):

```
python evolve.py --config config/default.yaml --generations 50 --population 100
```

## Usage Examples

### 1) Configure Operators

Enable or disable available operators via YAML. For example:

```yaml
operators:
  unary: ["sin", "cos", "exp", "log", "tanh", "sigmoid", "relu"]
  binary: ["+", "-", "*", "/", "**"]
  trainable_constants: true
```

- Unary ops include `sin`, `cos`, `exp`, `log`, `tanh`, `sigmoid`, `relu`.
- Binary ops include `+`, `-`, `*`, `/`, `**` (power uses clamping/eps for stability).
- `trainable_constants: true` allows numeric constants in formulas to become `nn.Parameter` during training.

### 2) Fine-Tune Formula Parameters

Enable joint training of formula constants with the model via the `formula_optimization` block:

```yaml
evolution:
  population_size: 100
  num_generations: 50

# Optimization of numeric coefficients within formulas
formula_optimization:
  enable: true
  learning_rate: 1e-3
  epochs: 3           # number of additional fine-tuning epochs per evaluation
  coefficient_range: [-2.0, 2.0]
```

With `enable: true`, EvoActiv creates `nn.Parameter`s for constants detected in an expression and optimizes them along with the network weights during evaluation. This is handled inside `train_evaluator.py` and `activation_builder.py`.

### 3) Post-Generation Fine-Tuning (Top-K)

Enable automatic fine-tuning of the best formulas after each generation:

```yaml
# Automatic fine-tuning of top formulas after each generation
fine_tuning:
  enable: true        # Enable post-generation fine-tuning
  top_k: 3            # Number of best formulas to fine-tune
  extra_epochs: 5     # Additional training epochs for fine-tuning
```

When enabled, EvoActiv will:
- Select the top-K formulas from each generation based on fitness
- Perform additional training epochs on each formula
- Log improvements in fitness and formula parameters
- Keep the improved versions for the next generation
- Save fine-tuning logs in the results file for analysis

### 4) Dataset Configuration (Universal Loader)

Switch dataset type or point to a custom CSV/JSON/NumPy dataset in `config/default.yaml`. Examples:

Torchvision:
```yaml
dataset:
  type: torchvision
  name: mnist      # options: mnist, fashionmnist, cifar10
  data_dir: ./data/torchvision
  batch_size: 128
  train_ratio: 0.8
```

CSV:
```yaml
dataset:
  type: custom
  path: data/train.csv
  input_columns: ["x1","x2","x3"]
  target_column: "y"
  batch_size: 128
  train_ratio: 0.8
```

JSON:
```yaml
dataset:
  type: custom
  path: data/data.json
  json_input_key: inputs
  json_target_key: targets
  batch_size: 128
  train_ratio: 0.8
```

NumPy:
```yaml
dataset:
  type: custom
  path: data/dataset.npz
  batch_size: 128
  train_ratio: 0.8
```

Notes:
- For legacy `type: mnist`, use `type: torchvision` and `name: mnist`.
- NumPy `.npz` files should contain arrays under keys `X` and `y`.

### 5) Model Settings

Adjust model architecture, optimizer, and training parameters under `model`:

```yaml
model:
  hidden_layers: [128, 64]
  dropout: 0.1
  learning_rate: 1e-3
  epochs: 10
  early_stopping_patience: 3
```

## How It Works

- EvoActiv generates candidate formulas as expression trees and compiles them to PyTorch functions (`activation_builder.py`).
- Safety wrappers (clamp, eps) are used in `log`, `/`, and `**` to keep training numerically stable.
- Each formula is evaluated by training a small `SimpleNet` on a dataset (`train_evaluator.py`).
- If `formula_optimization.enable` is set, constants in the formula become trainable parameters and are optimized jointly with the model weights.
- Evolution (`evo_core/evolution.py`) runs mutation, crossover, and selection, tracking the best formula by validation accuracy.

## Results and Analysis

- Each run saves a JSON summary under `results/`, e.g. `results_YYYYMMDD_HHMMSS.json`.
- Aggregated best formulas are appended to `results/formulas.json`.
- You can analyze results:

```
python analyze_results.py --results results/results_YYYYMMDD_HHMMSS.json --output results
```

## Tips

- For reproducibility, set seeds and deterministic flags if needed.
- For better search quality, increase `num_generations` and `population_size`.
- Ensure GPU drivers and CUDA-enabled PyTorch if you want faster training.

## Universal Dataset Loader

In `config/default.yaml`, define the `dataset` section, for example:

- CSV:
  - `type: custom`
  - `path: data/train.csv`
  - `input_columns: ["x1","x2","x3"]`
  - `target_column: "y"`
  - `batch_size: 128`
  - `train_ratio: 0.8`
- JSON:
  - `type: custom`
  - `path: data/data.json`
  - `json_input_key: "inputs"` (list of arrays)
  - `json_target_key: "targets"` (list of values or arrays)
- NumPy:
  - `type: custom`
  - `path: data/dataset.npz`
  - keys inside file: `X` and `y`
- Torchvision:
  - `type: torchvision`
  - `name: mnist` or `fashionmnist` or `cifar10`
  - `data_dir: data/torchvision`
  - `batch_size: 128`
  - `train_ratio: 0.8`

For legacy configs using `type: mnist`, use `type: torchvision` with `name: mnist`.

## Visualizations

- The `analyze_results.py` script plots fitness and diversity from saved history. Run:
  - `python analyze_results.py --results results/results_YYYYMMDD_HHMMSS.json`
- Plots are saved under `results/` (git-ignored), convenient for local analysis.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.