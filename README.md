# EvoActiv

Evolutionary discovery and optimization of activation functions for neural networks.

## Overview

EvoActiv uses evolutionary search to discover novel activation formulas for neural networks. It builds expression trees, mutates and crosses them, and evaluates candidate formulas by training a simple model on a dataset (MNIST by default). Beyond standard functions like ReLU, sigmoid, or tanh, EvoActiv explores composite and parametric formulas that can be trained alongside the model.

## Highlights

- Evolutionary search of activation formulas using genetic programming.
- Trainable numeric constants in formulas (become `nn.Parameter`s).
- Operator set fully configurable via `config/default.yaml`.
- Configurable formula optimization modes: `joint`, `finetune_only`, `none`.
- Safe activation validation to reject NaN/Inf values and bad gradients.
- Early stopping, logging, and JSON results with parameter traces.
- CPU/GPU support via PyTorch; automatic device selection.
- Result analysis script to visualize and compare top formulas.

## Project Structure

```
EvoActiv/
├── data/                     # Datasets (created on first run)
├── results/                  # JSON, plots, models (created on run)
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
git clone https://github.com/AlexSheff/EvoActiv.git
cd EvoActiv
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: . .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Dependency notes:
- Versions are pinned for compatibility: `numpy>=1.25,<2.0`, `torch>=2.0,<3.0`, `torchvision>=0.15,<0.16`.
- MNIST and other torchvision datasets are downloaded on first run.

## Quick Start

Run a small evolutionary search with default settings:

```
python evolve.py --config config/default.yaml --output results
```

Run a fast smoke test (1 generation, tiny population):

```
python evolve.py --config config/default.yaml --generations 1 --population_size 1
```

Increase search quality (more exploration):

```
python evolve.py --config config/default.yaml --generations 50 --population_size 100
```

## Configuration Reference

The full experiment is controlled by `config/default.yaml`.

```yaml
evolution:
  population_size: 100
  num_generations: 50
  mutation_rate: 0.2
  crossover_rate: 0.7
  elitism_count: 5
  tournament_size: 3
  max_formula_depth: 5
  seed: 42

operators:
  unary: ["sin", "cos", "exp", "log", "tanh", "sigmoid", "relu"]
  binary: ["+", "-", "*", "/", "**"]
  trainable_constants: true

dataset:
  type: "mnist"        # Options: mnist, torchvision, custom
  batch_size: 64
  train_ratio: 0.8
  data_dir: "../data"

model:
  hidden_layers: [128, 64]
  dropout_rate: 0.2
  learning_rate: 0.001
  weight_decay: 0.0001
  epochs: 10
  early_stopping_patience: 3

formula_optimization:
  enable: true          # Enables trainable constants in formulas
  mode: joint           # Options: joint, finetune_only, none
  learning_rate: 0.01   # LR for formula params (if set)
  parameter_lr_mult: 10.0  # Multiplier relative to model LR (if LR not set)
  epochs: 50            # Fine-tuning epochs when enabled
  coefficient_range: [-5.0, 5.0]  # Initial constant range for generation

fine_tuning:
  enable: true
  top_k: 3
  extra_epochs: 5

logging:
  log_interval: 10
  save_top_k: 5
  results_dir: "../results"
  save_models: true
  save_plots: true
  verbose: true
```

Key points:
- `operators` define the available unary/binary operations used in formula generation.
- `trainable_constants` controls whether numeric constants become parameters during training.
- `coefficient_range` governs the initial constant values during random tree generation.
- `formula_optimization.mode` selects training strategy:
  - `joint`: train model and activation parameters together.
  - `finetune_only`: freeze model, optimize only activation parameters.
  - `none`: disable trainable constants.
- `seed` ensures reproducible runs across `random`, `numpy`, and `torch`.

## Dataset Configuration (Universal Loader)

Switch dataset type or point to a custom CSV/JSON/NumPy dataset.

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
- Legacy `type: mnist` is supported; prefer `type: torchvision` + `name: mnist`.
- NumPy `.npz` files should contain arrays under keys `X` and `y`.

## How It Works

- EvoActiv generates candidate formulas as expression trees and compiles them to PyTorch modules (`activation_builder.py`).
- Safety check validates each activation on a random batch to reject NaN/Inf or bad gradients.
- Each formula is evaluated by training `SimpleNet` on the chosen dataset (`train_evaluator.py`).
- If formula optimization is enabled, constants become trainable parameters and are optimized according to the selected `mode`.
- Evolution (`evo_core/evolution.py`) runs mutation, crossover, selection, and logs per-generation stats (fitness and diversity).

## Results and Analysis

- Each run saves a JSON summary under `results/`, e.g. `results_YYYYMMDD_HHMMSS.json`.
- Aggregated best formulas are appended to `results/formulas.json`.
- Analyze results:

```
python analyze_results.py --results results/results_YYYYMMDD_HHMMSS.json --output results
```

## Troubleshooting

- NumPy 2.0 incompatibility: dependencies are pinned to `numpy<2.0` in `requirements.txt` to avoid known issues.
- If dataset download fails, check network and `data_dir` paths; retry after clearing partial downloads.
- If training is slow, reduce `model.epochs`, `batch_size`, or run with `--generations 1 --population_size 1`.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
