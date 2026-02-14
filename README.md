# AI-Driven Power Flow Prediction for Smart Grid Systems

A machine learning approach to approximate AC power flow solutions for the IEEE 9-bus system, replacing iterative numerical solvers with neural networks for near-instantaneous inference.

## Overview

AC power flow analysis is fundamental to power system operations but relies on iterative solvers (e.g., Newton-Raphson) that scale poorly with system size. This project trains neural networks to predict bus voltage magnitudes and angles directly from load conditions, achieving test MSE of **1.85 × 10⁻⁷** — orders of magnitude faster than traditional methods at inference time.

### Problem Formulation

- **Input** (6-dim): Active and reactive power loads at buses 5, 7, 9 → `[P₅, Q₅, P₇, Q₇, P₉, Q₉]`
- **Output** (18-dim): Voltage angles and magnitudes at all 9 buses → `[θ₁...θ₉, Vm₁...Vm₉]`
- **Training data**: Generated via Fast Decoupled Load Flow (FDLF) algorithm implemented from scratch

## Project Structure

```
├── data/
│   ├── case9.m                # IEEE 9-bus system parameters (MATPOWER format)
│   ├── generate_data.py       # FDLF solver & dataset generation
│   ├── runPF.m                # MATLAB reference implementation
│   ├── train_dataset.npz      # Training set (10,000 samples)
│   ├── val_dataset.npz        # Validation set (2,000 samples)
│   └── test_dataset.npz       # Test set (2,000 samples)
├── models/
│   ├── checkpoints/           # Saved model weights
│   ├── evaluate_models.py     # Model evaluation & metrics
│   ├── mlp_models.py          # Neural network architectures
│   └── train.py               # Training pipeline with early stopping & LR scheduling
├── ESE_562_Final_Project_Report.pdf
└── requirements.txt           # Python dependencies

```

## Key Results

| Model | Loss Type | Test MSE |
|-------|-----------|----------|
| **Baseline MLP** | **MSE** | **2.61 × 10⁻⁵** |
| Baseline MLP | Physics-informed | 2.89 × 10⁻⁵ |
| Improved MLP | MSE | 2.98 × 10⁻⁵ |
| Physics-informed MLP | MSE | 6.81 × 10⁻⁵ |

After hyperparameter sweep (45 configurations), the best model (2-layer MLP, 128 neurons, lr=0.0001) achieved **1.85 × 10⁻⁷** test MSE.

**Key findings:**
- Simple architectures outperform complex ones (BatchNorm, Dropout, physics-informed loss all degraded performance)
- Smaller learning rates consistently yielded better results
- The smooth, low-dimensional nature of power flow equations favors modest-size MLPs

## Getting Started

### Prerequisites

```bash
pip install torch numpy matplotlib scipy
```

### Generate Training Data

```bash
python generate_data.py
```

This runs the FDLF solver to produce:
- Training set: 10,000 samples (load range: 0.5×–1.5× nominal)
- Validation set: 2,000 samples (load range: 0.5×–1.5× nominal)
- Test set: 2,000 samples (load range: 0.3×–1.7× nominal, wider for generalization testing)

All 14,000 samples converge with tolerance ε = 10⁻⁸, validated against MATLAB's `runpf`.

### Train Model

```bash
python train.py
```

Default configuration:
- Optimizer: Adam (lr=1e-3) with ReduceLROnPlateau scheduler
- Batch size: 64
- Early stopping: patience=15 epochs
- Checkpoints saved to `checkpoints/`

### MATLAB Reference

To validate against MATPOWER:

```matlab
run('case9.m')
run('runPF.m')
```

## Technical Details

### Fast Decoupled Load Flow (FDLF)

The FDLF algorithm exploits the decoupling between P-θ and Q-V relationships in power systems:

```
Δθ = -(B')⁻¹ · ΔP    (angle update)
ΔVm = -(B'')⁻¹ · ΔQ   (magnitude update)
```

Our Python implementation (`generate_data.py`) achieves 100% convergence across all samples, with results matching the MATLAB reference solver.

### Neural Network Architectures

Three architectures were compared:

1. **Baseline MLP**: 3 hidden layers [64, 128, 64], ReLU, no regularization (~18K params)
2. **Improved MLP**: [128, 256, 128] with BatchNorm + Dropout (~58K params)
3. **Physics-informed MLP**: Dual-head design separating angle/magnitude prediction (~120K params)

### Hyperparameter Search

Systematic grid search over 45 configurations varying:
- Hidden layers: 2–4 layers
- Width: 64–256 neurons
- Learning rate: 0.0001–0.01
- Epochs: 100–150

## Tech Stack

- **Python**, **PyTorch** — model development and training
- **NumPy**, **SciPy** — numerical computing and FDLF implementation
- **Matplotlib** — visualization
- **MATLAB/MATPOWER** — reference solver validation
- **Weights & Biases** — experiment tracking and hyperparameter sweep

## Authors

- **Guoao Li** — Stony Brook University
- **Lichen Yang** — Stony Brook University

## References

- Chow, J.H. (1982). *Time-Scale Modeling of Dynamic Networks with Applications to Power Systems*. Springer-Verlag.
- Schulz, R.P., Turner, A.E., & Ewart, D.N. (1974). *Long Term Power System Dynamics*. EPRI Report 90-7-0.
