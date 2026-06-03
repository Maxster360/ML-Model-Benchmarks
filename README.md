# ML Model Benchmarking on CPU/GPU Architectures

> **Python | PyTorch | TensorFlow | NumPy | Scikit-learn**

A comprehensive benchmarking framework for evaluating deep learning models (CNN, MLP) across CPU and GPU execution paths. Profiles inference latency, training throughput, and GPU hardware utilization to identify pipeline bottlenecks and memory-bound operations.

## Project Overview

This project trains and evaluates Convolutional Neural Networks (CNN) and Multi-Layer Perceptrons (MLP) for image classification on the **CIFAR-10** dataset using both PyTorch and TensorFlow. It then profiles inference performance across CPU and GPU backends, compares against traditional ML baselines, and generates publication-quality visualizations of all results.

### Key Features

- **Dual Framework Training**: Side-by-side CNN/MLP training with PyTorch and TensorFlow
- **CPU vs GPU Inference Profiling**: Latency measurement across execution paths with varying batch sizes
- **GPU Kernel Analysis**: CUDA event-based kernel timing, memory bandwidth estimation, and allocation tracking
- **Baseline Comparisons**: Scikit-learn models (Logistic Regression, Random Forest, SVM) for reference
- **Comprehensive Metrics**: Accuracy, F1 Score, AUC-ROC, throughput (img/s), and latency statistics
- **Automated Visualization**: Matplotlib/Seaborn charts for latency, throughput, accuracy, training curves, and GPU memory

## Project Structure

```
ML-Model-Benchmarks/
├── main.py                      # Main orchestrator (runs full pipeline)
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py    # CIFAR-10 loading, feature engineering, subsets
│   ├── utils.py                 # Timing, device detection, logging, reproducibility
│   ├── baselines.py             # Logistic Regression, Random Forest, SVM
│   ├── benchmark.py             # CPU/GPU inference latency profiling
│   ├── gpu_profiler.py          # GPU memory, kernel timing, bandwidth analysis
│   ├── visualize.py             # Chart generation (latency, accuracy, GPU memory)
│   ├── train_pytorch.py         # PyTorch training pipeline with metrics
│   ├── train_tensorflow.py      # TensorFlow training pipeline with metrics
│   └── models/
│       ├── __init__.py
│       ├── pytorch_models.py    # PyTorch CNN (3-block) and MLP (3-layer)
│       └── tensorflow_models.py # TensorFlow/Keras CNN and MLP (matched arch)
├── notebooks/
│   ├── benchmark_analysis.ipynb # Interactive analysis and exploration
│   └── run_on_gpu_colab.ipynb   # One-click GPU run on Google Colab
├── tests/
│   └── test_models.py           # Unit tests for model architectures
├── data/                        # Dataset cache (auto-downloaded)
├── results/                     # JSON results + PNG charts (auto-generated)
├── requirements.txt             # Python dependencies
└── LICENSE                      # MIT License
```

## Quick Start

### Prerequisites

- Python 3.9+
- pip (Python package manager)
- CUDA-compatible GPU + CUDA Toolkit (optional, for GPU benchmarks)

### Installation

```bash
git clone https://github.com/Maxster360/ML-Model-Benchmarks.git
cd ML-Model-Benchmarks
pip install -r requirements.txt
```

### Running the Benchmarks

```bash
# Full pipeline (training + baselines + benchmarks + GPU profiling + charts)
python main.py

# Quick mode (3 epochs, 10% data subset, fewer repeats)
python main.py --quick

# Skip training, only run benchmarks on existing models
python main.py --skip-training

# Custom configuration
python main.py --epochs 20 --batch-size 128 --lr 0.0005
```

### Running on a GPU (Colab)

This machine doesn't need a GPU to run, but the GPU profiling and CPU-vs-GPU
inference comparison only produce numbers when a CUDA device is present. The
quickest way to get those is the bundled Colab notebook
[`notebooks/run_on_gpu_colab.ipynb`](notebooks/run_on_gpu_colab.ipynb):

1. Open it in [Google Colab](https://colab.research.google.com/github/Maxster360/ML-Model-Benchmarks/blob/main/notebooks/run_on_gpu_colab.ipynb).
2. **Runtime → Change runtime type → GPU**, then **Runtime → Run all**.

It clones the repo, installs only the deps Colab is missing (Colab already ships
CUDA-enabled torch/TensorFlow), runs the inference benchmarks + GPU profiler, and
prints the **CPU-vs-GPU inference speedup at batch 128**. The optional full-pipeline
cell trains all four models on the GPU and regenerates every result file.

### Running Individual Components

```bash
# Train only PyTorch models
python -m src.train_pytorch --model cnn --epochs 15
python -m src.train_pytorch --model mlp --epochs 15

# Train only TensorFlow models
python -m src.train_tensorflow --model cnn --epochs 15

# Run sklearn baselines (10% subset for speed)
python -m src.baselines --subset 0.1

# Run inference benchmarks only
python -m src.benchmark --warmup 10 --repeats 100

# Run GPU profiling only
python -m src.gpu_profiler

# Generate visualizations from existing results
python -m src.visualize
```

### Running Tests

```bash
python -m pytest tests/ -v
```

## Methodology

### Dataset
- **CIFAR-10**: 60,000 32x32 color images across 10 classes (50K train / 10K test)
- Standard normalization (per-channel mean/std) for CNN inputs
- Flattened + StandardScaler for traditional ML baselines

### Deep Learning Models

| Model | Architecture | Parameters | Framework |
|-------|-------------|-----------|-----------|
| CNN | 3x Conv-BN-ReLU-Pool + 2x FC + Dropout | ~621K | PyTorch, TensorFlow |
| MLP | 3x FC-BN-ReLU-Dropout + FC output | ~1.74M | PyTorch, TensorFlow |

- **Optimizer**: Adam (lr=0.001) with StepLR scheduler
- **Loss**: Cross-Entropy
- **Regularization**: Dropout (0.2-0.5), BatchNorm

### Baseline Models
- **Logistic Regression**: SAGA solver, multinomial
- **Random Forest**: 100 trees, max_depth=20
- **SVM**: RBF kernel, C=10, probability=True

### Benchmarking Metrics
- **Accuracy, F1 Score (weighted), AUC-ROC (one-vs-rest)**
- **Inference Latency**: Mean, std, min, max over 100 timed runs (after warmup)
- **Throughput**: Images processed per second at each batch size
- **GPU Memory**: Allocated, peak, reserved memory per batch size
- **Kernel Timing**: CUDA event-based GPU kernel execution time vs CPU wall time
- **Memory Bandwidth**: Effective data throughput (GB/s) during inference

## Generated Outputs

After running `python main.py`, the `results/` directory will contain:

| File | Description |
|------|------------|
| `pytorch_cnn_results.json` | PyTorch CNN training metrics and history |
| `pytorch_mlp_results.json` | PyTorch MLP training metrics and history |
| `tensorflow_cnn_results.json` | TensorFlow CNN training metrics and history |
| `tensorflow_mlp_results.json` | TensorFlow MLP training metrics and history |
| `baseline_results.json` | Scikit-learn baseline comparison results |
| `inference_benchmarks.json` | CPU vs GPU inference latency data |
| `gpu_profiling_results.json` | GPU memory, kernel, and bandwidth profiles |
| `latency_comparison.png` | Inference latency bar chart |
| `throughput_comparison.png` | Throughput curves across batch sizes |
| `accuracy_comparison.png` | Model accuracy/F1/AUC comparison chart |
| `training_curves.png` | Training loss and accuracy over epochs |
| `gpu_memory_profile.png` | GPU memory usage vs batch size |

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Deep Learning | PyTorch 2.0+, TensorFlow 2.13+ | CNN/MLP training and inference |
| Data Processing | NumPy 1.24+, Pandas 2.0+ | Feature engineering, data manipulation |
| ML Baselines | Scikit-learn 1.3+ | Traditional classifier benchmarks |
| Visualization | Matplotlib 3.7+, Seaborn 0.12+ | Publication-quality charts |
| Profiling | psutil, GPUtil, CUDA Events | Hardware utilization metrics |
| Notebook | Jupyter | Interactive analysis |

## Key Findings

The numbers below are from a **quick-mode run** (`python main.py --quick`: 3 epochs, 10%
CIFAR-10 subset — 5,000 train / 1,000 test — on a 20-core CPU, **no CUDA GPU present**).
They are intentionally modest: 3 epochs on a 10% subset is far from convergence. A full
run (`python main.py`, 10 epochs on all 50K images, ideally on a GPU) will produce higher
accuracies and is expected to change the relative ordering. All raw numbers and the charts
that visualize them live in [`results/`](results/).

### Classification metrics (test set)

| Model | Accuracy | F1 (weighted) | AUC-ROC |
|-------|:--------:|:-------------:|:-------:|
| PyTorch CNN | **0.443** | 0.429 | **0.877** |
| PyTorch MLP | 0.327 | 0.320 | 0.794 |
| TensorFlow CNN | 0.125 | 0.034 | 0.756 |
| TensorFlow MLP | 0.304 | 0.296 | 0.795 |
| SVM (RBF) | 0.445 | 0.446 | 0.859 |
| Random Forest | 0.381 | 0.378 | 0.809 |
| Logistic Regression | 0.278 | 0.278 | 0.705 |

### CPU inference throughput (images/sec, batch size 256)

| Model | PyTorch | TensorFlow |
|-------|:-------:|:----------:|
| CNN | 8,172 | 3,276 |
| MLP | 88,261 | 12,742 |

### Observations

- **CNN beats MLP in PyTorch**: the PyTorch CNN (0.443 acc, 0.877 AUC) clearly outperforms
  the PyTorch MLP (0.327 acc, 0.794 AUC), consistent with spatial feature extraction helping
  on images even at this small scale.
- **MLP is far cheaper to run than CNN on CPU**: despite having more parameters (~1.74M vs
  ~621K), the MLP reaches ~88K img/s vs the CNN's ~8K img/s in PyTorch at batch 256 — the CNN
  is compute-bound by its convolutions, while the MLP is a handful of dense matmuls.
- **Throughput scales with batch size**: PyTorch CNN rises from ~961 img/s (batch 1) to
  ~8,172 img/s (batch 256); the MLP from ~2,934 to ~88,261 img/s.
- **PyTorch is much faster than TensorFlow for CPU inference here**: at batch 1, PyTorch CNN
  latency is ~1.0 ms vs TensorFlow's ~12.4 ms (eager-mode per-call overhead dominates at
  small batches). The gap narrows as batch size grows.
- **Classical baselines are competitive at this scale**: the RBF SVM (0.445 acc) matches the
  best neural network in this short run, and Random Forest (0.381) beats both MLPs — with only
  3 epochs on 5K images, the deep nets have no real advantage yet.
- **TensorFlow CNN under-performed on the test set** (0.125 acc despite ~0.41 train acc and
  0.756 AUC). This is a BatchNorm running-statistics artifact: with so few update steps the
  inference-time BN statistics are poorly estimated, collapsing the arg-max predictions even
  though the model's ranking (AUC) is reasonable. It is expected to resolve with more epochs.
- **GPU metrics were not measured**: this run was CPU-only, so `gpu_profiling_results.json`
  is empty and the GPU memory chart is skipped. Re-run on a CUDA machine to populate kernel
  timing, bandwidth, and memory profiles — see [Running on a GPU](#running-on-a-gpu-colab).

## Author

**Mathew Sabu** - [GitHub](https://github.com/Maxster360)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.