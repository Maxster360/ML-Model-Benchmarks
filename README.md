# ML Model Benchmarking on CPU/GPU Architectures

A comprehensive benchmarking framework for evaluating deep learning models (CNN, MLP) across CPU and GPU execution paths. Built with PyTorch, TensorFlow, NumPy, and Scikit-learn.

## Project Overview

This project profiles and compares the performance of Convolutional Neural Networks (CNN) and Multi-Layer Perceptrons (MLP) for image classification tasks. It measures inference latency, training throughput, and hardware utilization across different compute backends.

### Key Features

- **Dual Framework Support**: Side-by-side benchmarking with PyTorch and TensorFlow
- **CPU vs GPU Profiling**: Inference latency measurement across execution paths
- **GPU Kernel Analysis**: Memory bandwidth utilization, cache hit rates, and kernel execution patterns
- **Baseline Comparisons**: Scikit-learn models (Logistic Regression, Random Forest, SVM) for reference
- **Comprehensive Metrics**: Accuracy, F1 Score, AUC-ROC, and inference latency
- **Automated Visualization**: Charts and tables for benchmark results

## Project Structure

```
ML-Model-Benchmarks/
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py    # Data loading and feature engineering
│   ├── utils.py                 # Shared utilities and timing functions
│   ├── baselines.py             # Scikit-learn baseline models
│   ├── benchmark.py             # CPU/GPU inference benchmarking
│   ├── gpu_profiler.py          # GPU profiling and analysis
│   ├── visualize.py             # Results visualization
│   ├── train_pytorch.py         # PyTorch training pipeline
│   ├── train_tensorflow.py      # TensorFlow training pipeline
│   └── models/
│       ├── __init__.py
│       ├── pytorch_models.py    # PyTorch CNN and MLP
│       └── tensorflow_models.py # TensorFlow CNN and MLP
├── data/                        # Dataset storage (auto-downloaded)
├── results/                     # Benchmark outputs and charts
├── notebooks/                   # Jupyter notebook analysis
├── tests/                       # Unit tests
├── requirements.txt             # Python dependencies
├── LICENSE                      # MIT License
└── README.md
```

## Quick Start

### Prerequisites

- Python 3.9+
- CUDA-compatible GPU (optional, for GPU benchmarks)

### Installation

```bash
git clone https://github.com/Maxster360/ML-Model-Benchmarks.git
cd ML-Model-Benchmarks
pip install -r requirements.txt
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Deep Learning | PyTorch, TensorFlow |
| Data Processing | NumPy, Pandas |
| ML Baselines | Scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Profiling | psutil, GPUtil, CUDA Toolkit |

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.