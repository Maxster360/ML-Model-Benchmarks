"""
Results Visualization
======================
Generates publication-quality charts for benchmark results using
Matplotlib and Seaborn. Covers latency comparisons, accuracy metrics,
GPU memory profiles, and training curves.
"""

import json
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving figures
import seaborn as sns

from src.utils import ensure_dir, log_section


# Consistent style for all plots
sns.set_theme(style="whitegrid", palette="deep", font_scale=1.1)
COLORS = {
    "pytorch": "#EE4C2C",      # PyTorch red
    "tensorflow": "#FF6F00",   # TensorFlow orange
    "cpu": "#1976D2",          # Blue
    "gpu": "#43A047",          # Green
    "cnn": "#7B1FA2",          # Purple
    "mlp": "#00897B",          # Teal
}


# ---------------------------------------------------------------------------
# Inference Latency Charts
# ---------------------------------------------------------------------------

def plot_latency_comparison(benchmark_data, save_dir="./results"):
    """Plot inference latency across batch sizes for CPU vs GPU.

    Creates grouped bar charts comparing frameworks and devices.

    Args:
        benchmark_data (dict): Output from run_inference_benchmarks().
        save_dir (str): Directory to save the chart.
    """
    benchmarks = benchmark_data.get("benchmarks", [])
    if not benchmarks:
        print("  [SKIP] No benchmark data to plot.")
        return

    ensure_dir(save_dir)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, model_type in enumerate(["cnn", "mlp"]):
        ax = axes[idx]
        model_data = [b for b in benchmarks if b["model_type"] == model_type]

        # Group by framework-device
        groups = {}
        for b in model_data:
            key = f"{b['framework']}-{b['device']}"
            if key not in groups:
                groups[key] = {"batch_sizes": [], "latencies": []}
            groups[key]["batch_sizes"].append(b["batch_size"])
            groups[key]["latencies"].append(b["mean_latency_ms"])

        x_positions = np.arange(len(next(iter(groups.values()))["batch_sizes"]))
        width = 0.2
        offset = 0

        for label, data in groups.items():
            ax.bar(x_positions + offset * width, data["latencies"], width,
                   label=label, alpha=0.85)
            offset += 1

        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Latency (ms)")
        ax.set_title(f"{model_type.upper()} Inference Latency")
        batch_labels = groups[next(iter(groups))]["batch_sizes"]
        ax.set_xticks(x_positions + width * (len(groups) - 1) / 2)
        ax.set_xticklabels(batch_labels)
        ax.legend(fontsize=9)
        ax.set_yscale("log")

    plt.tight_layout()
    path = os.path.join(save_dir, "latency_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_throughput_comparison(benchmark_data, save_dir="./results"):
    """Plot throughput (images/sec) across batch sizes.

    Args:
        benchmark_data (dict): Output from run_inference_benchmarks().
        save_dir (str): Directory to save the chart.
    """
    benchmarks = benchmark_data.get("benchmarks", [])
    if not benchmarks:
        return

    ensure_dir(save_dir)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, model_type in enumerate(["cnn", "mlp"]):
        ax = axes[idx]
        model_data = [b for b in benchmarks if b["model_type"] == model_type]

        groups = {}
        for b in model_data:
            key = f"{b['framework']}-{b['device']}"
            if key not in groups:
                groups[key] = {"batch_sizes": [], "throughput": []}
            groups[key]["batch_sizes"].append(b["batch_size"])
            groups[key]["throughput"].append(b["throughput_img_per_sec"])

        for label, data in groups.items():
            ax.plot(data["batch_sizes"], data["throughput"], marker="o",
                    linewidth=2, label=label)

        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Throughput (images/sec)")
        ax.set_title(f"{model_type.upper()} Throughput")
        ax.legend(fontsize=9)
        ax.set_xscale("log", base=2)

    plt.tight_layout()
    path = os.path.join(save_dir, "throughput_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Model Accuracy Comparison
# ---------------------------------------------------------------------------

def plot_accuracy_comparison(results_dir="./results", save_dir="./results"):
    """Plot accuracy/F1/AUC comparison across all models.

    Reads JSON result files from the results directory and creates
    a grouped bar chart comparing all models and frameworks.

    Args:
        results_dir (str): Directory containing result JSON files.
        save_dir (str): Directory to save the chart.
    """
    ensure_dir(save_dir)

    # Collect metrics from all result files
    models_data = []

    # PyTorch results
    for model_type in ["cnn", "mlp"]:
        path = os.path.join(results_dir, f"pytorch_{model_type}_results.json")
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            models_data.append({
                "name": f"PyTorch {model_type.upper()}",
                "accuracy": data["test_metrics"]["accuracy"],
                "f1_score": data["test_metrics"]["f1_score"],
                "auc_roc": data["test_metrics"]["auc_roc"],
            })

    # TensorFlow results
    for model_type in ["cnn", "mlp"]:
        path = os.path.join(results_dir, f"tensorflow_{model_type}_results.json")
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            models_data.append({
                "name": f"TF {model_type.upper()}",
                "accuracy": data["test_metrics"]["accuracy"],
                "f1_score": data["test_metrics"]["f1_score"],
                "auc_roc": data["test_metrics"]["auc_roc"],
            })

    # Baseline results
    baseline_path = os.path.join(results_dir, "baseline_results.json")
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            baselines = json.load(f)
        for b in baselines:
            models_data.append({
                "name": b["model_name"][:15],
                "accuracy": b["metrics"]["accuracy"],
                "f1_score": b["metrics"]["f1_score"],
                "auc_roc": b["metrics"]["auc_roc"],
            })

    if not models_data:
        print("  [SKIP] No result files found for accuracy comparison.")
        return

    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 6))

    model_names = [d["name"] for d in models_data]
    x = np.arange(len(model_names))
    width = 0.25

    acc_vals = [d["accuracy"] for d in models_data]
    f1_vals = [d["f1_score"] for d in models_data]
    auc_vals = [d["auc_roc"] for d in models_data]

    ax.bar(x - width, acc_vals, width, label="Accuracy", color="#1976D2")
    ax.bar(x, f1_vals, width, label="F1 Score", color="#43A047")
    ax.bar(x + width, auc_vals, width, label="AUC-ROC", color="#FF6F00")

    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=30, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    path = os.path.join(save_dir, "accuracy_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Training Curves
# ---------------------------------------------------------------------------

def plot_training_curves(results_dir="./results", save_dir="./results"):
    """Plot training loss and accuracy curves for DL models.

    Args:
        results_dir (str): Directory containing result JSON files.
        save_dir (str): Directory to save the chart.
    """
    ensure_dir(save_dir)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    files = {
        "PyTorch CNN": "pytorch_cnn_results.json",
        "PyTorch MLP": "pytorch_mlp_results.json",
        "TF CNN": "tensorflow_cnn_results.json",
        "TF MLP": "tensorflow_mlp_results.json",
    }

    has_data = False
    for label, filename in files.items():
        path = os.path.join(results_dir, filename)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)

        history = data.get("history", {})
        if not history:
            continue

        has_data = True
        epochs = range(1, len(history["train_loss"]) + 1)

        axes[0].plot(epochs, history["train_loss"], marker=".", label=label)
        axes[1].plot(epochs, history["train_accuracy"], marker=".", label=label)

    if not has_data:
        print("  [SKIP] No training history data found.")
        plt.close()
        return

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend()

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training Accuracy")
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# GPU Memory Profile Chart
# ---------------------------------------------------------------------------

def plot_gpu_memory_profile(gpu_data, save_dir="./results"):
    """Plot GPU memory usage across batch sizes.

    Args:
        gpu_data (dict): Output from run_gpu_profiling().
        save_dir (str): Directory to save the chart.
    """
    mem_profiles = gpu_data.get("memory_profiles", [])
    if not mem_profiles:
        print("  [SKIP] No GPU memory data to plot.")
        return

    ensure_dir(save_dir)
    fig, ax = plt.subplots(figsize=(10, 6))

    for model_type in ["cnn", "mlp"]:
        model_data = [m for m in mem_profiles if m["model_type"] == model_type]
        if not model_data:
            continue

        batch_sizes = [m["batch_size"] for m in model_data]
        peak_mem = [m["memory_peak_mb"] for m in model_data]

        ax.plot(batch_sizes, peak_mem, marker="s", linewidth=2,
                label=f"{model_type.upper()} Peak Memory",
                color=COLORS.get(model_type, "#333"))

    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Peak GPU Memory (MB)")
    ax.set_title("GPU Memory Usage vs Batch Size")
    ax.legend()

    plt.tight_layout()
    path = os.path.join(save_dir, "gpu_memory_profile.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Generate All Visualizations
# ---------------------------------------------------------------------------

def generate_all_visualizations(results_dir="./results", save_dir="./results"):
    """Generate all benchmark visualization charts.

    Reads result files from the results directory and creates
    comprehensive comparison charts.

    Args:
        results_dir (str): Directory containing benchmark result JSONs.
        save_dir (str): Directory for saving generated charts.
    """
    log_section("GENERATING VISUALIZATIONS")

    # Accuracy and training curves (from individual model results)
    plot_accuracy_comparison(results_dir=results_dir, save_dir=save_dir)
    plot_training_curves(results_dir=results_dir, save_dir=save_dir)

    # Inference benchmark charts
    bench_path = os.path.join(results_dir, "inference_benchmarks.json")
    if os.path.exists(bench_path):
        with open(bench_path) as f:
            bench_data = json.load(f)
        plot_latency_comparison(bench_data, save_dir=save_dir)
        plot_throughput_comparison(bench_data, save_dir=save_dir)

    # GPU profiling charts
    gpu_path = os.path.join(results_dir, "gpu_profiling_results.json")
    if os.path.exists(gpu_path):
        with open(gpu_path) as f:
            gpu_data = json.load(f)
        plot_gpu_memory_profile(gpu_data, save_dir=save_dir)

    log_section("Visualization Complete")
    print(f"  Charts saved to: {save_dir}/")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    generate_all_visualizations()
