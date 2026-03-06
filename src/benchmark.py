"""
CPU vs GPU Inference Benchmarking
==================================
Profiles inference latency for PyTorch and TensorFlow models across
CPU and GPU execution paths. Isolates pipeline bottlenecks and
memory-bound operations by varying batch sizes and model architectures.
"""

import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from src.models.pytorch_models import build_pytorch_model
from src.utils import (
    set_seed,
    get_device_info,
    Timer,
    measure_latency,
    log_section,
    ensure_dir,
    get_timestamp,
)


# ---------------------------------------------------------------------------
# PyTorch Inference Benchmarking
# ---------------------------------------------------------------------------

def benchmark_pytorch_inference(model_type="cnn", batch_sizes=None, warmup=10, repeats=100):
    """Benchmark PyTorch model inference on CPU and GPU.

    Runs the model forward pass with varying batch sizes, measuring
    latency statistics (mean, std, min, max) on each device.

    Args:
        model_type (str): 'cnn' or 'mlp'.
        batch_sizes (list[int]): Batch sizes to profile.
        warmup (int): Warmup iterations before timing.
        repeats (int): Number of timed iterations.

    Returns:
        list[dict]: Benchmark results per (device, batch_size) combination.
    """
    if batch_sizes is None:
        batch_sizes = [1, 8, 32, 64, 128, 256]

    set_seed(42)
    results = []

    devices_to_test = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices_to_test.append(torch.device("cuda"))

    for device in devices_to_test:
        model = build_pytorch_model(model_type=model_type, device=device)
        model.eval()

        log_section(f"PyTorch {model_type.upper()} on {device}")

        for bs in batch_sizes:
            # Create dummy input matching CIFAR-10 dimensions
            dummy_input = torch.randn(bs, 3, 32, 32, device=device)

            def forward_pass():
                with torch.no_grad():
                    _ = model(dummy_input)

            latency = measure_latency(forward_pass, warmup=warmup, repeats=repeats)

            # Compute throughput (images per second)
            throughput = (bs / latency["mean_ms"]) * 1000.0

            result = {
                "framework": "pytorch",
                "model_type": model_type,
                "device": str(device),
                "batch_size": bs,
                "mean_latency_ms": latency["mean_ms"],
                "std_latency_ms": latency["std_ms"],
                "min_latency_ms": latency["min_ms"],
                "max_latency_ms": latency["max_ms"],
                "throughput_img_per_sec": throughput,
            }
            results.append(result)

            print(
                f"  Batch {bs:>4d}:  "
                f"Mean={latency['mean_ms']:>8.2f} ms  "
                f"Std={latency['std_ms']:>6.2f} ms  "
                f"Throughput={throughput:>8.0f} img/s"
            )

    return results


# ---------------------------------------------------------------------------
# TensorFlow Inference Benchmarking
# ---------------------------------------------------------------------------

def benchmark_tensorflow_inference(model_type="cnn", batch_sizes=None, warmup=10, repeats=100):
    """Benchmark TensorFlow model inference on CPU and GPU.

    Args:
        model_type (str): 'cnn' or 'mlp'.
        batch_sizes (list[int]): Batch sizes to profile.
        warmup (int): Warmup iterations before timing.
        repeats (int): Number of timed iterations.

    Returns:
        list[dict]: Benchmark results per (device, batch_size) combination.
    """
    import tensorflow as tf
    from src.models.tensorflow_models import build_tensorflow_model

    if batch_sizes is None:
        batch_sizes = [1, 8, 32, 64, 128, 256]

    set_seed(42)
    results = []

    # Determine available devices
    devices_to_test = ["/CPU:0"]
    if tf.config.list_physical_devices("GPU"):
        devices_to_test.append("/GPU:0")

    for tf_device in devices_to_test:
        with tf.device(tf_device):
            model = build_tensorflow_model(model_type=model_type)

            # Build model with dummy forward pass
            dummy_build = tf.zeros((1, 32, 32, 3))
            _ = model(dummy_build, training=False)

            device_label = "GPU" if "GPU" in tf_device else "CPU"
            log_section(f"TensorFlow {model_type.upper()} on {device_label}")

            for bs in batch_sizes:
                dummy_input = tf.random.normal((bs, 32, 32, 3))

                # Warmup
                for _ in range(warmup):
                    _ = model(dummy_input, training=False)

                # Timed runs
                times = []
                for _ in range(repeats):
                    start = time.perf_counter()
                    _ = model(dummy_input, training=False)
                    times.append((time.perf_counter() - start) * 1000.0)

                times = np.array(times)
                mean_ms = float(np.mean(times))
                throughput = (bs / mean_ms) * 1000.0

                result = {
                    "framework": "tensorflow",
                    "model_type": model_type,
                    "device": device_label,
                    "batch_size": bs,
                    "mean_latency_ms": mean_ms,
                    "std_latency_ms": float(np.std(times)),
                    "min_latency_ms": float(np.min(times)),
                    "max_latency_ms": float(np.max(times)),
                    "throughput_img_per_sec": throughput,
                }
                results.append(result)

                print(
                    f"  Batch {bs:>4d}:  "
                    f"Mean={mean_ms:>8.2f} ms  "
                    f"Std={np.std(times):>6.2f} ms  "
                    f"Throughput={throughput:>8.0f} img/s"
                )

    return results


# ---------------------------------------------------------------------------
# Full Benchmark Suite
# ---------------------------------------------------------------------------

def run_inference_benchmarks(save_dir="./results", batch_sizes=None, warmup=10, repeats=100):
    """Run complete inference benchmarks across all models and frameworks.

    Profiles CNN and MLP architectures in both PyTorch and TensorFlow,
    on CPU and GPU (if available).

    Args:
        save_dir (str): Directory for saving benchmark results.
        batch_sizes (list[int]): Batch sizes to test.
        warmup (int): Warmup iterations.
        repeats (int): Timed iterations.

    Returns:
        dict: All benchmark results organized by framework and model.
    """
    if batch_sizes is None:
        batch_sizes = [1, 8, 32, 64, 128, 256]

    all_results = {
        "device_info": get_device_info(),
        "config": {
            "batch_sizes": batch_sizes,
            "warmup": warmup,
            "repeats": repeats,
        },
        "benchmarks": [],
    }

    log_section("INFERENCE BENCHMARK SUITE")

    # PyTorch benchmarks
    for model_type in ["cnn", "mlp"]:
        results = benchmark_pytorch_inference(
            model_type=model_type,
            batch_sizes=batch_sizes,
            warmup=warmup,
            repeats=repeats,
        )
        all_results["benchmarks"].extend(results)

    # TensorFlow benchmarks
    for model_type in ["cnn", "mlp"]:
        results = benchmark_tensorflow_inference(
            model_type=model_type,
            batch_sizes=batch_sizes,
            warmup=warmup,
            repeats=repeats,
        )
        all_results["benchmarks"].extend(results)

    # --- Save ---
    ensure_dir(save_dir)
    result_path = os.path.join(save_dir, "inference_benchmarks.json")
    with open(result_path, "w") as f:
        json.dump(all_results, f, indent=2)

    log_section("Benchmark Complete")
    print(f"  Total configs tested: {len(all_results['benchmarks'])}")
    print(f"  Results saved to: {result_path}")

    return all_results


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run inference benchmarks")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--repeats", type=int, default=100, help="Timed iterations")
    args = parser.parse_args()

    run_inference_benchmarks(warmup=args.warmup, repeats=args.repeats)
