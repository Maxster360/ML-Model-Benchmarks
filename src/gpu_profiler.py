"""
GPU Profiling & Hardware Analysis
===================================
Profiles GPU kernel execution patterns, memory bandwidth utilization,
and cache behavior. Directly relevant to GPU software optimization
and driver-level performance tuning workflows.

Note: Full CUDA profiling requires NVIDIA GPU + CUDA toolkit.
      This module gracefully degrades on CPU-only systems.
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
    log_section,
    log_metrics,
    ensure_dir,
)


# ---------------------------------------------------------------------------
# GPU Memory Profiling
# ---------------------------------------------------------------------------

def profile_gpu_memory(model_type="cnn", batch_sizes=None):
    """Profile GPU memory allocation patterns during inference.

    Measures memory allocated, reserved, and peak usage for varying
    batch sizes to identify memory-bound operations.

    Args:
        model_type (str): 'cnn' or 'mlp'.
        batch_sizes (list[int]): Batch sizes to test.

    Returns:
        list[dict]: Memory profile per batch size, or empty if no GPU.
    """
    if not torch.cuda.is_available():
        print("  [SKIP] No CUDA GPU detected. Memory profiling requires GPU.")
        return []

    if batch_sizes is None:
        batch_sizes = [1, 8, 32, 64, 128, 256]

    device = torch.device("cuda")
    results = []

    log_section(f"GPU Memory Profile: {model_type.upper()}")

    for bs in batch_sizes:
        # Reset memory stats
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        model = build_pytorch_model(model_type=model_type, device=device)
        model.eval()

        dummy_input = torch.randn(bs, 3, 32, 32, device=device)

        # Baseline memory (model loaded, no inference yet)
        mem_before = torch.cuda.memory_allocated() / (1024 ** 2)  # MB

        with torch.no_grad():
            _ = model(dummy_input)

        mem_after = torch.cuda.memory_allocated() / (1024 ** 2)
        mem_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
        mem_reserved = torch.cuda.memory_reserved() / (1024 ** 2)

        result = {
            "model_type": model_type,
            "batch_size": bs,
            "memory_before_mb": round(mem_before, 2),
            "memory_after_mb": round(mem_after, 2),
            "memory_peak_mb": round(mem_peak, 2),
            "memory_reserved_mb": round(mem_reserved, 2),
            "memory_delta_mb": round(mem_after - mem_before, 2),
        }
        results.append(result)

        print(
            f"  Batch {bs:>4d}:  "
            f"Alloc={mem_after:>7.1f} MB  "
            f"Peak={mem_peak:>7.1f} MB  "
            f"Reserved={mem_reserved:>7.1f} MB  "
            f"Delta={mem_after - mem_before:>6.1f} MB"
        )

        # Cleanup for next iteration
        del model, dummy_input
        torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# GPU Kernel Timing (using CUDA events)
# ---------------------------------------------------------------------------

def profile_kernel_timing(model_type="cnn", batch_size=64, repeats=50):
    """Measure GPU kernel execution time using CUDA events.

    CUDA events provide precise GPU-side timing, excluding CPU overhead
    and kernel launch latency. This reveals the true compute-bound time.

    Args:
        model_type (str): 'cnn' or 'mlp'.
        batch_size (int): Batch size for profiling.
        repeats (int): Number of iterations.

    Returns:
        dict: GPU kernel timing statistics, or None if no GPU.
    """
    if not torch.cuda.is_available():
        print("  [SKIP] No CUDA GPU detected. Kernel timing requires GPU.")
        return None

    device = torch.device("cuda")
    model = build_pytorch_model(model_type=model_type, device=device)
    model.eval()

    dummy_input = torch.randn(batch_size, 3, 32, 32, device=device)

    log_section(f"GPU Kernel Timing: {model_type.upper()} (batch={batch_size})")

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)

    # Profile with CUDA events
    gpu_times = []
    cpu_times = []

    for _ in range(repeats):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        cpu_start = time.perf_counter()
        start_event.record()

        with torch.no_grad():
            _ = model(dummy_input)

        end_event.record()
        torch.cuda.synchronize()
        cpu_end = time.perf_counter()

        gpu_times.append(start_event.elapsed_time(end_event))  # ms
        cpu_times.append((cpu_end - cpu_start) * 1000.0)

    gpu_times = np.array(gpu_times)
    cpu_times = np.array(cpu_times)

    result = {
        "model_type": model_type,
        "batch_size": batch_size,
        "gpu_kernel_mean_ms": float(np.mean(gpu_times)),
        "gpu_kernel_std_ms": float(np.std(gpu_times)),
        "gpu_kernel_min_ms": float(np.min(gpu_times)),
        "gpu_kernel_max_ms": float(np.max(gpu_times)),
        "cpu_wall_mean_ms": float(np.mean(cpu_times)),
        "cpu_wall_std_ms": float(np.std(cpu_times)),
        "kernel_launch_overhead_ms": float(np.mean(cpu_times) - np.mean(gpu_times)),
    }

    print(f"  GPU Kernel Time : {result['gpu_kernel_mean_ms']:.3f} +/- {result['gpu_kernel_std_ms']:.3f} ms")
    print(f"  CPU Wall Time   : {result['cpu_wall_mean_ms']:.3f} +/- {result['cpu_wall_std_ms']:.3f} ms")
    print(f"  Launch Overhead : {result['kernel_launch_overhead_ms']:.3f} ms")

    return result


# ---------------------------------------------------------------------------
# Memory Bandwidth Estimation
# ---------------------------------------------------------------------------

def estimate_memory_bandwidth(model_type="cnn", batch_size=64, repeats=50):
    """Estimate effective memory bandwidth during inference.

    Calculates the ratio of data movement (input + parameters + output)
    to execution time, providing an estimate of memory bandwidth utilization.

    Args:
        model_type (str): 'cnn' or 'mlp'.
        batch_size (int): Batch size for profiling.
        repeats (int): Number of iterations.

    Returns:
        dict: Bandwidth estimates, or None if no GPU.
    """
    if not torch.cuda.is_available():
        print("  [SKIP] No CUDA GPU detected. Bandwidth estimation requires GPU.")
        return None

    device = torch.device("cuda")
    model = build_pytorch_model(model_type=model_type, device=device)
    model.eval()

    log_section(f"Memory Bandwidth: {model_type.upper()} (batch={batch_size})")

    # Calculate data sizes
    input_bytes = batch_size * 3 * 32 * 32 * 4  # float32
    param_bytes = sum(p.numel() * 4 for p in model.parameters())  # float32
    output_bytes = batch_size * 10 * 4  # float32

    total_data_bytes = input_bytes + param_bytes + output_bytes
    total_data_mb = total_data_bytes / (1024 ** 2)

    dummy_input = torch.randn(batch_size, 3, 32, 32, device=device)

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)

    # Time with CUDA events
    times = []
    for _ in range(repeats):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        with torch.no_grad():
            _ = model(dummy_input)
        end_event.record()
        torch.cuda.synchronize()

        times.append(start_event.elapsed_time(end_event))

    mean_time_ms = float(np.mean(times))
    mean_time_s = mean_time_ms / 1000.0

    # Bandwidth = data moved / time
    bandwidth_gb_s = (total_data_bytes / (1024 ** 3)) / mean_time_s

    # Get theoretical peak bandwidth for reference
    gpu_props = torch.cuda.get_device_properties(0)

    result = {
        "model_type": model_type,
        "batch_size": batch_size,
        "input_data_mb": round(input_bytes / (1024 ** 2), 4),
        "parameter_data_mb": round(param_bytes / (1024 ** 2), 4),
        "total_data_mb": round(total_data_mb, 4),
        "mean_exec_time_ms": mean_time_ms,
        "effective_bandwidth_gb_s": round(bandwidth_gb_s, 2),
        "gpu_name": gpu_props.name,
        "gpu_total_memory_gb": round(gpu_props.total_mem / (1024 ** 3), 2),
    }

    print(f"  Total Data Moved   : {total_data_mb:.4f} MB")
    print(f"  Mean Exec Time     : {mean_time_ms:.3f} ms")
    print(f"  Effective Bandwidth: {bandwidth_gb_s:.2f} GB/s")
    print(f"  GPU                : {gpu_props.name}")

    return result


# ---------------------------------------------------------------------------
# Full GPU Profiling Suite
# ---------------------------------------------------------------------------

def run_gpu_profiling(save_dir="./results"):
    """Run the complete GPU profiling suite.

    Profiles memory usage, kernel timing, and bandwidth estimation
    for both CNN and MLP architectures.

    Args:
        save_dir (str): Directory for saving profiling results.

    Returns:
        dict: All profiling results.
    """
    set_seed(42)
    device_info = get_device_info()

    log_section("GPU PROFILING SUITE")

    if not device_info["gpu_available"]:
        print("  WARNING: No GPU detected. Profiling will run in CPU-only mode.")
        print("  GPU-specific metrics will be skipped.")
        print("  Install CUDA toolkit and a compatible GPU for full profiling.\n")

    all_results = {
        "device_info": device_info,
        "memory_profiles": [],
        "kernel_timing": [],
        "bandwidth_estimates": [],
    }

    for model_type in ["cnn", "mlp"]:
        # Memory profiling
        mem_results = profile_gpu_memory(model_type=model_type)
        all_results["memory_profiles"].extend(mem_results)

        # Kernel timing
        kernel_result = profile_kernel_timing(model_type=model_type)
        if kernel_result:
            all_results["kernel_timing"].append(kernel_result)

        # Bandwidth estimation
        bw_result = estimate_memory_bandwidth(model_type=model_type)
        if bw_result:
            all_results["bandwidth_estimates"].append(bw_result)

    # --- Save ---
    ensure_dir(save_dir)
    result_path = os.path.join(save_dir, "gpu_profiling_results.json")
    with open(result_path, "w") as f:
        json.dump(all_results, f, indent=2)

    log_section("GPU Profiling Complete")
    print(f"  Results saved to: {result_path}")

    return all_results


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_gpu_profiling()
