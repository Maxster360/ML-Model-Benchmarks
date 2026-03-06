"""
Utility Functions
=================
Shared helpers for timing, device detection, logging, and reproducibility.
"""

import time
import random
import platform
import os
from datetime import datetime
from functools import wraps

import numpy as np
import torch
import psutil


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed=42):
    """Set random seeds across all frameworks for reproducible experiments.

    Args:
        seed (int): Random seed value. Default is 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Device Detection
# ---------------------------------------------------------------------------

def get_device_info():
    """Detect available compute devices and return a summary dictionary.

    Returns:
        dict: Keys include 'cpu_name', 'cpu_cores', 'ram_gb',
              'gpu_available', 'gpu_name', 'gpu_memory_gb'.
    """
    info = {
        "cpu_name": platform.processor() or "Unknown CPU",
        "cpu_cores": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": None,
        "gpu_memory_gb": None,
    }

    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_mem / (1024 ** 3), 2
        )

    return info


def get_torch_device():
    """Return the best available PyTorch device (CUDA > CPU).

    Returns:
        torch.device: 'cuda' if a GPU is available, otherwise 'cpu'.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Timing Utilities
# ---------------------------------------------------------------------------

class Timer:
    """Context-manager timer for measuring wall-clock execution time.

    Usage:
        with Timer("forward pass") as t:
            output = model(input_tensor)
        print(t.elapsed_ms)
    """

    def __init__(self, label="block"):
        self.label = label
        self.elapsed_ms = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.perf_counter() - self.start) * 1000.0


def time_function(func):
    """Decorator that prints the execution time of a function.

    Args:
        func (callable): Function to wrap.

    Returns:
        callable: Wrapped function that prints elapsed time.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000.0
        print(f"[TIMER] {func.__name__}: {elapsed:.2f} ms")
        return result
    return wrapper


def measure_latency(func, *args, warmup=5, repeats=50, **kwargs):
    """Measure average inference latency of a callable.

    Runs the function several times for warmup, then measures the average
    latency over *repeats* iterations.

    Args:
        func (callable): Function to benchmark.
        *args: Positional arguments forwarded to *func*.
        warmup (int): Number of warmup calls (ignored in timing).
        repeats (int): Number of timed calls.
        **kwargs: Keyword arguments forwarded to *func*.

    Returns:
        dict: 'mean_ms', 'std_ms', 'min_ms', 'max_ms'.
    """
    # Warmup runs (let GPU kernels compile / caches warm)
    for _ in range(warmup):
        func(*args, **kwargs)

    # Synchronize GPU before timing if CUDA is available
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        func(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000.0)

    times = np.array(times)
    return {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
    }


# ---------------------------------------------------------------------------
# Logging Helpers
# ---------------------------------------------------------------------------

def log_section(title):
    """Print a formatted section header for console output.

    Args:
        title (str): Section title text.
    """
    width = 60
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def log_metrics(metrics_dict, title="Metrics"):
    """Pretty-print a dictionary of metric name-value pairs.

    Args:
        metrics_dict (dict): Metric names mapped to numeric values.
        title (str): Header text.
    """
    log_section(title)
    for key, value in metrics_dict.items():
        if isinstance(value, float):
            print(f"  {key:<30s} : {value:.4f}")
        else:
            print(f"  {key:<30s} : {value}")


def get_timestamp():
    """Return a filesystem-safe timestamp string.

    Returns:
        str: Timestamp in 'YYYYMMDD_HHMMSS' format.
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ---------------------------------------------------------------------------
# File I/O Helpers
# ---------------------------------------------------------------------------

def ensure_dir(path):
    """Create a directory (and parents) if it does not already exist.

    Args:
        path (str): Directory path to create.
    """
    os.makedirs(path, exist_ok=True)
