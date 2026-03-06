"""
ML Model Benchmarking - Main Runner
=====================================
Orchestrates the complete benchmarking pipeline:
    1. Train PyTorch models (CNN + MLP)
    2. Train TensorFlow models (CNN + MLP)
    3. Run Scikit-learn baselines
    4. Run inference benchmarks (CPU vs GPU)
    5. Run GPU profiling
    6. Generate visualizations

Usage:
    python main.py                    # Run everything
    python main.py --skip-training    # Skip training, only benchmark
    python main.py --quick            # Fast run with fewer epochs/subset data
"""

import argparse
import time

from src.utils import log_section, get_device_info, set_seed, get_timestamp
from src.train_pytorch import train_pytorch_model
from src.train_tensorflow import train_tensorflow_model
from src.baselines import run_all_baselines
from src.benchmark import run_inference_benchmarks
from src.gpu_profiler import run_gpu_profiling
from src.visualize import generate_all_visualizations


def main():
    parser = argparse.ArgumentParser(
        description="ML Model Benchmarking on CPU/GPU Architectures"
    )
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="Training epochs for deep learning models (default: 10)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Batch size for training and benchmarking (default: 64)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001,
        help="Learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--skip-training", action="store_true",
        help="Skip model training (use existing result files)"
    )
    parser.add_argument(
        "--skip-baselines", action="store_true",
        help="Skip sklearn baseline evaluation"
    )
    parser.add_argument(
        "--skip-benchmarks", action="store_true",
        help="Skip inference benchmarking"
    )
    parser.add_argument(
        "--skip-gpu-profiling", action="store_true",
        help="Skip GPU profiling"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: 3 epochs, 10%% data subset, fewer benchmark repeats"
    )
    parser.add_argument(
        "--results-dir", type=str, default="./results",
        help="Directory for saving all results (default: ./results)"
    )
    args = parser.parse_args()

    # Quick mode overrides
    if args.quick:
        args.epochs = 3
        subset_fraction = 0.1
        bench_repeats = 20
    else:
        subset_fraction = 1.0
        bench_repeats = 100

    # ------------------------------------------------------------------
    set_seed(42)
    start_time = time.perf_counter()

    log_section("ML MODEL BENCHMARKING SUITE")
    print(f"  Timestamp : {get_timestamp()}")
    print(f"  Mode      : {'Quick' if args.quick else 'Full'}")
    print(f"  Epochs    : {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")

    device_info = get_device_info()
    print(f"  CPU       : {device_info['cpu_name']}")
    print(f"  CPU Cores : {device_info['cpu_cores']}")
    print(f"  RAM       : {device_info['ram_gb']} GB")
    print(f"  GPU       : {device_info['gpu_name'] or 'None (CPU only)'}")

    # ------------------------------------------------------------------
    # Phase 1: Model Training
    # ------------------------------------------------------------------
    if not args.skip_training:
        log_section("PHASE 1: MODEL TRAINING")

        # PyTorch CNN
        train_pytorch_model(
            model_type="cnn", epochs=args.epochs,
            batch_size=args.batch_size, learning_rate=args.lr,
            save_dir=args.results_dir,
        )
        # PyTorch MLP
        train_pytorch_model(
            model_type="mlp", epochs=args.epochs,
            batch_size=args.batch_size, learning_rate=args.lr,
            save_dir=args.results_dir,
        )
        # TensorFlow CNN
        train_tensorflow_model(
            model_type="cnn", epochs=args.epochs,
            batch_size=args.batch_size, learning_rate=args.lr,
            save_dir=args.results_dir,
        )
        # TensorFlow MLP
        train_tensorflow_model(
            model_type="mlp", epochs=args.epochs,
            batch_size=args.batch_size, learning_rate=args.lr,
            save_dir=args.results_dir,
        )

    # ------------------------------------------------------------------
    # Phase 2: Sklearn Baselines
    # ------------------------------------------------------------------
    if not args.skip_baselines:
        log_section("PHASE 2: SKLEARN BASELINES")
        run_all_baselines(
            save_dir=args.results_dir,
            subset_fraction=subset_fraction,
        )

    # ------------------------------------------------------------------
    # Phase 3: Inference Benchmarks
    # ------------------------------------------------------------------
    if not args.skip_benchmarks:
        log_section("PHASE 3: INFERENCE BENCHMARKS")
        run_inference_benchmarks(
            save_dir=args.results_dir,
            repeats=bench_repeats,
        )

    # ------------------------------------------------------------------
    # Phase 4: GPU Profiling
    # ------------------------------------------------------------------
    if not args.skip_gpu_profiling:
        log_section("PHASE 4: GPU PROFILING")
        run_gpu_profiling(save_dir=args.results_dir)

    # ------------------------------------------------------------------
    # Phase 5: Visualization
    # ------------------------------------------------------------------
    log_section("PHASE 5: VISUALIZATION")
    generate_all_visualizations(
        results_dir=args.results_dir,
        save_dir=args.results_dir,
    )

    # ------------------------------------------------------------------
    total_time = (time.perf_counter() - start_time) / 60.0
    log_section("BENCHMARKING COMPLETE")
    print(f"  Total time: {total_time:.1f} minutes")
    print(f"  Results   : {args.results_dir}/")
    print(f"  Charts    : {args.results_dir}/*.png")


if __name__ == "__main__":
    main()
