"""
Scikit-learn Baseline Models
=============================
Traditional ML classifiers for CIFAR-10 to provide baseline performance
references against deep learning models (CNN, MLP).

Models:
    - Logistic Regression (linear baseline)
    - Random Forest (ensemble baseline)
    - Support Vector Machine (kernel-based baseline)

These baselines help contextualize deep learning gains and identify
whether complex architectures are justified for the task.
"""

import time
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import label_binarize

from src.data_preprocessing import load_cifar10_numpy, create_subset
from src.utils import (
    set_seed,
    Timer,
    log_section,
    log_metrics,
    ensure_dir,
)


# ---------------------------------------------------------------------------
# Baseline Model Definitions
# ---------------------------------------------------------------------------

BASELINE_MODELS = {
    "logistic_regression": {
        "name": "Logistic Regression",
        "model": lambda: LogisticRegression(
            max_iter=1000,
            solver="saga",
            multi_class="multinomial",
            n_jobs=-1,
            random_state=42,
        ),
    },
    "random_forest": {
        "name": "Random Forest",
        "model": lambda: RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            n_jobs=-1,
            random_state=42,
        ),
    },
    "svm": {
        "name": "Support Vector Machine",
        "model": lambda: SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            probability=True,  # Needed for AUC-ROC
            random_state=42,
        ),
    },
}


# ---------------------------------------------------------------------------
# Training & Evaluation
# ---------------------------------------------------------------------------

def train_and_evaluate_baseline(model_key, X_train, X_test, y_train, y_test, num_classes=10):
    """Train a single baseline model and compute evaluation metrics.

    Args:
        model_key (str): Key in BASELINE_MODELS dictionary.
        X_train (np.ndarray): Training features (N, D).
        X_test (np.ndarray): Test features (M, D).
        y_train (np.ndarray): Training labels.
        y_test (np.ndarray): Test labels.
        num_classes (int): Number of classes.

    Returns:
        dict: Model name, metrics (accuracy, F1, AUC), and timing info.
    """
    model_info = BASELINE_MODELS[model_key]
    model_name = model_info["name"]
    clf = model_info["model"]()

    log_section(f"Baseline: {model_name}")

    # --- Training ---
    print(f"  Training {model_name}...")
    with Timer("training") as train_timer:
        clf.fit(X_train, y_train)
    print(f"  Training time: {train_timer.elapsed_ms:.0f} ms")

    # --- Inference ---
    with Timer("inference") as infer_timer:
        y_pred = clf.predict(X_test)
    print(f"  Inference time: {infer_timer.elapsed_ms:.0f} ms")

    # --- Metrics ---
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    # AUC-ROC requires probability estimates
    try:
        if hasattr(clf, "predict_proba"):
            y_prob = clf.predict_proba(X_test)
        else:
            y_prob = label_binarize(y_pred, classes=range(num_classes))
        auc = roc_auc_score(y_test, y_prob, multi_class="ovr")
    except ValueError:
        auc = 0.0

    metrics = {
        "accuracy": accuracy,
        "f1_score": f1,
        "auc_roc": auc,
    }
    log_metrics(metrics, title=f"{model_name} Metrics")

    return {
        "model_name": model_name,
        "model_key": model_key,
        "metrics": metrics,
        "training_time_ms": train_timer.elapsed_ms,
        "inference_time_ms": infer_timer.elapsed_ms,
        "num_train_samples": len(X_train),
        "num_test_samples": len(X_test),
    }


# ---------------------------------------------------------------------------
# Run All Baselines
# ---------------------------------------------------------------------------

def run_all_baselines(
    data_dir="./data",
    save_dir="./results",
    subset_fraction=1.0,
):
    """Run all baseline models and save combined results.

    Args:
        data_dir (str): Dataset cache directory.
        save_dir (str): Directory for saving results JSON.
        subset_fraction (float): Fraction of data to use (for faster runs).

    Returns:
        list[dict]: Results for each baseline model.
    """
    set_seed(42)

    log_section("Loading CIFAR-10 for Sklearn Baselines")
    X_train, X_test, y_train, y_test, class_names = load_cifar10_numpy(data_dir=data_dir)

    # Optionally use a subset for faster experimentation
    if subset_fraction < 1.0:
        print(f"  Using {subset_fraction*100:.0f}% subset of training data")
        X_train, y_train = create_subset(X_train, y_train, fraction=subset_fraction)
        X_test, y_test = create_subset(X_test, y_test, fraction=subset_fraction)

    print(f"  Training samples: {len(X_train):,}")
    print(f"  Test samples    : {len(X_test):,}")
    print(f"  Feature dim     : {X_train.shape[1]}")

    all_results = []
    for model_key in BASELINE_MODELS:
        result = train_and_evaluate_baseline(
            model_key, X_train, X_test, y_train, y_test
        )
        all_results.append(result)

    # --- Save Results ---
    ensure_dir(save_dir)
    result_path = os.path.join(save_dir, "baseline_results.json")
    with open(result_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Baseline results saved to: {result_path}")

    # --- Summary Table ---
    log_section("Baseline Comparison Summary")
    print(f"  {'Model':<25s} {'Accuracy':>10s} {'F1 Score':>10s} {'AUC-ROC':>10s} {'Train (ms)':>12s}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*12}")
    for r in all_results:
        m = r["metrics"]
        print(
            f"  {r['model_name']:<25s} "
            f"{m['accuracy']:>10.4f} "
            f"{m['f1_score']:>10.4f} "
            f"{m['auc_roc']:>10.4f} "
            f"{r['training_time_ms']:>12.0f}"
        )

    return all_results


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run sklearn baselines on CIFAR-10")
    parser.add_argument(
        "--subset", type=float, default=0.1,
        help="Fraction of data to use (0.1 = 10%%, default for quick run)"
    )
    args = parser.parse_args()

    run_all_baselines(subset_fraction=args.subset)
