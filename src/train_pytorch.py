"""
PyTorch Training Pipeline
==========================
Training and evaluation loops for CNN and MLP models on CIFAR-10.
Supports both CPU and GPU execution with timing instrumentation.
"""

import time
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.models.pytorch_models import build_pytorch_model, count_parameters
from src.data_preprocessing import load_cifar10_pytorch
from src.utils import (
    set_seed,
    get_device_info,
    get_torch_device,
    Timer,
    log_section,
    log_metrics,
    ensure_dir,
    get_timestamp,
)


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train the model for one epoch and return average loss + accuracy.

    Args:
        model (nn.Module): PyTorch model.
        train_loader (DataLoader): Training data loader.
        criterion: Loss function.
        optimizer: Optimizer instance.
        device (torch.device): Compute device.

    Returns:
        dict: 'loss' and 'accuracy' for the epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return {"loss": epoch_loss, "accuracy": epoch_acc}


# ---------------------------------------------------------------------------
# Evaluation Loop
# ---------------------------------------------------------------------------

def evaluate_model(model, test_loader, criterion, device, num_classes=10):
    """Evaluate the model on the test set with comprehensive metrics.

    Args:
        model (nn.Module): Trained PyTorch model.
        test_loader (DataLoader): Test data loader.
        criterion: Loss function.
        device (torch.device): Compute device.
        num_classes (int): Number of classes for AUC computation.

    Returns:
        dict: 'loss', 'accuracy', 'f1_score', 'auc_roc' metrics.
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    all_targets = np.array(all_targets)
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)

    total = len(all_targets)
    test_loss = running_loss / total
    accuracy = accuracy_score(all_targets, all_predictions)
    f1 = f1_score(all_targets, all_predictions, average="weighted")

    # AUC-ROC (one-vs-rest for multiclass)
    try:
        auc = roc_auc_score(all_targets, all_probabilities, multi_class="ovr")
    except ValueError:
        auc = 0.0  # Fallback if evaluation fails on small subsets

    return {
        "loss": test_loss,
        "accuracy": accuracy,
        "f1_score": f1,
        "auc_roc": auc,
    }


# ---------------------------------------------------------------------------
# Full Training Pipeline
# ---------------------------------------------------------------------------

def train_pytorch_model(
    model_type="cnn",
    epochs=10,
    batch_size=64,
    learning_rate=0.001,
    data_dir="./data",
    save_dir="./results",
    subset_fraction=1.0,
):
    """End-to-end PyTorch training pipeline with profiling.

    Trains a CNN or MLP on CIFAR-10, logs per-epoch metrics, measures
    wall-clock training time, and saves results to JSON.

    Args:
        model_type (str): 'cnn' or 'mlp'.
        epochs (int): Number of training epochs.
        batch_size (int): Mini-batch size.
        learning_rate (float): Initial learning rate for Adam optimizer.
        data_dir (str): Dataset cache directory.
        save_dir (str): Directory for saving results JSON.

    Returns:
        dict: Complete training results including per-epoch history,
              final test metrics, and timing information.
    """
    set_seed(42)
    device = get_torch_device()
    device_info = get_device_info()

    log_section(f"PyTorch {model_type.upper()} Training")
    print(f"  Device       : {device}")
    print(f"  Epochs       : {epochs}")
    print(f"  Batch Size   : {batch_size}")
    print(f"  Learning Rate: {learning_rate}")

    # --- Data ---
    train_loader, test_loader, class_names = load_cifar10_pytorch(
        batch_size=batch_size, data_dir=data_dir, subset_fraction=subset_fraction
    )

    # --- Model ---
    model = build_pytorch_model(model_type=model_type, device=device)
    num_params = count_parameters(model)
    print(f"  Parameters   : {num_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # --- Training ---
    history = {"train_loss": [], "train_accuracy": []}
    total_train_time = 0.0

    for epoch in range(1, epochs + 1):
        with Timer(f"Epoch {epoch}") as t:
            metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()

        total_train_time += t.elapsed_ms
        history["train_loss"].append(metrics["loss"])
        history["train_accuracy"].append(metrics["accuracy"])

        print(
            f"  Epoch [{epoch:>2d}/{epochs}] "
            f"Loss: {metrics['loss']:.4f}  "
            f"Acc: {metrics['accuracy']:.4f}  "
            f"Time: {t.elapsed_ms:.0f} ms"
        )

    # --- Evaluation ---
    log_section(f"PyTorch {model_type.upper()} Evaluation")
    with Timer("evaluation") as eval_timer:
        test_metrics = evaluate_model(model, test_loader, criterion, device)

    log_metrics(test_metrics, title="Test Metrics")

    # --- Save Results ---
    results = {
        "framework": "pytorch",
        "model_type": model_type,
        "device": str(device),
        "device_info": device_info,
        "num_parameters": num_params,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "training_time_ms": total_train_time,
        "eval_time_ms": eval_timer.elapsed_ms,
        "history": history,
        "test_metrics": test_metrics,
    }

    ensure_dir(save_dir)
    result_path = os.path.join(save_dir, f"pytorch_{model_type}_results.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {result_path}")

    return results


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train PyTorch models on CIFAR-10")
    parser.add_argument("--model", choices=["cnn", "mlp"], default="cnn", help="Model type")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    args = parser.parse_args()

    train_pytorch_model(
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
