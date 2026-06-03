"""
TensorFlow Training Pipeline
==============================
Training and evaluation loops for CNN and MLP models on CIFAR-10.
Mirrors the PyTorch pipeline for consistent cross-framework benchmarking.
"""

import time
import json
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.models.tensorflow_models import build_tensorflow_model, count_tf_parameters
from src.data_preprocessing import load_cifar10_tensorflow
from src.utils import (
    set_seed,
    get_device_info,
    Timer,
    log_section,
    log_metrics,
    ensure_dir,
    get_timestamp,
)


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train_one_epoch_tf(model, train_ds, loss_fn, optimizer):
    """Train the TensorFlow model for one epoch.

    Args:
        model (keras.Model): TensorFlow model.
        train_ds (tf.data.Dataset): Training dataset.
        loss_fn: Loss function.
        optimizer: Keras optimizer.

    Returns:
        dict: 'loss' and 'accuracy' for the epoch.
    """
    epoch_loss = []
    epoch_correct = 0
    epoch_total = 0

    for images, labels in train_ds:
        with tf.GradientTape() as tape:
            logits = model(images, training=True)
            loss = loss_fn(labels, logits)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        epoch_loss.append(loss.numpy())
        predictions = tf.argmax(logits, axis=1).numpy()
        epoch_correct += np.sum(predictions == labels.numpy())
        epoch_total += len(labels)

    avg_loss = float(np.mean(epoch_loss))
    accuracy = epoch_correct / epoch_total
    return {"loss": avg_loss, "accuracy": accuracy}


# ---------------------------------------------------------------------------
# Evaluation Loop
# ---------------------------------------------------------------------------

def evaluate_model_tf(model, test_ds, loss_fn, num_classes=10):
    """Evaluate a TensorFlow model with comprehensive metrics.

    Args:
        model (keras.Model): Trained TensorFlow model.
        test_ds (tf.data.Dataset): Test dataset.
        loss_fn: Loss function.
        num_classes (int): Number of classes for AUC computation.

    Returns:
        dict: 'loss', 'accuracy', 'f1_score', 'auc_roc' metrics.
    """
    all_losses = []
    all_targets = []
    all_predictions = []
    all_probabilities = []

    for images, labels in test_ds:
        logits = model(images, training=False)
        loss = loss_fn(labels, logits)

        all_losses.append(loss.numpy())
        probabilities = tf.nn.softmax(logits, axis=1).numpy()
        predictions = tf.argmax(logits, axis=1).numpy()

        all_targets.extend(labels.numpy())
        all_predictions.extend(predictions)
        all_probabilities.extend(probabilities)

    all_targets = np.array(all_targets)
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)

    avg_loss = float(np.mean(all_losses))
    accuracy = accuracy_score(all_targets, all_predictions)
    f1 = f1_score(all_targets, all_predictions, average="weighted")

    try:
        auc = roc_auc_score(all_targets, all_probabilities, multi_class="ovr")
    except ValueError:
        auc = 0.0

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "f1_score": f1,
        "auc_roc": auc,
    }


# ---------------------------------------------------------------------------
# Full Training Pipeline
# ---------------------------------------------------------------------------

def train_tensorflow_model(
    model_type="cnn",
    epochs=10,
    batch_size=64,
    learning_rate=0.001,
    save_dir="./results",
    subset_fraction=1.0,
):
    """End-to-end TensorFlow training pipeline with profiling.

    Trains a CNN or MLP on CIFAR-10, logs per-epoch metrics, measures
    wall-clock training time, and saves results to JSON.

    Args:
        model_type (str): 'cnn' or 'mlp'.
        epochs (int): Number of training epochs.
        batch_size (int): Mini-batch size.
        learning_rate (float): Initial learning rate for Adam optimizer.
        save_dir (str): Directory for saving results JSON.

    Returns:
        dict: Complete training results including per-epoch history,
              final test metrics, and timing information.
    """
    set_seed(42)
    device_info = get_device_info()

    # Check TensorFlow GPU availability
    gpus = tf.config.list_physical_devices("GPU")
    tf_device = "GPU" if gpus else "CPU"

    log_section(f"TensorFlow {model_type.upper()} Training")
    print(f"  Device       : {tf_device}")
    print(f"  TF GPUs      : {len(gpus)}")
    print(f"  Epochs       : {epochs}")
    print(f"  Batch Size   : {batch_size}")
    print(f"  Learning Rate: {learning_rate}")

    # --- Data ---
    train_ds, test_ds, class_names = load_cifar10_tensorflow(
        batch_size=batch_size, subset_fraction=subset_fraction
    )

    # --- Model ---
    model = build_tensorflow_model(model_type=model_type)

    # Build the model by running a dummy forward pass
    dummy_input = tf.zeros((1, 32, 32, 3))
    _ = model(dummy_input)

    num_params = count_tf_parameters(model)
    print(f"  Parameters   : {num_params:,}")

    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    # --- Training ---
    history = {"train_loss": [], "train_accuracy": []}
    total_train_time = 0.0

    for epoch in range(1, epochs + 1):
        with Timer(f"Epoch {epoch}") as t:
            metrics = train_one_epoch_tf(model, train_ds, loss_fn, optimizer)

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
    log_section(f"TensorFlow {model_type.upper()} Evaluation")
    with Timer("evaluation") as eval_timer:
        test_metrics = evaluate_model_tf(model, test_ds, loss_fn)

    log_metrics(test_metrics, title="Test Metrics")

    # --- Save Results ---
    results = {
        "framework": "tensorflow",
        "model_type": model_type,
        "device": tf_device,
        "device_info": device_info,
        "num_parameters": int(num_params),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "training_time_ms": total_train_time,
        "eval_time_ms": eval_timer.elapsed_ms,
        "history": history,
        "test_metrics": test_metrics,
    }

    ensure_dir(save_dir)
    result_path = os.path.join(save_dir, f"tensorflow_{model_type}_results.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {result_path}")

    return results


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train TensorFlow models on CIFAR-10")
    parser.add_argument("--model", choices=["cnn", "mlp"], default="cnn", help="Model type")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    args = parser.parse_args()

    train_tensorflow_model(
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
