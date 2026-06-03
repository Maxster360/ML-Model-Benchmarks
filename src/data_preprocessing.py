"""
Data Preprocessing Pipeline
============================
Handles loading, preprocessing, and feature engineering for benchmark datasets.
Uses CIFAR-10 as the primary benchmark dataset (standard for CNN/MLP comparison).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

import torch
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms


# ---------------------------------------------------------------------------
# CIFAR-10 Loaders (primary benchmark dataset)
# ---------------------------------------------------------------------------

def load_cifar10_pytorch(batch_size=64, data_dir="./data", subset_fraction=1.0):
    """Download and prepare CIFAR-10 as PyTorch DataLoaders.

    Applies standard normalization (mean/std per channel) suitable for
    CNN training.  Returns separate loaders for training and testing.

    Args:
        batch_size (int): Mini-batch size.
        data_dir (str): Path where the dataset will be cached.
        subset_fraction (float): Fraction of each split to keep (for quick
            runs).  Values < 1.0 take a random reproducible subset.

    Returns:
        tuple: (train_loader, test_loader, class_names)
    """
    # Standard CIFAR-10 normalization values (precomputed over training set)
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616],
    )

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        normalize,
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    class_names = train_dataset.classes  # capture before any Subset wrapping

    # Optionally take a smaller random subset for quick experiments
    if subset_fraction < 1.0:
        from torch.utils.data import Subset
        rng = np.random.default_rng(42)
        for ds_name, ds in (("train", train_dataset), ("test", test_dataset)):
            n_keep = max(batch_size + 1, int(len(ds) * subset_fraction))
            indices = rng.choice(len(ds), size=n_keep, replace=False)
            if ds_name == "train":
                train_dataset = Subset(train_dataset, indices.tolist())
            else:
                test_dataset = Subset(test_dataset, indices.tolist())

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return train_loader, test_loader, class_names


def load_cifar10_numpy(data_dir="./data"):
    """Load CIFAR-10 as raw NumPy arrays for sklearn baselines.

    Images are flattened from (3, 32, 32) to 3072-d vectors and
    standardized with zero-mean / unit-variance scaling.

    Args:
        data_dir (str): Path where the dataset will be cached.

    Returns:
        tuple: (X_train, X_test, y_train, y_test, class_names)
    """
    # Re-use torchvision downloader, then extract numpy arrays
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True
    )

    X_train = np.array(train_dataset.data, dtype=np.float32)  # (50000, 32, 32, 3)
    y_train = np.array(train_dataset.targets)
    X_test = np.array(test_dataset.data, dtype=np.float32)
    y_test = np.array(test_dataset.targets)

    # Flatten for traditional ML models: (N, 32, 32, 3) -> (N, 3072)
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    # Standardize pixel values
    scaler = StandardScaler()
    X_train_flat = scaler.fit_transform(X_train_flat)
    X_test_flat = scaler.transform(X_test_flat)

    class_names = train_dataset.classes
    return X_train_flat, X_test_flat, y_train, y_test, class_names


# ---------------------------------------------------------------------------
# Feature Engineering Utilities
# ---------------------------------------------------------------------------

def compute_image_statistics(images_np):
    """Compute per-channel statistics for a batch of images.

    Useful for exploratory data analysis before model training.

    Args:
        images_np (np.ndarray): Images of shape (N, H, W, C) or (N, C, H, W).

    Returns:
        pd.DataFrame: Per-channel mean, std, min, max statistics.
    """
    # Ensure channel-last format
    if images_np.ndim == 4 and images_np.shape[1] in (1, 3):
        images_np = np.transpose(images_np, (0, 2, 3, 1))

    n_channels = images_np.shape[-1]
    channel_names = ["Red", "Green", "Blue"][:n_channels]

    stats = []
    for ch_idx, ch_name in enumerate(channel_names):
        channel_data = images_np[..., ch_idx]
        stats.append({
            "channel": ch_name,
            "mean": float(np.mean(channel_data)),
            "std": float(np.std(channel_data)),
            "min": float(np.min(channel_data)),
            "max": float(np.max(channel_data)),
        })

    return pd.DataFrame(stats)


def compute_class_distribution(labels):
    """Count samples per class and return as a DataFrame.

    Args:
        labels (np.ndarray): 1-D array of integer class labels.

    Returns:
        pd.DataFrame: Columns 'class_id', 'count', 'percentage'.
    """
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    rows = [
        {"class_id": int(c), "count": int(cnt), "percentage": round(cnt / total * 100, 2)}
        for c, cnt in zip(unique, counts)
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# TensorFlow Data Helpers
# ---------------------------------------------------------------------------

def load_cifar10_tensorflow(batch_size=64, subset_fraction=1.0):
    """Load CIFAR-10 through TensorFlow / Keras utilities.

    Returns tf.data.Dataset pipelines with standard preprocessing
    (float32 cast, normalization, batching, prefetching).

    Args:
        batch_size (int): Mini-batch size.
        subset_fraction (float): Fraction of each split to keep (for quick
            runs).  Values < 1.0 take a random reproducible subset.

    Returns:
        tuple: (train_ds, test_ds, class_names)
    """
    # Deferred import so TensorFlow is only loaded when needed
    import tensorflow as tf

    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Normalize pixel values to [0, 1]
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    # Optionally take a smaller random subset for quick experiments
    if subset_fraction < 1.0:
        rng = np.random.default_rng(42)
        n_train = max(batch_size + 1, int(len(X_train) * subset_fraction))
        n_test = max(batch_size + 1, int(len(X_test) * subset_fraction))
        train_idx = rng.choice(len(X_train), size=n_train, replace=False)
        test_idx = rng.choice(len(X_test), size=n_test, replace=False)
        X_train, y_train = X_train[train_idx], y_train[train_idx]
        X_test, y_test = X_test[test_idx], y_test[test_idx]

    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .shuffle(10000)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_ds = (
        tf.data.Dataset.from_tensor_slices((X_test, y_test))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    class_names = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ]
    return train_ds, test_ds, class_names


# ---------------------------------------------------------------------------
# Subset Sampler (for quick experiments)
# ---------------------------------------------------------------------------

def create_subset(X, y, fraction=0.1, seed=42):
    """Return a stratified random subset of the data.

    Useful for fast prototyping and debugging before full-scale runs.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Label vector.
        fraction (float): Fraction of data to keep (0 < fraction <= 1).
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: (X_subset, y_subset)
    """
    if fraction >= 1.0:
        return X, y

    _, X_subset, _, y_subset = train_test_split(
        X, y, test_size=fraction, stratify=y, random_state=seed
    )
    return X_subset, y_subset
