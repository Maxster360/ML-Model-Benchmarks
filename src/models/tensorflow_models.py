"""
TensorFlow / Keras Model Definitions
=====================================
CNN and MLP architectures for CIFAR-10 image classification benchmarking.
Mirror the PyTorch model designs for fair cross-framework comparison.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models


# ---------------------------------------------------------------------------
# CNN Model
# ---------------------------------------------------------------------------

def build_tf_cnn(num_classes=10):
    """Build a TensorFlow/Keras CNN matching the PyTorch CNN architecture.

    Architecture:
        - 3 convolutional blocks (Conv2D -> BatchNorm -> ReLU -> MaxPool)
        - 2 dense layers with dropout
        - Input: (32, 32, 3), Output: (num_classes,)

    Args:
        num_classes (int): Number of output classes.

    Returns:
        keras.Model: Compiled Keras model.
    """
    model = models.Sequential([
        # Block 1: 3 -> 32 filters
        layers.Conv2D(32, (3, 3), padding="same", input_shape=(32, 32, 3)),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D((2, 2)),

        # Block 2: 32 -> 64 filters
        layers.Conv2D(64, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D((2, 2)),

        # Block 3: 64 -> 128 filters
        layers.Conv2D(128, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D((2, 2)),

        # Classifier head
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes),
    ])

    return model


# ---------------------------------------------------------------------------
# MLP Model
# ---------------------------------------------------------------------------

def build_tf_mlp(input_dim=3072, num_classes=10):
    """Build a TensorFlow/Keras MLP matching the PyTorch MLP architecture.

    Architecture:
        - 3 hidden layers with BatchNorm, ReLU, Dropout
        - Input: flattened (3072,) or image (32, 32, 3)
        - Output: (num_classes,)

    Args:
        input_dim (int): Flattened input dimension (default 3*32*32).
        num_classes (int): Number of output classes.

    Returns:
        keras.Model: Compiled Keras model.
    """
    model = models.Sequential([
        # Flatten images if they come as (32, 32, 3)
        layers.Flatten(input_shape=(32, 32, 3)),

        # Layer 1: 3072 -> 512
        layers.Dense(512),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Dropout(0.4),

        # Layer 2: 512 -> 256
        layers.Dense(256),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Dropout(0.3),

        # Layer 3: 256 -> 128
        layers.Dense(128),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Dropout(0.2),

        # Output layer
        layers.Dense(num_classes),
    ])

    return model


# ---------------------------------------------------------------------------
# Model Factory
# ---------------------------------------------------------------------------

def build_tensorflow_model(model_type="cnn", num_classes=10):
    """Instantiate and return a TensorFlow model.

    Args:
        model_type (str): 'cnn' or 'mlp'.
        num_classes (int): Number of output classes.

    Returns:
        keras.Model: The constructed (uncompiled) Keras model.

    Raises:
        ValueError: If model_type is not 'cnn' or 'mlp'.
    """
    builders = {
        "cnn": build_tf_cnn,
        "mlp": build_tf_mlp,
    }

    if model_type not in builders:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Choose from {list(builders.keys())}"
        )

    return builders[model_type](num_classes=num_classes)


def count_tf_parameters(model):
    """Count total trainable parameters in a Keras model.

    Args:
        model (keras.Model): Keras model (must be built).

    Returns:
        int: Number of trainable parameters.
    """
    return sum(
        tf.reduce_prod(var.shape).numpy()
        for var in model.trainable_variables
    )
