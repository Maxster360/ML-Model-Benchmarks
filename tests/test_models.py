"""
Unit Tests for Model Architectures
====================================
Validates model construction, output shapes, and parameter counts
for both PyTorch and TensorFlow models.
"""

import sys
import os
import unittest

import numpy as np
import torch

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.pytorch_models import CNN, MLP, build_pytorch_model, count_parameters
from src.utils import Timer, set_seed, get_device_info


class TestPyTorchCNN(unittest.TestCase):
    """Tests for the PyTorch CNN model."""

    def setUp(self):
        set_seed(42)
        self.model = CNN(num_classes=10)
        self.model.eval()

    def test_output_shape(self):
        """CNN output should be (batch_size, 10) for CIFAR-10."""
        x = torch.randn(8, 3, 32, 32)
        with torch.no_grad():
            output = self.model(x)
        self.assertEqual(output.shape, (8, 10))

    def test_single_sample(self):
        """CNN should handle batch_size=1."""
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            output = self.model(x)
        self.assertEqual(output.shape, (1, 10))

    def test_parameter_count(self):
        """CNN should have a reasonable number of parameters."""
        num_params = count_parameters(self.model)
        self.assertGreater(num_params, 10000, "CNN should have >10K parameters")
        self.assertLess(num_params, 5000000, "CNN should have <5M parameters")

    def test_output_not_nan(self):
        """CNN outputs should not contain NaN values."""
        x = torch.randn(4, 3, 32, 32)
        with torch.no_grad():
            output = self.model(x)
        self.assertFalse(torch.isnan(output).any(), "Output contains NaN values")


class TestPyTorchMLP(unittest.TestCase):
    """Tests for the PyTorch MLP model."""

    def setUp(self):
        set_seed(42)
        self.model = MLP(num_classes=10)
        self.model.eval()

    def test_output_shape_from_images(self):
        """MLP should accept (B, 3, 32, 32) and output (B, 10)."""
        x = torch.randn(8, 3, 32, 32)
        with torch.no_grad():
            output = self.model(x)
        self.assertEqual(output.shape, (8, 10))

    def test_output_shape_from_flat(self):
        """MLP should accept (B, 3072) pre-flattened input."""
        x = torch.randn(8, 3072)
        with torch.no_grad():
            output = self.model(x)
        self.assertEqual(output.shape, (8, 10))

    def test_parameter_count(self):
        """MLP should have a reasonable number of parameters."""
        num_params = count_parameters(self.model)
        self.assertGreater(num_params, 100000, "MLP should have >100K parameters")
        self.assertLess(num_params, 10000000, "MLP should have <10M parameters")


class TestModelFactory(unittest.TestCase):
    """Tests for the build_pytorch_model factory function."""

    def test_build_cnn(self):
        model = build_pytorch_model(model_type="cnn", device=torch.device("cpu"))
        self.assertIsInstance(model, CNN)

    def test_build_mlp(self):
        model = build_pytorch_model(model_type="mlp", device=torch.device("cpu"))
        self.assertIsInstance(model, MLP)

    def test_invalid_model_type(self):
        with self.assertRaises(ValueError):
            build_pytorch_model(model_type="invalid")


class TestUtils(unittest.TestCase):
    """Tests for utility functions."""

    def test_timer_context_manager(self):
        """Timer should measure elapsed time in milliseconds."""
        with Timer("test") as t:
            _ = sum(range(10000))
        self.assertGreater(t.elapsed_ms, 0.0)

    def test_device_info_structure(self):
        """get_device_info should return expected keys."""
        info = get_device_info()
        expected_keys = ["cpu_name", "cpu_cores", "ram_gb", "gpu_available"]
        for key in expected_keys:
            self.assertIn(key, info)

    def test_set_seed_reproducibility(self):
        """Setting the same seed should produce identical random values."""
        set_seed(123)
        a = torch.randn(5)
        set_seed(123)
        b = torch.randn(5)
        self.assertTrue(torch.equal(a, b))


if __name__ == "__main__":
    unittest.main()
