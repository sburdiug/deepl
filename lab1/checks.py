"""Independent PyTorch and finite-difference checks for the NumPy model."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional

from model import forward

PARAMETER_SAMPLES = (("W1", (0, 0)), ("b1", (0,)), ("W2", (0, 0)), ("b2", (0,)))


def check_with_pytorch(
    x: np.ndarray, y: np.ndarray, parameters: dict[str, np.ndarray],
    numpy_loss: float, numpy_gradients: dict[str, np.ndarray],
) -> dict[str, object]:
    """Compare the NumPy result with a separately-built torch.nn network."""
    torch.set_default_dtype(torch.float64)
    first, second = nn.Linear(4, 8), nn.Linear(8, 3)
    with torch.no_grad():
        first.weight.copy_(torch.from_numpy(parameters["W1"].T))
        first.bias.copy_(torch.from_numpy(parameters["b1"]))
        second.weight.copy_(torch.from_numpy(parameters["W2"].T))
        second.bias.copy_(torch.from_numpy(parameters["b2"]))

    logits = second(torch.relu(first(torch.from_numpy(x))))
    torch_loss = functional.cross_entropy(logits, torch.from_numpy(y))
    torch_loss.backward()
    torch_gradients = {
        "W1": first.weight.grad.detach().numpy().T,
        "b1": first.bias.grad.detach().numpy(),
        "W2": second.weight.grad.detach().numpy().T,
        "b2": second.bias.grad.detach().numpy(),
    }
    differences = {name: float(np.abs(numpy_gradients[name] - torch_gradients[name]).max())
                   for name in numpy_gradients}
    loss_difference = abs(numpy_loss - torch_loss.item())
    values = [numpy_loss, torch_loss.item(), loss_difference, *differences.values()]
    return {
        "numpy_loss": numpy_loss,
        "torch_loss": torch_loss.item(),
        "loss_difference": loss_difference,
        "gradient_differences": differences,
        "passed": bool(np.isfinite(values).all() and max(values[2:]) <= 1e-12),
    }


def check_numerical_gradients(
    x: np.ndarray, y: np.ndarray, parameters: dict[str, np.ndarray],
    manual_gradients: dict[str, np.ndarray], epsilon: float = 1e-6,
) -> dict[str, dict[str, float | bool]]:
    """Use central differences for the four parameter entries specified in the task."""
    results: dict[str, dict[str, float | bool]] = {}
    for name, index in PARAMETER_SAMPLES:
        original = parameters[name][index]
        parameters[name][index] = original + epsilon
        loss_plus, _ = forward(x, y, parameters)
        parameters[name][index] = original - epsilon
        loss_minus, _ = forward(x, y, parameters)
        parameters[name][index] = original
        numerical = (loss_plus - loss_minus) / (2.0 * epsilon)
        manual = float(manual_gradients[name][index])
        difference = abs(manual - numerical)
        results[f"{name}{index}"] = {
            "manual": manual,
            "numerical": numerical,
            "difference": difference,
            "passed": bool(np.isfinite([manual, numerical, difference]).all() and difference <= 1e-7),
        }
    return results
