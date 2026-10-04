"""NumPy implementation of a 4 -> 8 -> 3 classifier and its manual gradients."""

import numpy as np


def initialize_parameters() -> dict[str, np.ndarray]:
    """Create the fixed initial parameters required by the assignment."""
    rng = np.random.default_rng(0)
    return {
        "W1": rng.normal(0.0, np.sqrt(2.0 / 4), size=(4, 8)),
        "b1": np.zeros(8, dtype=np.float64),
        "W2": rng.normal(0.0, np.sqrt(2.0 / (8 + 3)), size=(8, 3)),
        "b2": np.zeros(3, dtype=np.float64),
    }


def forward(
    x: np.ndarray, y: np.ndarray, parameters: dict[str, np.ndarray]
) -> tuple[float, dict[str, np.ndarray]]:
    """Run Linear -> ReLU -> Linear and mean stable cross-entropy."""
    z1 = x @ parameters["W1"] + parameters["b1"]
    a1 = np.maximum(z1, 0.0)
    logits = a1 @ parameters["W2"] + parameters["b2"]
    shifted_logits = logits - logits.max(axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(np.exp(shifted_logits).sum(axis=1, keepdims=True))
    loss = -log_probs[np.arange(x.shape[0]), y].mean()
    return float(loss), {"x": x, "z1": z1, "a1": a1, "log_probs": log_probs}


def backward(
    cache: dict[str, np.ndarray], y: np.ndarray, parameters: dict[str, np.ndarray], *,
    omit_batch_mean: bool = False,
) -> dict[str, np.ndarray]:

    grad_logits = np.exp(cache["log_probs"])
    grad_logits[np.arange(y.size), y] -= 1.0
    if not omit_batch_mean:
        grad_logits /= y.size

    d_w2 = cache["a1"].T @ grad_logits
    d_b2 = grad_logits.sum(axis=0)
    d_a1 = grad_logits @ parameters["W2"].T
    d_z1 = d_a1 * (cache["z1"] > 0.0)
    return {
        "W1": cache["x"].T @ d_z1,
        "b1": d_z1.sum(axis=0),
        "W2": d_w2,
        "b2": d_b2,
    }
