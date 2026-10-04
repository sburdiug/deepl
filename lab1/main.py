"""Run all required backpropagation checks with the fixed Iris setup."""

import numpy as np
from sklearn.datasets import load_iris

from checks import check_numerical_gradients, check_with_pytorch
from model import backward, forward, initialize_parameters


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    iris = load_iris()
    rng = np.random.default_rng(0)
    train_indices: list[int] = []
    test_indices: list[int] = []
    for label in range(3):
        indices = np.flatnonzero(iris.target == label)
        rng.shuffle(indices)
        train_indices.extend(indices[:35])
        test_indices.extend(indices[35:])
    x_train, x_test = iris.data[train_indices], iris.data[test_indices]
    mean, std = x_train.mean(axis=0), x_train.std(axis=0, ddof=0)
    return (x_train - mean) / std, iris.target[train_indices], (x_test - mean) / std, iris.target[test_indices]


def print_pytorch_check(title: str, result: dict[str, object]) -> None:
    print(f"\n{title}")
    print(f"loss NumPy: {result['numpy_loss']:.16f}")
    print(f"loss PyTorch: {result['torch_loss']:.16f}")
    print(f"loss max abs difference: {result['loss_difference']:.3e}")
    for name, difference in result["gradient_differences"].items():
        print(f"{name}: {difference:.3e}")
    print(f"passed (<= 1e-12): {result['passed']}")


def print_numerical_check(title: str, result: dict[str, dict[str, float | bool]]) -> None:
    print(f"\n{title}")
    for name, values in result.items():
        print(f"{name}: manual={values['manual']:.10f}, numerical={values['numerical']:.10f}, "
              f"difference={values['difference']:.3e}, passed={values['passed']}")


def main() -> None:
    x_train, y_train, _, _ = load_data()
    parameters = initialize_parameters()
    loss, cache = forward(x_train, y_train, parameters)
    gradients = backward(cache, y_train, parameters)
    print_pytorch_check("Correct implementation: PyTorch check", check_with_pytorch(x_train, y_train, parameters, loss, gradients))
    print_numerical_check("Correct implementation: numerical check", check_numerical_gradients(x_train, y_train, parameters, gradients))

    # Deliberate experiment: loss remains the same, while every non-zero gradient is N times larger.
    buggy_gradients = backward(cache, y_train, parameters, omit_batch_mean=True)
    print_pytorch_check("Deliberate bug: PyTorch check", check_with_pytorch(x_train, y_train, parameters, loss, buggy_gradients))
    print_numerical_check("Deliberate bug: numerical check", check_numerical_gradients(x_train, y_train, parameters, buggy_gradients))


if __name__ == "__main__":
    main()
