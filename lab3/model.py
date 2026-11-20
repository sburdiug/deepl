"""ResNet-18 regimes with explicit optimizer and BatchNorm freezing."""

import numpy as np
import torch
from torchvision.models import ResNet18_Weights, resnet18

RUNS = [("scratch", 1), ("frozen", 1), ("frozen", 2), ("frozen", 3),
        ("unfreeze", 1), ("unfreeze", 2), ("unfreeze", 3)]
TRAINABLE = {"scratch": 11179590, "frozen": 3078, "unfreeze": 8396806}


def build_model(regime: str, seed: int, device: str):
    if regime not in TRAINABLE:
        raise ValueError(f"Unknown regime: {regime}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    weights = None if regime == "scratch" else ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    model.fc = torch.nn.Linear(512, 6)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(regime == "scratch" or name.startswith("fc.")
                                 or (regime == "unfreeze" and name.startswith("layer4.")))
    model.to(device=device, dtype=torch.float32)
    if regime == "scratch":
        groups = [{"params": list(model.parameters()), "lr": 3e-3}]
    else:
        groups = [{"params": list(model.fc.parameters()), "lr": 3e-3}]
        if regime == "unfreeze":
            groups.append({"params": list(model.layer4.parameters()), "lr": 3e-4})
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    return model, optimizer


def set_train_mode(model: torch.nn.Module, regime: str) -> None:
    model.train()
    if regime != "scratch":
        for name, module in model.named_children():
            if name != "fc" and not (regime == "unfreeze" and name == "layer4"):
                module.eval()


def train_epoch(model, optimizer, loader, regime: str, device: str) -> float:
    set_train_mode(model, regime)
    total = 0.0
    for x, y in loader:
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x.to(device)), y.to(device))
        loss.backward()
        optimizer.step()
        total += loss.item() * y.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def predict(model, loader, device: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits, labels = [], []
    for x, y in loader:
        logits.append(model(x.to(device)).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(logits), np.concatenate(labels)
