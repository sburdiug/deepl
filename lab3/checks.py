"""Preflight checks and the deliberate frozen-BatchNorm mistake."""

from itertools import islice
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data import CLASSES, MEAN, STD, check_split, make_loader
from model import TRAINABLE, build_model, set_train_mode


def parameter_check(model, optimizer, regime: str) -> dict[str, object]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimized = [p for group in optimizer.param_groups for p in group["params"]]
    assert len({id(p) for p in optimized}) == len(optimized)
    assert {id(p) for p in optimized} == {id(p) for p in model.parameters() if p.requires_grad}
    assert all(p.requires_grad for p in optimized)
    assert total == 11179590 and trainable == TRAINABLE[regime]
    groups = [{"parameters": sum(p.numel() for p in g["params"]), "lr": g["lr"],
               "weight_decay": g["weight_decay"]} for g in optimizer.param_groups]
    expected = {"scratch": [(11179590, .003)], "frozen": [(3078, .003)],
                "unfreeze": [(3078, .003), (8393728, .0003)]}
    assert [(g["parameters"], g["lr"]) for g in groups] == expected[regime]
    assert all(g["weight_decay"] == 1e-4 for g in groups)
    return {"regime": regime, "total": total, "trainable": trainable, "groups": groups}


def changed_modules(model, before: dict[str, torch.Tensor]) -> list[dict[str, object]]:
    rows = []
    for name, module in model.named_children():
        parameters = list(module.named_parameters())
        buffers = [(n, b) for n, b in module.named_buffers()
                   if n.endswith(("running_mean", "running_var", "num_batches_tracked"))]
        if not parameters and not buffers:
            continue
        changed_p = any(not torch.equal(before[f"{name}.{n}"], p.detach().cpu()) for n, p in parameters)
        changed_bn = any(not torch.equal(before[f"{name}.{n}"], b.detach().cpu()) for n, b in buffers)
        rows.append({"module": name, "parameters_changed": changed_p,
                     "bn_changed": changed_bn if buffers else None})
    return rows


def mode_check(root: Path, split: Path, regime: str, device: str, buggy: bool = False) -> dict[str, object]:
    model, optimizer = build_model(regime, 1, device)
    parameter_check(model, optimizer, regime)
    before = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    if buggy:
        model.train()
    else:
        set_train_mode(model, regime)
    losses = []
    for x, y in islice(make_loader(root, split, "train"), 5):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x.to(device)), y.to(device))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert len(losses) == 5
    rows = changed_modules(model, before)
    for row in rows:
        expected_p = row["module"] == "fc" or (regime == "unfreeze" and row["module"] == "layer4")
        expected_bn = (buggy or expected_p) if row["bn_changed"] is not None else None
        row["passed"] = row["parameters_changed"] == expected_p and row["bn_changed"] == expected_bn
        assert row["passed"], (regime, row)
    changes = {name: float((model.state_dict()[f"layer1.0.bn1.{name}"].cpu()
                            - before[f"layer1.0.bn1.{name}"]).abs().max())
               for name in ("running_mean", "running_var")}
    return {"regime": regime, "buggy": buggy, "modules": rows, "losses": losses,
            "layer1.0.bn1_max_changes": changes}


def preflight(root: Path, split: Path, out: Path, device: str) -> dict[str, object]:
    result = {"split": check_split(split), "parameters": [], "initial_loss": {}, "modes": []}
    for regime in TRAINABLE:
        model, optimizer = build_model(regime, 1, device)
        result["parameters"].append(parameter_check(model, optimizer, regime))
        if regime in ("scratch", "frozen"):
            set_train_mode(model, regime)
            x, y = next(iter(make_loader(root, split, "train")))
            with torch.no_grad():
                loss = torch.nn.functional.cross_entropy(model(x.to(device)), y.to(device)).item()
            if not np.isfinite(loss):
                raise ValueError("Initial loss is not finite.")
            result["initial_loss"][regime] = loss
            if regime == "scratch":
                images = (x[:16] * torch.tensor(STD)[:, None, None] + torch.tensor(MEAN)[:, None, None]).clamp(0, 1)
                fig, axes = plt.subplots(4, 4, figsize=(9, 9))
                for ax, image, label in zip(axes.flat, images, y[:16], strict=True):
                    ax.imshow(image.permute(1, 2, 0).numpy())
                    ax.set_title(CLASSES[int(label)])
                    ax.axis("off")
                fig.tight_layout()
                fig.savefig(out / "batch.png", dpi=130)
                plt.close(fig)
        del model, optimizer
    for regime in ("frozen", "unfreeze"):
        result["modes"].append(mode_check(root, split, regime, device))
        print(f"Mode check {regime}: passed", flush=True)
    result["bn_prediction"] = (
        "При звичайному model.train() параметри замороженого кістяка не зміняться, "
        "але running_mean/running_var і num_batches_tracked BN оновляться. "
        "Train-loss може спадати, тому сама його крива не виявляє цю помилку."
    )
    print(result["bn_prediction"], flush=True)
    result["bn_bug"] = mode_check(root, split, "frozen", device, buggy=True)
    print("Deliberate BatchNorm experiment: matched prediction", flush=True)
    return result
