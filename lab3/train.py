"""Check the protocol, train exactly seven runs and select using validation."""

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import sklearn
import torch
import torchvision
from sklearn.metrics import f1_score

from checks import preflight
from data import CLASSES, create_split, image_transform, make_loader, split_hash
from model import RUNS, build_model, predict, train_epoch


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def git_state(root: Path) -> dict[str, object]:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def select_run(logs: list[dict]) -> dict:
    if [(r["regime"], r["seed"]) for r in logs] != RUNS:
        raise ValueError("Expected the seven prescribed runs in protocol order.")
    means = {regime: float(np.mean([r["history"][-1]["validation_f1"] for r in logs if r["regime"] == regime]))
             for regime in ("scratch", "frozen", "unfreeze")}
    regime = max(means, key=means.get)
    return max((r for r in logs if r["regime"] == regime), key=lambda r: r["history"][-1]["validation_f1"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    parser.add_argument("--out", type=Path, help="New directory for all experiment artifacts")
    args = parser.parse_args()
    root = args.data.resolve()
    source = Path(__file__).resolve().parent
    out = args.out.resolve() if args.out is not None else source
    if args.out is not None:
        try:
            out.mkdir(parents=True)
        except FileExistsError:
            parser.error("--out must name a new directory.")
    elif (out / "runs").exists() or (out / "lab3_model.pt").exists():
        parser.error("Results already exist. Use --out with a new directory for fresh training.")
    revision = git_state(source)
    device = args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    torch.set_num_threads(4)
    print(f"Training device: {device}", flush=True)
    split = out / "split.csv"
    if not split.exists():
        create_split(root, split)
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    checks = preflight(root, split, figures, device)
    write_json(out / "checks.json", checks)
    fingerprint = split_hash(split)
    logs = []
    for regime, seed in RUNS:
        print(f"\nRun {regime}, seed={seed}", flush=True)
        model, optimizer = build_model(regime, seed, device)
        train = make_loader(root, split, "train")
        validation = make_loader(root, split, "validation")
        history = []
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for epoch in range(8):
            loss = train_epoch(model, optimizer, train, regime, device)
            logits, y = predict(model, validation, device)
            f1 = float(f1_score(y, logits.argmax(axis=1), labels=range(6), average="macro", zero_division=0))
            history.append({"epoch": epoch + 1, "train_loss": loss, "validation_f1": f1})
            print(f"Epoch {epoch + 1}/8: train_loss={loss:.6f}, validation_f1={f1:.6f}", flush=True)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        directory = out / "runs" / f"{regime}_seed{seed}"
        directory.mkdir(parents=True)
        metadata = {
            "regime": regime, "seed": seed, "split_sha256": fingerprint, "git": revision,
            "python": platform.python_version(), "torch": str(torch.__version__),
            "torchvision": torchvision.__version__, "numpy": np.__version__, "scikit_learn": sklearn.__version__,
            "device": device, "device_name": torch.cuda.get_device_name() if device == "cuda" else platform.processor(),
            "torch_num_threads": torch.get_num_threads(), "dtype": "float32",
            "epochs": 8, "optimizer_steps": 304, "batch_size": 32, "eval_batch_size": 64,
            "shuffle": True, "eval_shuffle": False, "drop_last": False, "num_workers": 0,
            "generator": None, "sampler": "RandomSampler (default)",
            "architecture": str(model), "classes": CLASSES, "img_size": 160,
            "weights": None if regime == "scratch" else "ResNet18_Weights.IMAGENET1K_V1",
            "train_transforms": repr(image_transform(True)), "eval_transforms": repr(image_transform(False)),
            "optimizer": "AdamW", "optimizer_defaults": optimizer.defaults,
            "optimizer_groups": [{k: v for k, v in group.items() if k != "params"}
                                 | {"numel": sum(p.numel() for p in group["params"])} for group in optimizer.param_groups],
            "loss": "CrossEntropyLoss(reduction='mean', label_smoothing=0, weight=None, ignore_index=-100)",
            "batchnorm": {"momentum": .1, "eps": 1e-5, "affine": True, "track_running_stats": True},
            "scheduler": None, "gradient_clipping": None, "mixed_precision": False,
            "early_stopping": False, "checkpoint": "epoch 8 only",
            "history": history, "training_seconds": elapsed,
            "state_sha256": state_hash(model.state_dict()),
        }
        write_json(directory / "run.json", metadata)
        np.save(directory / "val_logits.npy", logits)
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, directory / "model.pt")
        logs.append(metadata)
        del model, optimizer
    chosen = select_run(logs)
    path = out / "runs" / f"{chosen['regime']}_seed{chosen['seed']}" / "model.pt"
    state = torch.load(path, map_location="cpu", weights_only=True)
    torch.save({"state_dict": state, "regime": chosen["regime"], "seed": chosen["seed"],
                "classes": CLASSES, "img_size": 160}, out / "lab3_model.pt")
    print(f"Selected on validation: {chosen['regime']}, seed={chosen['seed']}", flush=True)


if __name__ == "__main__":
    main()
