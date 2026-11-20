"""The prescribed filename split and image transforms."""

import csv
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
MEAN = (.485, .456, .406)
STD = (.229, .224, .225)
EXPECTED = {
    "train": [(200, 1996761), (200, 2032625), (200, 2183923), (200, 1950345), (200, 1987613), (200, 1841817)],
    "validation": [(218, 4837323), (237, 5276962), (276, 6135652), (262, 5789535), (255, 5655068), (250, 5541845)],
    "test": [(219, 4837559), (237, 5291326), (277, 6157250), (263, 5849765), (255, 5668011), (251, 5547533)],
}


def source_folder(root: Path, part: str) -> Path:
    name = "seg_train" if part == "train" else "seg_test"
    folder = root / name
    return folder / name if (folder / name).is_dir() else folder


def create_split(root: Path, path: Path) -> None:
    rows = []
    for source in ("train", "validation"):
        rng = np.random.default_rng(0)
        for label in CLASSES:
            files = sorted(source_folder(root, source).joinpath(label).glob("*.jpg"), key=lambda p: p.name)
            if not files:
                raise ValueError(f"No images: {root}, {source}, {label}")
            indices = np.arange(len(files))
            rng.shuffle(indices)
            if source == "train":
                parts = (("train", indices[:200]),)
            else:
                cut = len(files) // 2
                parts = (("validation", indices[:cut]), ("test", indices[cut:]))
            for part, selected in parts:
                rows.extend({"file": files[i].name, "label": label, "part": part} for i in selected)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["file", "label", "part"])
        writer.writeheader()
        writer.writerows(rows)


def read_split(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or set(rows[0]) != {"file", "label", "part"}:
        raise ValueError("split.csv must contain file,label,part.")
    names = [row["file"] for row in rows]
    if len(set(names)) != len(names):
        raise ValueError("A filename occurs in more than one split row.")
    for row in rows:
        if row["label"] not in CLASSES or row["part"] not in EXPECTED:
            raise ValueError(f"Invalid split row: {row}")
        if Path(row["file"]).name != row["file"] or not row["file"].endswith(".jpg"):
            raise ValueError(f"Invalid filename: {row['file']}")
    return rows


def check_split(path: Path) -> list[dict[str, object]]:
    rows = read_split(path)
    control = []
    for part in EXPECTED:
        for label, expected in zip(CLASSES, EXPECTED[part], strict=True):
            ids = [int(Path(r["file"]).stem) for r in rows if r["part"] == part and r["label"] == label]
            actual = len(ids), sum(ids)
            if actual != expected:
                raise ValueError(f"Split mismatch {part}/{label}: {actual} != {expected}")
            control.append({"part": part, "class": label, "count": actual[0], "sum": actual[1], "passed": True})
    return control


def split_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def image_transform(training: bool) -> transforms.Compose:
    spatial = [transforms.RandomResizedCrop(160, scale=(.7, 1.0)), transforms.RandomHorizontalFlip()] if training else [transforms.Resize((160, 160))]
    return transforms.Compose([*spatial, transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


class Images(Dataset):
    def __init__(self, root: Path, split: Path, part: str):
        self.root, self.part = root, part
        self.rows = [r for r in read_split(split) if r["part"] == part]
        self.transform = image_transform(part == "train")

    def __len__(self) -> int:
        return len(self.rows)

    def path(self, index: int) -> Path:
        row = self.rows[index]
        return source_folder(self.root, self.part) / row["label"] / row["file"]

    def __getitem__(self, index: int):
        with Image.open(self.path(index)) as source:
            image = self.transform(source.convert("RGB"))
        return image, CLASSES.index(self.rows[index]["label"])


def make_loader(root: Path, split: Path, part: str) -> DataLoader:
    return DataLoader(Images(root, split, part), batch_size=32 if part == "train" else 64,
                      shuffle=part == "train", drop_last=False, num_workers=0)
