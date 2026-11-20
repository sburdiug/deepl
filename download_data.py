"""Download the two public Kaggle datasets into the ignored data/ directory."""

from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile, is_zipfile
import shutil


def main() -> None:
    root = Path(__file__).resolve().parent / "data"
    for name, dataset, marker in (
        ("creditcard", "mlg-ulb/creditcardfraud", "creditcard.csv"),
        ("intel", "puneet6060/intel-image-classification", "seg_train/seg_train"),
    ):
        directory = root / name
        if (directory / marker).exists():
            print(f"Already downloaded: {directory}")
            continue
        directory.mkdir(parents=True, exist_ok=True)
        archive = root / f"{name}.zip"
        if not archive.exists():
            temporary = archive.with_suffix(".part")
            print(f"Downloading {dataset}", flush=True)
            with urlopen(f"https://www.kaggle.com/api/v1/datasets/download/{dataset}", timeout=120) as source:
                with temporary.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
            if not is_zipfile(temporary):
                raise ValueError(f"Kaggle did not return a ZIP archive: {temporary}")
            temporary.replace(archive)
        with ZipFile(archive) as source:
            for entry in source.infolist():
                target = (directory / entry.filename).resolve()
                if not target.is_relative_to(directory.resolve()):
                    raise ValueError(f"Unsafe archive entry: {entry.filename}")
                if name != "intel" or entry.filename.startswith(("seg_train/", "seg_test/")):
                    source.extract(entry, directory)
        if not (directory / marker).exists():
            raise ValueError(f"Unexpected dataset layout in {directory}")
        print(f"Ready: {directory}", flush=True)


if __name__ == "__main__":
    main()
