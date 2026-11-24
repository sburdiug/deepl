"""Compare the Lab 3 checkpoint as ONNX FP32 and static INT8 on the CPU."""

import argparse
from collections import Counter
import hashlib
import json
import platform
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process
from PIL import Image
from sklearn.metrics import f1_score
import torch
from torchvision import transforms
from torchvision.models import resnet18

from lab3.data import CLASSES, MEAN, STD, read_split, source_folder


def load_model(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["classes"] != CLASSES or checkpoint["img_size"] != 160:
        raise ValueError("The checkpoint must use the Lab 3 class order and 160×160 inputs.")
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(512, len(CLASSES))
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval(), checkpoint


def export_fp32(model, size, output):
    started = perf_counter()
    with torch.no_grad():
        torch.onnx.export(
            model, (torch.zeros(1, 3, size, size, dtype=torch.float32),), str(output),
            dynamo=False, opset_version=17, input_names=["input"], output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        )
    return perf_counter() - started


def graph_checks(path):
    model = onnx.load(path)
    onnx.checker.check_model(model)
    ops = Counter(node.op_type for node in model.graph.node)
    if (ops["Conv"], ops["BatchNormalization"]) != (20, 0):
        raise ValueError(f"Unexpected exported operators: {ops}")
    initializers = {value.name: value for value in model.graph.initializer}
    ranges = []
    for node in model.graph.node:
        if node.op_type == "Conv":
            weights = onnx.numpy_helper.to_array(initializers[node.input[1]])
            alpha = np.abs(weights).max(axis=(1, 2, 3))
            ratio = float(alpha.max() / np.median(alpha))
            if not np.isfinite(ratio):
                raise ValueError(f"Invalid channel range ratio in {node.name}")
            ranges.append({"name": node.name, "ratio": ratio})
    median = float(np.median([entry["ratio"] for entry in ranges]))
    return {"conv": ops["Conv"], "batch_normalization": ops["BatchNormalization"],
            "ranges": sorted(ranges, key=lambda entry: entry["ratio"], reverse=True),
            "median_ratio": median, "median_bit_loss": float(np.log2(median))}


def split_rows(path):
    rows = read_split(path)
    train = sorted((row for row in rows if row["part"] == "train"), key=lambda row: row["file"])
    test = [row for row in rows if row["part"] == "test"]
    control = {part: {"count": len(group), "sum": sum(int(Path(row["file"]).stem) for row in group)}
               for part, group in (("train", train), ("test", test))}
    if control != {"train": {"count": 1200, "sum": 11993084},
                   "test": {"count": 1502, "sum": 33351444}}:
        raise ValueError(f"Lab 3 split control failed: {control}")
    indices = np.random.default_rng(0).permutation(len(train))[:200]
    calibration = [train[index] for index in indices]
    categories = dict(Counter(row["label"] for row in calibration))
    name_sum = sum(int(Path(row["file"]).stem) for row in calibration)
    if name_sum != 1921470 or categories != dict(zip(CLASSES, (40, 35, 36, 33, 29, 27), strict=True)):
        raise ValueError(f"Calibration control failed: {name_sum}, {categories}")
    return test, calibration, control, {"count": len(calibration), "sum": name_sum, "classes": categories}


def load_image(row, root, transform):
    path = source_folder(root, row["part"]) / row["label"] / row["file"]
    with Image.open(path) as image:
        return transform(image.convert("RGB"))


class CalibrationImages(CalibrationDataReader):
    def __init__(self, rows, root, transform):
        self.items = iter(rows)
        self.root = root
        self.transform = transform

    def get_next(self):
        row = next(self.items, None)
        if row is None:
            return None
        return {"input": load_image(row, self.root, self.transform).unsqueeze(0).numpy()}


def quantize(prepared, destination, calibration, root, transform, per_channel):
    started = perf_counter()
    quantize_static(
        prepared, destination, CalibrationImages(calibration, root, transform),
        quant_format=QuantFormat.QDQ, activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8, per_channel=per_channel, calibrate_method=CalibrationMethod.MinMax,
    )
    return perf_counter() - started


def session(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def forward_batches(test, root, transform, predictors):
    outputs = [[] for _ in predictors]
    for start in range(0, len(test), 64):
        batch = torch.stack([load_image(row, root, transform) for row in test[start:start + 64]])
        for output, predict in zip(outputs, predictors, strict=True):
            output.append(predict(batch))
    logits = [np.concatenate(output) for output in outputs]
    if any(array.shape != (len(test), len(CLASSES)) or not np.isfinite(array).all() for array in logits):
        raise ValueError("Expected finite logits for every test image and class.")
    return logits


def macro_f1(labels, predictions):
    return float(f1_score(labels, predictions, labels=range(len(CLASSES)), average="macro", zero_division=0))


def quality(labels, predictions, reference):
    correct = predictions == labels
    base_correct = reference == labels
    b = int(np.count_nonzero(correct & ~base_correct))
    c = int(np.count_nonzero(~correct & base_correct))
    detectable = abs(b - c) > 2 * np.sqrt(b + c)
    return {"macro_f1": macro_f1(labels, predictions), "accuracy": float(correct.mean()),
            "agreement": float((predictions == reference).mean()), "b": b, "c": c,
            "accuracy_delta": (b - c) / len(labels), "paired_se": float(np.sqrt(b + c) / len(labels)),
            "detectable_change": bool(detectable)}


def benchmark(sessions, paths, first):
    feed = {"input": first}
    rounds = {name: [] for name in sessions}
    for _ in range(3):
        for name, runtime in sessions.items():
            for _ in range(20):
                runtime.run(["logits"], feed)
            samples = []
            for _ in range(100):
                started = perf_counter()
                runtime.run(["logits"], feed)
                ended = perf_counter()
                samples.append((ended - started) * 1000)
            rounds[name].append(samples)
    stats = {}
    for name, samples in rounds.items():
        p50, p95 = np.percentile(samples, (50, 95))
        stats[name] = {"size_mb": paths[name].stat().st_size / 1e6, "p50_ms": float(p50),
                       "p95_ms": float(p95), "round_p50_ms": np.median(samples, axis=1).tolist(),
                       "round_p95_ms": np.percentile(samples, 95, axis=1).tolist(), "samples_ms": samples}
    fp, int8 = (stats[name]["round_p50_ms"] for name in ("fp32", "int8"))
    stats["measured_difference"] = max(fp) < min(int8) or max(int8) < min(fp)
    stats["speedup_p50"] = stats["fp32"]["p50_ms"] / stats["int8"]["p50_ms"]
    return stats


def cpu_name():
    if platform.system() == "Windows":
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    return platform.processor() or platform.machine()


def experiment(args, directory):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    model, checkpoint = load_model(args.model)
    size = checkpoint["img_size"]
    transform = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(),
                                    transforms.Normalize(MEAN, STD)])
    test, calibration, control, calibration_control = split_rows(args.split)
    labels = np.array([CLASSES.index(row["label"]) for row in test])
    paths = {name: directory / f"lab4_{name}.onnx" for name in ("fp32", "int8", "int8_per_tensor")}
    seconds = {"export": export_fp32(model, size, paths["fp32"])}
    graph = graph_checks(paths["fp32"])
    print(f"Checking FP32 logits on {len(test)} images", flush=True)
    fp32 = session(paths["fp32"])
    pytorch_logits, fp32_logits = forward_batches(test, args.data, transform, (
        lambda batch: model(batch).numpy(),
        lambda batch: fp32.run(["logits"], {"input": batch.numpy()})[0],
    ))
    reference = fp32_logits.argmax(axis=1)
    difference = float(np.abs(fp32_logits - pytorch_logits).max())
    class_agreement = float((reference == pytorch_logits.argmax(axis=1)).mean())
    if difference >= 1e-3 or class_agreement != 1.0:
        raise ValueError(f"FP32 export failed: max logit difference {difference}, agreement {class_agreement}")
    export = {"max_abs_logit_difference": difference, "class_agreement": class_agreement,
              "pytorch_macro_f1": macro_f1(labels, pytorch_logits.argmax(axis=1)),
              "onnx_macro_f1": macro_f1(labels, reference), **graph}
    prepared = directory / "lab4_prep.onnx"
    started = perf_counter()
    quant_pre_process(paths["fp32"], prepared, skip_symbolic_shape=True)
    seconds["preprocess"] = perf_counter() - started
    scores = {"fp32": quality(labels, reference, reference)}
    for name, per_channel in (("int8", True), ("int8_per_tensor", False)):
        print(f"Calibrating and evaluating {name}", flush=True)
        seconds[name] = quantize(prepared, paths[name], calibration, args.data, transform, per_channel)
        runtime = session(paths[name])
        logits, = forward_batches(test, args.data, transform, (
            lambda batch: runtime.run(["logits"], {"input": batch.numpy()})[0],
        ))
        scores[name] = quality(labels, logits.argmax(axis=1), reference)
    accuracy = scores["fp32"]["accuracy"]
    first = load_image(test[0], args.data, transform).unsqueeze(0).numpy()
    print("Measuring one-thread CPU latency", flush=True)
    timing = benchmark({"fp32": fp32, "int8": session(paths["int8"])}, paths, first)
    return {"model": {"regime": checkpoint["regime"], "seed": checkpoint["seed"], "img_size": size,
                      "checkpoint_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
                      "split_sha256": hashlib.sha256(args.split.read_bytes()).hexdigest()},
            "split": control, "calibration": calibration_control, "export": export, "stage_seconds": seconds,
            "quality": scores, "fp32_accuracy_se": float(np.sqrt(accuracy * (1 - accuracy) / len(labels))),
            "latency": timing,
            "environment": {"cpu": cpu_name(), "os": platform.platform(), "python": platform.python_version(),
                            "torch": str(torch.__version__), "onnx": onnx.__version__, "onnxruntime": ort.__version__,
                            "provider": "CPUExecutionProvider", "intra_op_threads": 1, "inter_op_threads": 1,
                            "benchmark_image": test[0]["file"], "latency_batch": 1, "quality_batch": 64,
                            "warmups_per_round": 20, "measurements_per_round": 100, "rounds": 3}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("lab4_results.json"))
    args = parser.parse_args()
    with TemporaryDirectory(prefix="deepl-lab4-") as directory, torch.no_grad():
        results = experiment(args, Path(directory))
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Results: {args.out.resolve()}")


if __name__ == "__main__":
    main()
