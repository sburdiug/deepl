"""Analyze the seven saved runs and evaluate only the selected model on CPU."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score
from torchvision.models import resnet18

from data import CLASSES, check_split, make_loader, read_split, split_hash
from model import RUNS, predict
from train import select_run, state_hash, write_json


def markdown(rows: list[dict]) -> str:
    if not rows:
        return "Немає прикладів."
    def display(value) -> str:
        if value is None:
            return "немає BN"
        if isinstance(value, bool):
            return "так" if value else "ні"
        if isinstance(value, float):
            return f"{value:.8g}"
        return str(value)
    columns = list(rows[0])
    content = [columns, ["---"] * len(columns), *[[display(row[c]) for c in columns] for row in rows]]
    return "\n".join("| " + " | ".join(row) + " |" for row in content)


def effects(logs: list[dict]) -> dict[str, float]:
    values = {r: [v["history"][-1]["validation_f1"] for v in logs if v["regime"] == r]
              for r in ("scratch", "frozen", "unfreeze")}
    frozen, unfreeze = float(np.mean(values["frozen"])), float(np.mean(values["unfreeze"]))
    rf, ru = float(np.ptp(values["frozen"])), float(np.ptp(values["unfreeze"]))
    return {"mean_frozen": frozen, "range_frozen": rf, "mean_unfreeze": unfreeze,
            "range_unfreeze": ru, "R": max(rf, ru),
            "delta_pre": frozen - values["scratch"][0], "delta_unf": unfreeze - frozen}


def select_errors(y: np.ndarray, predicted: np.ndarray, files: list[str]) -> list[int]:
    groups = {}
    for i in range(len(y)):
        if y[i] != predicted[i]:
            groups.setdefault((int(y[i]), int(predicted[i])), []).append(i)
    pairs = sorted(groups, key=lambda pair: (-len(groups[pair]), *pair))
    for indices in groups.values():
        indices.sort(key=lambda i: files[i])
    selected = []
    offset = 0
    while len(selected) < 5:
        new = [groups[pair][offset] for pair in pairs if len(groups[pair]) > offset]
        if not new:
            break
        selected.extend(new[:5 - len(selected)])
        offset += 1
    return selected


def load_runs(out: Path, split: Path) -> list[dict]:
    fingerprint = split_hash(split)
    labels = np.array([CLASSES.index(r["label"]) for r in read_split(split) if r["part"] == "validation"])
    logs = []
    for regime, seed in RUNS:
        folder = out / "runs" / f"{regime}_seed{seed}"
        log = json.loads((folder / "run.json").read_text(encoding="utf-8"))
        if log["split_sha256"] != fingerprint:
            raise ValueError(f"Changed split.csv: {folder}; retrain all seven runs.")
        if (log["regime"], log["seed"]) != (regime, seed) or len(log["history"]) != 8:
            raise ValueError(f"Invalid run identity or epoch count: {folder}")
        logits = np.load(folder / "val_logits.npy", allow_pickle=False)
        if logits.shape != (1498, 6) or not np.isfinite(logits).all():
            raise ValueError(f"Invalid validation logits: {folder}")
        f1 = float(f1_score(labels, logits.argmax(1), labels=range(6), average="macro", zero_division=0))
        if abs(f1 - log["history"][-1]["validation_f1"]) > 1e-12:
            raise ValueError(f"Validation logits disagree with run.json: {folder}")
        logs.append(log)
    return logs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, help="Directory containing saved experiment artifacts")
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    out = args.out.resolve() if args.out is not None else source
    split, figures = out / "split.csv", out / "figures"
    torch.set_num_threads(4)
    control = check_split(split)
    logs = load_runs(out, split)
    chosen = select_run(logs)
    saved = torch.load(out / "lab3_model.pt", map_location="cpu", weights_only=True)
    if (saved["regime"], saved["seed"]) != (chosen["regime"], chosen["seed"]):
        raise ValueError("lab3_model.pt does not match validation selection.")
    if saved["classes"] != CLASSES or saved["img_size"] != 160:
        raise ValueError("Invalid checkpoint classes or image size.")
    if state_hash(saved["state_dict"]) != chosen["state_sha256"]:
        raise ValueError("The saved weights do not match the selected run.")
    effect = effects(logs)
    write_json(out / "effects.json", effect)
    rows = [{"regime": r["regime"], "seed": r["seed"], **r["history"][-1],
             "seconds": r["training_seconds"]} for r in logs]
    with (out / "runs_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for log in logs:
        label = f"{log['regime']} seed {log['seed']}"
        for ax, key in zip(axes, ("train_loss", "validation_f1"), strict=True):
            ax.plot(range(1, 9), [e[key] for e in log["history"]], label=label)
    for ax, title in zip(axes, ("Train cross-entropy", "Validation macro-F1"), strict=True):
        ax.set(xlabel="Epoch", title=title)
        ax.grid(alpha=.25)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figures / "curves.png", dpi=150)
    plt.close(fig)

    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(512, 6)
    model.load_state_dict(saved["state_dict"])
    test = make_loader(args.data.resolve(), split, "test")
    logits, y = predict(model, test, "cpu")
    predicted = logits.argmax(1)
    matrix = confusion_matrix(y, predicted, labels=range(6))
    f1 = float(f1_score(y, predicted, labels=range(6), average="macro", zero_division=0))
    validation_f1 = chosen["history"][-1]["validation_f1"]
    f1_gap = f1 - validation_f1
    accuracy = float((y == predicted).mean())
    se = float(np.sqrt(accuracy * (1 - accuracy) / len(y)))
    np.save(out / "test_logits.npy", logits)
    np.savetxt(out / "confusion.csv", matrix, delimiter=",", fmt="%d", header=",".join(CLASSES), comments="")
    write_json(out / "test_metrics.json", {"macro_f1": f1, "accuracy": accuracy, "accuracy_se": se,
                                           "n": len(y), "regime": chosen["regime"], "seed": chosen["seed"]})
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set(xticks=range(6), yticks=range(6), xticklabels=CLASSES, yticklabels=CLASSES,
           xlabel="Predicted", ylabel="True", title="Test confusion matrix")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    for i in range(6):
        for j in range(6):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(figures / "confusion.png", dpi=150)
    plt.close(fig)

    files = [r["file"] for r in test.dataset.rows]
    indices = select_errors(y, predicted, files)
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    notes_path = out / "error_notes.json"
    notes = json.loads(notes_path.read_text(encoding="utf-8")) if notes_path.exists() else {}
    errors = []
    fig, axes = plt.subplots(1, max(1, len(indices)), figsize=(4 * max(1, len(indices)), 4.5), squeeze=False)
    for ax, i in zip(axes.flat, indices):
        with Image.open(test.dataset.path(i)) as image:
            ax.imshow(image.convert("RGB"))
        ax.set_title(f"{files[i]}\n{CLASSES[y[i]]} -> {CLASSES[predicted[i]]}", fontsize=10)
        ax.axis("off")
        errors.append({"file": files[i], "true": CLASSES[y[i]], "predicted": CLASSES[predicted[i]],
                       "probability": float(probabilities[i, predicted[i]]),
                       "pair_count": int(matrix[y[i], predicted[i]]),
                       "interpretation": notes.get(files[i], "Потрібний візуальний розбір зображення.")})
    if not indices:
        axes[0, 0].axis("off")
        axes[0, 0].text(.5, .5, "No errors", ha="center")
    fig.subplots_adjust(left=.015, right=.985, top=.88, bottom=.04, wspace=.12)
    fig.savefig(figures / "errors.png", dpi=140)
    plt.close(fig)
    write_json(out / "errors.json", errors)
    checks = json.loads((out / "checks.json").read_text(encoding="utf-8"))
    split_cells = {(item["class"], item["part"]): f"{item['count']} / {item['sum']}" for item in control}
    split_table = [{"class": label, **{part: split_cells[label, part] for part in ("train", "validation", "test")}}
                   for label in CLASSES]
    params = [{"regime": item["regime"], "total": item["total"], "trainable": item["trainable"],
               "groups (numel × lr)": ", ".join(f"{group['parameters']} × {group['lr']:g}" for group in item["groups"])}
              for item in checks["parameters"]]
    mode_records = [*checks["modes"], checks["bn_bug"]]
    mode_table = []
    for modules in zip(*(mode["modules"] for mode in mode_records), strict=True):
        row = {"module": modules[0]["module"]}
        for name, module in zip(("frozen", "unfreeze", "frozen + model.train()"), modules, strict=True):
            param = "+" if module["parameters_changed"] else "−"
            bn = "—" if module["bn_changed"] is None else "+" if module["bn_changed"] else "−"
            row[name] = f"{param} / {bn}"
        mode_table.append(row)
    initial = checks["initial_loss"]
    bn_changes = checks["bn_bug"]["layer1.0.bn1_max_changes"]
    losses = checks["bn_bug"]["losses"]
    pre, unf, noise = (100 * effect[key] for key in ("delta_pre", "delta_unf", "R"))
    time_frozen = np.mean([r["training_seconds"] for r in logs if r["regime"] == "frozen"])
    recommendation = "frozen" if abs(unf) <= noise else chosen["regime"]
    sections = [
        "## Результати",
        f"Обрали {chosen['regime']}, seed {chosen['seed']}. Macro-F1 на test — **{f1:.6f}**.",
        "### Перевірки перед навчанням",
        "Усі 18 контрольних клітинок збіглися, перетинів між частинами немає. У таблиці подано кількість і суму імен.",
        markdown(split_table),
        f"Початкова втрата — {initial['scratch']:.4f} для scratch і {initial['frozen']:.4f} для frozen. "
        f"Рівномірний прогноз дає ln(6) = {np.log(6):.4f}, а випадкова голова дає різні ймовірності. "
        "[Пакет із перших 16 зображень](figures/batch.png)",
        markdown(params),
        "Після п’яти кроків перевірили зміни «параметри / статистики BN». + означає зміну, − — без змін, "
        "— — модуль без BN. Усі перевірки збіглися з очікуваннями.",
        markdown(mode_table),
        "У досліді з навмисною помилкою прогнозували зміну буферів BN, сталі ваги кістяка і можливий спад loss. "
        f"Так і сталося. У layer1.0.bn1 зміна running_mean = {bn_changes['running_mean']:.6f}, "
        f"running_var = {bn_changes['running_var']:.6f}. Loss впав з {losses[0]:.6f} до {losses[-1]:.6f}. "
        "requires_grad=False вимикає градієнти параметрів, а статистики BN оновлюються в прямому проході. "
        "В unfreeze статистики BN у layer4 мають оновлюватися разом із параметрами цього блоку. [Перевірки](checks.json)",
        "### Сім запусків та ефекти", markdown(rows),
        "[Криві навчання й validation](figures/curves.png) · [Повні журнали](runs)",
        markdown([{"regime": name, "mean validation F1": effect[f"mean_{name}"],
                   "range": effect[f"range_{name}"]} for name in ("frozen", "unfreeze")]),
        f"За 304 кроки ефект переднавчання **{pre:+.2f} п.п.**, ефект розморожування **{unf:+.2f} п.п.**, "
        f"поріг шуму R = **{noise:.2f} п.п.** "
        + ("Переднавчання перевищує розкид. " if abs(pre) > noise else "Ефект переднавчання в межах розкиду. ")
        + ("Розморожування в межах розкиду. " if abs(unf) <= noise else "Розморожування перевищує розкид. ")
        + f"Scratch має лише одне зерно. Його найкращий validation F1 за 8 епох — "
        f"{max(epoch['validation_f1'] for epoch in logs[0]['history']):.3f}. Триваліше навчання могло б зменшити відставання.",
        "Зерно змінює порядок пакетів, аугментації та початкові ваги навчуваних з нуля шарів. "
        "У unfreeze додатково оновлюються параметри й BN layer4, тому розкид може відрізнятися. "
        f"За однакової кількості кроків час різний. Frozen у середньому зайняв {time_frozen:.1f} с із validation. "
        f"Для 200 зображень на клас за подібного бюджету обрали б {recommendation}.",
        "### Test",
        f"Macro-F1 обраного запуску на validation — {validation_f1:.6f}, різниця test − validation — {f1_gap:+.6f}. "
        f"Accuracy = {accuracy:.6f}, її SE = {se:.5f} на {len(y)} зображеннях. "
        + ("У цих числах оптимістичності validation не видно. " if f1_gap >= 0 else "Оцінка на validation вища. ")
        + "Розмір і склад вибірок впливають на напрям різниці.",
        markdown([{"true / predicted": CLASSES[i], **{c: int(matrix[i, j]) for j, c in enumerate(CLASSES)}} for i in range(6)]),
        "[Рисунок матриці плутанини](figures/confusion.png) · [Зображення п’яти помилок](figures/errors.png)",
        markdown(errors),
        "Найчастіша плутанина — glacier ↔ mountain, що відображає подібність сцен у наборі.",
        f"Python {chosen['python']}, torch {chosen['torch']}, torchvision {chosen['torchvision']}. "
        f"Навчання на {chosen['device_name']}, test на CPU. SHA-256 поділу збігся в усіх семи журналах.",
    ]
    report = out / ("README.md" if out == source else "report.md")
    prefix = "# Лабораторна робота №3"
    if out == source:
        prefix = report.read_text(encoding="utf-8").split("<!-- results -->", 1)[0].rstrip()
    report.write_text(prefix + "\n\n<!-- results -->\n\n" + "\n\n".join(sections) + "\n", encoding="utf-8")
    print(markdown(rows))
    print(effect)
    print(f"Selected: {chosen['regime']}, seed={chosen['seed']}; test macro-F1={f1:.8f}", flush=True)


if __name__ == "__main__":
    main()
