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
    sections = ["## Результати", "### Перевірки до навчання", markdown(control),
                "Файли між частинами не перетинаються; всі 18 контрольних клітинок збігаються.",
                "![Навчальний пакет](figures/batch.png)",
                markdown([{"regime": k, "initial_loss": v, "ln(6)": float(np.log(6))}
                          for k, v in checks["initial_loss"].items()]),
                "Початкова втрата не мусить дорівнювати ln(6): випадкова голова дає нерівні ймовірності, "
                "а розподіли ознак у scratch і frozen різні.",
                markdown([{**{k: v for k, v in r.items() if k != "groups"}, "groups": str(r["groups"])}
                          for r in checks["parameters"]])]
    for mode in checks["modes"]:
        sections.extend([f"Режим `{mode['regime']}` після 5 кроків:", markdown(mode["modules"])])
    sections.extend(["### Навмисна помилка BatchNorm", "Прогноз до досліду: " + checks["bn_prediction"],
                     markdown(checks["bn_bug"]["modules"]),
                     "Максимальні абсолютні зміни layer1.0.bn1: " + str(checks["bn_bug"]["layer1.0.bn1_max_changes"]),
                     f"Train-loss у досліді з помилкою: {checks['bn_bug']['losses'][0]:.6f} -> "
                     f"{checks['bn_bug']['losses'][-1]:.6f}. Його спад не показує, що BN кістяка змінився.",
                     "Прогноз підтвердився. requires_grad керує градієнтами параметрів, а не буферами BN. "
                     "У layer4 режиму unfreeze ці буфери мають оновлюватися разом із навчанням стадії.",
                     "### Сім запусків", markdown(rows), "![Криві](figures/curves.png)",
                     "### Розклад ефектів", markdown([{"quantity": k, "value": v} for k, v in effect.items()])])
    for name, key in (("Попереднє навчання", "delta_pre"), ("Розморожування", "delta_unf")):
        value = effect[key]
        verdict = "не відрізняється від розкиду; результату немає" if abs(value) <= effect["R"] else "перевищує розкид; вартий перевірки, але не доведений"
        sections.append(f"{name}: напрям {'додатний' if value > 0 else 'від’ємний' if value < 0 else 'нульовий'}, "
                        f"величина {100 * value:.4f} пунктів macro-F1; R = {100 * effect['R']:.4f} пунктів; {verdict}.")
    times = {r: float(np.mean([v["training_seconds"] for v in logs if v["regime"] == r]))
             for r in ("scratch", "frozen", "unfreeze")}
    scratch = logs[0]["history"]
    sections.extend([
        "Зерно змінює всі початкові ваги в scratch, лише нову голову у frozen/unfreeze, "
        "а також порядок пакетів та аугментації у всіх режимах. Розморожування додає шляхи зміни "
        "параметрів і BN, тому розкид може відрізнятися.",
        f"R оцінено лише для frozen/unfreeze: розкид scratch за одним запуском невідомий, тому "
        f"висновок про delta_pre умовний щодо scratch seed=1. Його train-loss: "
        f"{scratch[0]['train_loss']:.6f} -> {scratch[-1]['train_loss']:.6f}; validation F1: "
        f"{scratch[0]['validation_f1']:.6f} -> {scratch[-1]['validation_f1']:.6f}.",
        f"Максимум validation F1 на кривій scratch за 8 епох: "
        f"{max(e['validation_f1'] for e in scratch):.6f}; мінімум підсумкових F1 frozen: "
        f"{min(r['history'][-1]['validation_f1'] for r in logs if r['regime'] == 'frozen'):.6f}. "
        "Це допомагає оцінити відставання в цьому запуску, але не замінює повторів scratch із різними зернами.",
        f"Однакові 304 кроки не означають однаковий час: середні секунди {times}. "
        "Час охоплює цикл восьми епох разом із validation, без завантаження ваг і початкових перевірок.",
        "### Вибір і test",
        f"За найбільшим середнім validation macro-F1 вибрано `{chosen['regime']}`, "
        f"потім найкращий його запуск seed={chosen['seed']}. За точних нічиїх використовується порядок "
        "scratch, frozen, unfreeze та менше зерно.",
        "Вибір між frozen/unfreeze визначається шумом." if abs(effect["delta_unf"]) <= effect["R"] else "Різниця frozen/unfreeze перевищує прийнятий поріг шуму.",
        f"Test macro-F1 = **{f1:.8f}**, validation обраного запуску = **{validation_f1:.8f}**. "
        f"Різниця test − validation дорівнює {f1_gap:+.8f}. Accuracy = {accuracy:.8f}, "
        f"SE accuracy = {se:.8f} при n={len(y)}. SE accuracy дає лише грубий орієнтир масштабу варіації. "
        "Стандартну похибку для macro-F1 не оцінювали.",
        "Вибір найкращого seed може завищувати оцінку validation в середньому. "
        + ("У цьому досліді завищення не спостерігаємо."
           if f1_gap >= 0 else "У цьому досліді macro-F1 на validation вища.")
        + " Через скінченний розмір і різний склад вибірок розрив може змінюватися за розміром і знаком.",
        markdown([{"true / predicted": CLASSES[i], **{c: int(matrix[i, j]) for j, c in enumerate(CLASSES)}} for i in range(6)]),
        "![Матриця плутанини](figures/confusion.png)", markdown(errors), "![Помилки](figures/errors.png)",
        "Плутанина glacier ↔ mountain є властивістю набору; перерозмітка чи зміна test не виконуються.",
        "### Висновок",
        f"За бюджету 304 кроки: delta_pre = {100 * effect['delta_pre']:.4f} п.п., "
        f"delta_unf = {100 * effect['delta_unf']:.4f} п.п., R = {100 * effect['R']:.4f} п.п. "
        "Інтерпретацію обох ефектів наведено окремо вище. За довшого навчання scratch міг би скоротити відставання.",
        (f"Для нової задачі з 200 зображеннями на клас почав би з frozen: "
         f"час {times['frozen']:.1f} с і розмах {100 * effect['range_frozen']:.4f} п.п.; "
         "виграш unfreeze не перевищує R." if abs(effect["delta_unf"]) <= effect["R"] else
         f"Для нової задачі з 200 зображеннями на клас взяв би {chosen['regime']}: "
         f"результат validation за цього бюджету найвищий; середній час {times[chosen['regime']]:.1f} с, "
         f"поріг шуму {100 * effect['R']:.4f} п.п.; переносимість виграшу потребує нової перевірки."),
        "### Відтворюваність",
        f"SHA-256 split.csv: `{split_hash(split)}`. У всіх семи журналах перевірено однаковий хеш. "
        f"Коміти: {sorted({r['git']['commit'] for r in logs})}; "
        f"наявність незакомічених змін на момент навчання: {any(r['git']['dirty'] for r in logs)}.",
        f"Python {chosen['python']}, torch {chosen['torch']}, torchvision {chosen['torchvision']}; "
        f"навчання {chosen['device_name']}, оцінка test CPU. Фіксовані налаштування умови збережено; "
        "версії середовища наведено у pyproject.toml, uv.lock і run.json.",
    ])
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
