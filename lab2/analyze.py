"""Select a threshold on validation, then evaluate the fixed rule on test."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

from checks import run_checks
from metrics import best_threshold, counts, threshold_table
from train import load_predictions, split_summary, write_json


def markdown(table: pd.DataFrame) -> str:
    def display(value: object) -> str:
        return f"{value:.8g}" if isinstance(value, (float, np.floating)) else str(value)
    rows = [list(table.columns), ["---"] * len(table.columns), *table.itertuples(index=False, name=None)]
    return "\n".join("| " + " | ".join(map(display, row)) + " |" for row in rows)


def error_examples(data: dict[str, np.ndarray], threshold: float) -> pd.DataFrame:
    y, scores = data["test_y"], data["test_scores"]
    rows = []
    for kind, mask in (("FP", (scores >= threshold) & (y == 0)),
                       ("FN", (scores < threshold) & (y == 1))):
        indices = np.flatnonzero(mask)
        indices = indices[np.argsort(data["test_row_ids"][indices])][:5]
        for i in indices:
            rows.append({"row": int(data["test_row_ids"][i]), "type": kind,
                         "label": int(y[i]), "score": float(scores[i]),
                         "Amount": float(data["test_amount"][i]),
                         "score_minus_threshold": float(scores[i]) - threshold})
    return pd.DataFrame(rows, columns=["row", "type", "label", "score", "Amount", "score_minus_threshold"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saved", type=Path, required=True)
    args = parser.parse_args()
    out = args.saved.resolve()
    source = Path(__file__).resolve().parent
    submitted = out == source / "results"
    report = source / "README.md" if submitted else out / "report.md"
    figure_prefix = "results/" if submitted else ""
    checks = run_checks(out)
    data = load_predictions(out)
    y, scores = data["validation_y"], data["validation_scores"]
    table = threshold_table(y, scores)
    table.to_csv(out / "threshold_candidates.csv", index=False, float_format="%.17g")
    threshold = best_threshold(table)
    write_json(out / "threshold.json", {"threshold": threshold if np.isfinite(threshold) else "inf",
                                        "rule": "score >= threshold", "chosen_on": "validation"})
    validation = pd.DataFrame([
        {"rule": name, "threshold": t, **counts(y, scores, t)}
        for name, t in (("0.5", .5), ("minimum cost", threshold), ("all negative", np.inf))
    ])
    validation.to_csv(out / "validation_summary.csv", index=False)
    auc = float(roc_auc_score(y, scores))
    precision, recall, _ = precision_recall_curve(y, scores)
    fpr, tpr, _ = roc_curve(y, scores)
    for name, xs, ys, xlabel, ylabel in (
        ("pr", recall, precision, "Recall", "Precision"),
        ("roc", fpr, tpr, "False positive rate", "True positive rate"),
    ):
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(xs, ys)
        ax.set(xlabel=xlabel, ylabel=ylabel, title=f"Validation {name.upper()}")
        ax.grid(alpha=.25)
        fig.tight_layout()
        fig.savefig(out / f"{name}.png", dpi=150)
        plt.close(fig)

    test = counts(data["test_y"], data["test_scores"], threshold)
    baseline = counts(data["test_y"], data["test_scores"], .5)
    tp, fp, fn, tn = (test[k] for k in ("TP", "FP", "FN", "TN"))
    test_precision = tp / (tp + fp) if tp + fp else None
    test_recall = tp / (tp + fn) if tp + fn else None
    write_json(out / "test_metrics.json", {**test, "precision": test_precision, "recall": test_recall,
                                           "baseline_0.5": baseline, "validation_roc_auc": auc})
    errors = error_examples(data, threshold)
    errors.to_csv(out / "errors.csv", index=False)
    confusion = pd.DataFrame([[tn, fp], [fn, tp]], index=["true 0", "true 1"], columns=["predicted 0", "predicted 1"])
    confusion.to_csv(out / "test_confusion.csv")
    check_tables = pd.concat([table.assign(case=name) for name, table in checks.items()])
    precision_text = f"{test_precision:.4g}" if test_precision is not None else "не визначена"
    recall_text = f"{test_recall:.4g}" if test_recall is not None else "не визначена"
    sections = [
        "## Результати",
        f"Поріг `{threshold:.17g}` зменшив вартість на test з **{baseline['C']} до {test['C']}**. "
        f"FN змінилися з {baseline['FN']} до {fn}, FP — з {baseline['FP']} до {fp}.",
        markdown(split_summary(data)), markdown(validation),
        f"На validation ROC-AUC = **{auc:.4f}**. AUC оцінює ранжування за всіма порогами, "
        "а вартість і precision залежать від конкретного порога й частки шахрайства. "
        f"[PR]({figure_prefix}pr.png) · [ROC]({figure_prefix}roc.png) · "
        f"[Усі кандидати]({figure_prefix}threshold_candidates.csv)",
        "### Перевірки A/B/C",
        "Повний перебір і швидкий пошук дали однакові лічильники. "
        f"Пороги A/B/C — `{best_threshold(checks['A'])}`, `{best_threshold(checks['B'])}`, "
        f"`{best_threshold(checks['C'])}`. A перевіряє однакові оцінки, B — відсутність позитивних міток, "
        "C — вибір більшого порога за нічиєї.",
        markdown(check_tables[["case", "threshold", "TP", "FP", "FN", "TN", "C"]]),
        "### Test", markdown(confusion.reset_index(names="class")),
        f"Precision = {precision_text}, recall = {recall_text}. За нульового знаменника метрика не визначена.",
        "Перші п’ять FP і FN за нульовим номером рядка CSV.", markdown(errors),
    ]
    error_stats = []
    for kind, sample in errors.groupby("type", sort=False):
        error_stats.append(f"{kind} — Amount {sample.Amount.min():.2f}–{sample.Amount.max():.2f}, "
                           f"медіанна відстань до порога {sample.score_minus_threshold.abs().median():.4g}")
    if error_stats:
        sections.append(". ".join(error_stats) + ". Ці приклади не дають підстав встановити причину помилок.")
    sections.append(
        "Виграш може змінитися, бо між двома порогами на validation і test різна кількість шахрайських та звичайних транзакцій. "
        f"[Середовище навчання]({figure_prefix}run.json)"
    )
    prefix = "# Лабораторна робота №2"
    if submitted:
        prefix = report.read_text(encoding="utf-8").split("<!-- results -->", 1)[0].rstrip()
    report.write_text(prefix + "\n\n<!-- results -->\n\n" + "\n\n".join(sections) + "\n", encoding="utf-8")
    print(validation.to_string(index=False))
    print(f"Validation ROC-AUC: {auc:.8f}; test: {test}")


if __name__ == "__main__":
    main()
