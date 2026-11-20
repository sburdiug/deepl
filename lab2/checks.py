"""Independent exhaustive checks of grouped scores, boundaries and ties."""

from pathlib import Path

import numpy as np
import pandas as pd

from metrics import best_threshold, threshold_table


def run_checks(out: Path | None = None) -> dict[str, pd.DataFrame]:
    cases = {
        "A": ([.9, .7, .7, .4, .2, .1], [0, 1, 0, 1, 0, 0], .4),
        "B": ([.9, .6, .3], [0, 0, 0], np.inf),
        "C": ([.8] + [.5] * 11 + [.2], [1, 1] + [0] * 11, .8),
    }
    results = {}
    for name, (scores, labels, expected) in cases.items():
        scores, y = np.asarray(scores), np.asarray(labels)
        rows = []
        for t in [np.inf] + sorted(set(scores.tolist()), reverse=True):
            tp = fp = fn = tn = 0
            for score, label in zip(scores, y, strict=True):
                if score >= t:
                    if label == 1:
                        tp += 1
                    else:
                        fp += 1
                elif label == 1:
                    fn += 1
                else:
                    tn += 1
            rows.append([t, tp, fp, fn, tn, 10 * fn + fp])
        slow = pd.DataFrame(rows, columns=["threshold", "TP", "FP", "FN", "TN", "C"])
        fast = threshold_table(y, scores)
        pd.testing.assert_frame_equal(fast, slow, check_dtype=False, check_exact=True)
        minimum = min(row[-1] for row in rows)
        slow_best = max(row[0] for row in rows if row[-1] == minimum)
        assert best_threshold(fast) == slow_best == expected, name
        results[name] = fast
        if out is not None:
            fast.to_csv(out / f"check_{name}.csv", index=False)
        print(f"Check {name}: passed, threshold={slow_best}")
    return results


if __name__ == "__main__":
    run_checks()
