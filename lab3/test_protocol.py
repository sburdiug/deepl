"""Small tests for selection and error ordering, independent of training."""

import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch

from analyze import effects, select_errors
from model import RUNS
from train import main as train, select_run, state_hash


class ProtocolTests(unittest.TestCase):
    def test_selection_uses_regime_mean_before_best_seed(self):
        scores = [.5, .9, .9, .9, .95, .8, .8]
        logs = [{"regime": r, "seed": s, "history": [{"validation_f1": f}]}
                for (r, s), f in zip(RUNS, scores, strict=True)]
        self.assertEqual((select_run(logs)["regime"], select_run(logs)["seed"]), ("frozen", 1))
        result = effects(logs)
        self.assertAlmostEqual(result["delta_pre"], .4)
        self.assertAlmostEqual(result["delta_unf"], -.05)
        self.assertAlmostEqual(result["R"], .15)

    def test_error_pairs_and_lexical_filenames(self):
        y = np.array([0, 0, 1, 1, 2, 2])
        predicted = np.array([1, 1, 0, 0, 0, 2])
        files = ["2.jpg", "10.jpg", "4.jpg", "3.jpg", "5.jpg", "6.jpg"]
        self.assertEqual(select_errors(y, predicted, files), [1, 3, 4, 0, 2])
        self.assertEqual(select_errors(y, y, files), [])

    def test_weight_fingerprint_detects_changes(self):
        state = {"weight": torch.tensor([1., 2.])}
        initial = state_hash(state)
        state["weight"][0] += 1
        self.assertNotEqual(initial, state_hash(state))

    def test_training_refuses_existing_output_directory(self):
        with TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "lab3_model.pt"
            checkpoint.write_bytes(b"saved checkpoint")
            with patch("sys.argv", ["train.py", "--data", folder, "--out", folder]):
                with redirect_stderr(StringIO()) as stderr, self.assertRaises(SystemExit) as caught:
                    train()
            self.assertEqual(caught.exception.code, 2)
            self.assertIn("--out must name a new directory", stderr.getvalue())
            self.assertEqual(checkpoint.read_bytes(), b"saved checkpoint")


if __name__ == "__main__":
    unittest.main()
