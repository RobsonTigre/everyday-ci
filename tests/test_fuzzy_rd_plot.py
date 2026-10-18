"""Check the student script's fuzzy RD probability plot on binary outcomes.

Run with: python -m unittest discover -s tests -p test_fuzzy_rd_plot.py -v
"""
import ast
from pathlib import Path
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "8. Regression discontinuity design Cutoffs as natural experiments.py"


def run_plot(data):
    source = SCRIPT.read_text()
    block = source.split("# Visualize the first stage\n", 1)[1].split("# 4.3 Estimating the LATE", 1)[0]
    tree = ast.parse(block)
    # Exercise the real plotting code without opening windows or overwriting figures.
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and isinstance(node.value.func.value, ast.Name)
        and node.value.func.value.id == "plt"
        and node.value.func.attr in {"show", "savefig"}
    )]
    namespace = {"plt": plt, "lowess": lowess, "data_fuzzy": data}
    exec(compile(tree, str(SCRIPT), "exec"), namespace)
    plt.close(namespace["fig"])
    return namespace


class FuzzyRDPlotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = pd.read_csv(ROOT / "data/rdd_fuzzy_fintech.csv")
        cls.published = run_plot(cls.data)

    def test_recovers_known_binary_fractions(self):
        # Valid minority outcomes must contribute to the estimated probabilities.
        below = np.tile([0, 0, 1, 0, 0, 0, 0, 1, 0, 0], 100)
        data = pd.DataFrame({
            "x": np.r_[np.linspace(500, 649, 1000), np.linspace(651, 800, 1000)],
            "D": np.r_[below, 1 - below],
        })
        plotted = run_plot(data)
        for side, probability in [("below", 0.2), ("above", 0.8)]:
            with self.subTest(side=side):
                np.testing.assert_allclose(
                    plotted["lowess_" + side][100:-100, 1], probability, atol=0.02, rtol=0,
                    err_msg="A probability curve must preserve genuine minority binary outcomes",
                )

    def test_published_curve_agrees_with_nearby_loan_receipt(self):
        for side, index, mask in [
            ("below", -1, (self.data.x >= 640) & (self.data.x < 650)),
            ("above", 0, (self.data.x >= 650) & (self.data.x < 660)),
        ]:
            with self.subTest(side=side):
                curve = self.published["lowess_" + side][:, 1]
                self.assertTrue(np.isfinite(curve).all())
                self.assertTrue(((curve >= 0) & (curve <= 1)).all())
                self.assertLess(
                    abs(curve[index] - self.data.loc[mask, "D"].mean()), 0.05,
                    "The cutoff curve must agree with nearby observed uptake within 5 percentage points",
                )

    def test_plot_has_no_unsupported_uncertainty_ribbon(self):
        ribbons = [artist for artist in self.published["ax"].collections
                   if isinstance(artist, PolyCollection)]
        self.assertEqual(len(ribbons), 0, "A fixed-width ribbon is not estimated uncertainty")


if __name__ == "__main__":
    unittest.main(verbosity=2)
