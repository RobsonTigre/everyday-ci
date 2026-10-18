"""Evaluate the actual student matched-window snippets on their shared CSV."""
from pathlib import Path
import subprocess
import unittest

import numpy as np
import pandas as pd
from rdrobust import rdplot, rdrobust

ROOT = Path(__file__).resolve().parents[1]
BASE = "8. Regression discontinuity design Cutoffs as natural experiments"
MARKER = "# Match the outcome and treatment jumps to the main fuzzy model's windows."


class FuzzyRatioTests(unittest.TestCase):
    def block(self, suffix):
        text = (ROOT / (BASE + suffix)).read_text()
        self.assertTrue(MARKER in text, "Show the ratio using the main fuzzy model's same h and b")
        return text.split(MARKER, 1)[1].split("# With covariates", 1)[0]

    def test_python_actual_matched_windows(self):
        block = self.block(".py")
        data = pd.read_csv(ROOT / "data/rdd_fuzzy_fintech.csv")
        model = rdrobust(data.y, data.x, c=650, fuzzy=data.D.values)
        namespace = {"data_fuzzy": data, "fuzzy_result": model, "rdrobust": rdrobust, "np": np}
        exec(compile(block, BASE + ".py", "exec"), namespace)
        self.assertLess(abs(namespace["fuzzy_ratio"] - model.coef.iloc[0, 0]), 1e-6)
        for name in ("fuzzy_reduced_form", "fuzzy_first_stage"):
            np.testing.assert_allclose(namespace[name].bws.values, model.bws.values, atol=0, rtol=0)

    def test_r_actual_matched_windows(self):
        block = self.block(".R")
        prefix = 'library(rdrobust)\ndata_fuzzy <- read.csv("data/rdd_fuzzy_fintech.csv")\nfuzzy_result <- rdrobust(data_fuzzy$y, data_fuzzy$x, c=650, fuzzy=data_fuzzy$D)\n'
        suffix = '\nstopifnot(abs(fuzzy_ratio - fuzzy_result$coef[1]) < 1e-6, identical(fuzzy_reduced_form$bws, fuzzy_result$bws), identical(fuzzy_first_stage$bws, fuzzy_result$bws))\n'
        result = subprocess.run(["Rscript", "-e", prefix + block + suffix], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


    def test_python_actual_plot_gap_matches_paired_reduced_form(self):
        source = (ROOT / (BASE + ".py")).read_text()
        plot_block = source.split("# 4.4 Visualization: The Reduced Form Jump", 1)[1].split("plt.savefig", 1)[0]
        data = pd.read_csv(ROOT / "data/rdd_fuzzy_fintech.csv")
        model = rdrobust(data.y, data.x, c=650, fuzzy=data.D.values)
        namespace = {"data_fuzzy": data, "fuzzy_result": model, "rdrobust": rdrobust, "np": np}
        exec(compile(self.block(".py"), BASE + ".py", "exec"), namespace)
        plots = []
        def capture_plot(*args, **kwargs):
            plots.append(rdplot(*args, **kwargs, hide=True))
            return plots[-1]
        namespace["rdplot"] = capture_plot
        exec(compile(plot_block, BASE + ".py", "exec"), namespace)
        gap = plots[0].coef.iloc[0, 1] - plots[0].coef.iloc[0, 0]
        target = namespace["fuzzy_reduced_form"].coef.iloc[0, 0]
        self.assertLess(abs(gap - target), 1e-6, f"Displayed cutoff gap {gap:.6f} must equal the paired outcome jump {target:.6f}")

    def test_r_actual_plot_gap_matches_paired_reduced_form(self):
        source = (ROOT / (BASE + ".R")).read_text()
        plot_block = source.split("# 4.4 Visualization: The Reduced Form Jump", 1)[1].split("# Add styling", 1)[0]
        prefix = 'library(rdrobust)\nlibrary(dplyr)\ndata_fuzzy <- read.csv("data/rdd_fuzzy_fintech.csv")\nfuzzy_result <- rdrobust(data_fuzzy$y, data_fuzzy$x, c=650, fuzzy=data_fuzzy$D)\n'
        capture = '\nrdplot <- function(...) { fuzzy_plotted <<- rdrobust::rdplot(..., hide=TRUE); fuzzy_plotted }\n'
        suffix = '\nplot_gap <- fuzzy_plotted$coef[1,2] - fuzzy_plotted$coef[1,1]\ncat(sprintf("Displayed cutoff gap %.6f; paired outcome jump %.6f\\n",plot_gap,fuzzy_reduced_form$coef[1]))\nstopifnot(abs(plot_gap - fuzzy_reduced_form$coef[1]) < 1e-6)\n'
        result = subprocess.run(["Rscript", "-e", prefix + self.block(".R") + capture + plot_block + suffix], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
