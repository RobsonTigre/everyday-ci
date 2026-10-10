"""Behavioral checks for the student script's real Chapter 11 CausalImpact code.

Run with the Chapter 11 environment: python tests/test_time_series_causalimpact.py.
These execute the actual analysis sections, without refitting unrelated C-ARIMA models.
"""
import ast
import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "4")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("STUDENT_ROOT", Path(__file__).resolve().parents[1]))
SCRIPT = ROOT / "11. Time series methods Measuring impact without a control group.py"


def run_examples():
    source = SCRIPT.read_text()
    namespace = {"pd": pd, "np": np, "plt": plt,
                 "data": pd.read_csv(ROOT / "data/time_series_campaign.csv", parse_dates=["date"]),
                 "intervention_time": 85, "total_days": 120, "placebo_intervention_time": 60}
    for node in ast.parse(source).body:
        is_import = isinstance(node, ast.ImportFrom) and node.module == "causalimpact"
        is_plot = isinstance(node, ast.FunctionDef) and node.name in {"apply_book_style", "plot_causalimpact_result"}
        is_palette = isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "BOOK_COLORS" for t in node.targets)
        if is_import or is_plot or is_palette:
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(SCRIPT), "exec"), namespace)
    main = source.split("# Main result: CausalImpact\n", 1)[1].split("# Robustness test 2 CausalImpact:", 1)[0]
    sensitivity = source.split("# Sensitivity analysis: Run CausalImpact without auxiliary series\n", 1)[1].split("# Appendix:", 1)[0]
    with tempfile.TemporaryDirectory() as folder:
        original_cwd = Path.cwd()
        try:
            os.chdir(folder)
            Path("images").mkdir()
            exec(compile(main, str(SCRIPT), "exec"), namespace)
            exec(compile(sensitivity, str(SCRIPT), "exec"), namespace)
        finally:
            os.chdir(original_cwd)
    return namespace


class Chapter11CausalImpactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.import_error = None
        try:
            cls.examples = run_examples()
        except ImportError as error:
            cls.import_error = error

    def setUp(self):
        if self.import_error is not None:
            self.fail("The student script still requests the legacy CausalImpact class; use the published tfp-causalimpact API: " + str(self.import_error))

    def test_main_and_sensitivity_recover_the_planted_effect(self):
        for name in ("impact_ci", "impact_no_aux"):
            with self.subTest(name=name):
                result = self.examples[name]
                average = result.summary.loc["average"]
                self.assertAlmostEqual(average.abs_effect, 20, delta=1)
                self.assertGreater(average.abs_effect_lower, 0)
                self.assertLess(average.p_value, 0.05)
                self.assertAlmostEqual(result.summary.loc["cumulative", "abs_effect"], 36 * average.abs_effect, delta=1e-3)
                self.assertIsNotNone(result.posterior_samples.seasonal_levels, "Weekly seasonality must be fitted, including without auxiliary data")
                self.assertEqual(len(result.posterior_samples.level), 4500)

    def test_placebo_has_no_detected_effect_and_correct_dates(self):
        result = self.examples["impact_placebo_ci"]
        average = result.summary.loc["average"]
        self.assertLess(abs(average.abs_effect), 1)
        self.assertLess(average.abs_effect_lower, 0)
        self.assertGreater(average.abs_effect_upper, 0)
        self.assertGreater(average.p_value, 0.05)
        self.assertIsNotNone(result.posterior_samples.seasonal_levels)
        self.assertEqual(result.series["post_period_start"].unique().tolist(), [59])
        self.assertEqual(result.series["post_period_end"].unique().tolist(), [79])
        self.assertAlmostEqual(result.summary.loc["cumulative", "abs_effect"], 21 * average.abs_effect, delta=1e-3)

    def test_prediction_intervals_remain_chronological_and_finite(self):
        result = self.examples["impact_ci"]
        self.assertEqual(result.series.index.tolist(), list(range(120)))
        self.assertEqual(result.series["post_period_start"].unique().tolist(), [84])
        forecast = result.series.loc[84:119, ["posterior_mean", "posterior_lower", "posterior_upper"]]
        self.assertTrue(np.isfinite(forecast.to_numpy()).all())
        self.assertTrue((forecast.posterior_lower < forecast.posterior_mean).all())
        self.assertTrue((forecast.posterior_mean < forecast.posterior_upper).all())
        self.assertLess((forecast.posterior_upper - forecast.posterior_lower).max(), 30)

    def test_static_plot_uses_native_point_and_cumulative_intervals(self):
        result = self.examples["impact_ci"]
        figure = self.examples["plot_causalimpact_result"](result, self.examples["data"]["date"], 84)
        try:
            self.assertEqual(len(figure.axes), 3)
            for axis, prefix in zip(figure.axes[1:], ("point_effects", "cumulative_effects")):
                np.testing.assert_allclose(axis.lines[0].get_ydata(), result.series[prefix + "_mean"])
                vertices = axis.collections[0].get_paths()[0].vertices[:, 1]
                self.assertAlmostEqual(vertices.min(), result.series[prefix + "_lower"].min())
                self.assertAlmostEqual(vertices.max(), result.series[prefix + "_upper"].max())
        finally:
            plt.close(figure)

    def test_post_outcomes_do_not_change_counterfactual_fitting(self):
        changed = dict(self.examples)
        changed["ci_data"] = self.examples["ci_data"].copy()
        changed["ci_data"].loc[84:, "y"] += 100
        fit = next(node for node in ast.parse(SCRIPT.read_text()).body
                   if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "impact_ci" for t in node.targets))
        exec(compile(ast.Module(body=[fit], type_ignores=[]), str(SCRIPT), "exec"), changed)
        fields = ["posterior_mean", "posterior_lower", "posterior_upper"]
        np.testing.assert_allclose(changed["impact_ci"].series[fields], self.examples["impact_ci"].series[fields], atol=1e-5, rtol=0)
        self.assertAlmostEqual(changed["impact_ci"].summary.loc["average", "abs_effect"] - self.examples["impact_ci"].summary.loc["average", "abs_effect"], 100, delta=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
