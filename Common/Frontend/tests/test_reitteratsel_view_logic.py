from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[3]
FRONTEND_DIR = ROOT_DIR / "Common" / "Frontend"
KG_DIR = ROOT_DIR / "Common" / "Micro" / "5_Model_KG"
for path in (FRONTEND_DIR, KG_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from reitteratsel_core import compute_final_distress_score, compute_sora_distress_score
from reitteratsel_view_logic import (
    build_macro_panel_context,
    build_ranking_view,
    get_label_row_for_period,
    get_latest_car_path_row_for_period,
    get_metric_value_for_period,
    resolve_macro_row,
)


class ReitteratselViewLogicTests(unittest.TestCase):
    def setUp(self) -> None:
        self.macro_df = pd.DataFrame(
            {
                "snapshot_ts": pd.to_datetime(["2025-01-15", "2025-02-15", "2025-03-15"]),
                "y_pred": [0.10, -0.20, 0.30],
                "predicted_level": [3.1, 2.8, 3.3],
                "fomc_decision_date": pd.to_datetime(["2025-01-29", "2025-02-26", "2025-03-19"]),
            }
        )
        self.fuzzy_df = pd.DataFrame(
            {
                "ticker": ["AAA", "AAA", "BBB", "BBB"],
                "period_id": [1, 2, 3, 4],
                "reit_name": ["Alpha", "Alpha", "Beta", "Beta"],
                "sector": ["Retail", "Retail", "Office", "Office"],
                "fiscal_year": [2023, 2024, 2023, 2024],
                "fiscal_year_end_date": pd.to_datetime(["2024-03-31", "2025-03-31", "2024-06-30", "2025-06-30"]),
                "distress_score_mamdani": [0.20, 0.90, 0.30, 0.80],
                "distress_level": ["stable", "critical", "stable", "high"],
                "null_count": [0, 0, 1, 1],
                "non_ok_count": [0, 0, 1, 1],
                "top_rule_ids": [None, None, None, None],
                "rule_trace_text": [None, None, None, None],
                "car_63wd": [0.01, 0.02, 0.03, 0.04],
                "car_126wd": [0.05, -0.10, 0.02, -0.08],
                "label_126wd": ["HEALTHY", "DISTRESSED", "WATCH", "DISTRESSED"],
            }
        )
        self.metric_df = pd.DataFrame(
            {
                "ticker": ["AAA", "AAA", "BBB", "BBB"],
                "period_id": [1, 2, 3, 4],
                "fiscal_year": [2023, 2024, 2023, 2024],
                "fiscal_year_end_date": pd.to_datetime(["2024-03-31", "2025-03-31", "2024-06-30", "2025-06-30"]),
                "metric_code": ["REFI_RISK", "REFI_RISK", "REFI_RISK", "REFI_RISK"],
                "metric_value": [0.10, 0.80, 0.20, 0.90],
                "calc_status": ["OK", "OK", "OK", "OK"],
            }
        )
        self.label_df = pd.DataFrame(
            {
                "ticker": ["AAA", "AAA"],
                "period_id": [1, 2],
                "fiscal_year": [2023, 2024],
                "anchor_date": pd.to_datetime(["2024-03-31", "2025-03-31"]),
                "anchor_trade_date": pd.to_datetime(["2024-04-01", "2025-04-01"]),
                "car_63wd": [0.01, -0.02],
                "car_126wd": [0.05, -0.10],
                "label_126wd": ["HEALTHY", "DISTRESSED"],
                "null_count": [0, 0],
                "non_ok_count": [0, 0],
                "notes": [None, None],
            }
        )
        self.car_path_df = pd.DataFrame(
            {
                "ticker": ["AAA", "AAA", "AAA", "BBB", "BBB"],
                "period_id": [1, 1, 2, 3, 4],
                "trade_date": pd.to_datetime(["2024-04-01", "2025-02-10", "2025-04-01", "2024-07-01", "2025-07-01"]),
                "accum_car_to_date": [0.00, -0.12, 0.00, -0.04, 0.00],
                "car_path_distress": [0.50, 0.85, 0.50, 0.45, 0.50],
            }
        )

    def test_resolve_macro_row_uses_selected_date_not_latest_overall(self) -> None:
        resolved = resolve_macro_row(self.macro_df, "2025-02-20")
        self.assertEqual(pd.Timestamp("2025-02-15"), resolved["snapshot_ts"])

    def test_build_ranking_view_resolves_annual_period_by_selected_date(self) -> None:
        ranking_view, macro_row, distress_sora = build_ranking_view(
            self.fuzzy_df,
            self.metric_df,
            self.macro_df,
            self.car_path_df,
            "2025-02-20",
        )
        aaa_row = ranking_view.loc[ranking_view["ticker"] == "AAA"].iloc[0]
        bbb_row = ranking_view.loc[ranking_view["ticker"] == "BBB"].iloc[0]
        self.assertEqual(1, int(aaa_row["period_id"]))
        self.assertEqual(3, int(bbb_row["period_id"]))
        self.assertEqual(pd.Timestamp("2025-02-15"), macro_row["snapshot_ts"])

    def test_build_ranking_view_derives_distress_sora_from_resolved_macro_row(self) -> None:
        _, macro_row, distress_sora = build_ranking_view(
            self.fuzzy_df,
            self.metric_df,
            self.macro_df,
            self.car_path_df,
            "2025-02-20",
        )
        self.assertEqual(pd.Timestamp("2025-02-15"), macro_row["snapshot_ts"])
        self.assertAlmostEqual(compute_sora_distress_score(-0.20), distress_sora)

    def test_build_ranking_view_uses_same_period_refi_not_latest_refi(self) -> None:
        ranking_view, _, distress_sora = build_ranking_view(
            self.fuzzy_df,
            self.metric_df,
            self.macro_df,
            self.car_path_df,
            "2025-02-20",
        )
        aaa_row = ranking_view.loc[ranking_view["ticker"] == "AAA"].iloc[0]
        self.assertAlmostEqual(0.10, float(aaa_row["refi_risk"]))
        expected = compute_final_distress_score(0.20, distress_sora, 0.10, 0.85)
        self.assertAlmostEqual(expected, float(aaa_row["final_distress"]))
        self.assertAlmostEqual(-0.12, float(aaa_row["accum_car_to_date"]))
        self.assertAlmostEqual(0.85, float(aaa_row["car_path_distress"]))

    def test_get_metric_value_for_period_uses_exact_period(self) -> None:
        value = get_metric_value_for_period(
            self.metric_df,
            ticker="AAA",
            period_id=1,
            metric_code="REFI_RISK",
        )
        self.assertAlmostEqual(0.10, value)

    def test_get_label_row_for_period_uses_exact_period(self) -> None:
        row = get_label_row_for_period(self.label_df, ticker="AAA", period_id=1)
        self.assertEqual("HEALTHY", row["label_126wd"])
        self.assertAlmostEqual(0.05, float(row["car_126wd"]))

    def test_get_latest_car_path_row_for_period_uses_selected_date(self) -> None:
        row = get_latest_car_path_row_for_period(
            self.car_path_df,
            ticker="AAA",
            period_id=1,
            selected_date="2025-02-20",
        )
        assert row is not None
        self.assertEqual(pd.Timestamp("2025-02-10"), pd.Timestamp(row["trade_date"]))
        self.assertAlmostEqual(0.85, float(row["car_path_distress"]))

    def test_build_macro_panel_context_uses_resolved_macro_row_not_latest_overall(self) -> None:
        macro_row = resolve_macro_row(self.macro_df, "2025-02-20")
        distress_sora = compute_sora_distress_score(float(macro_row["y_pred"]))
        context = build_macro_panel_context(macro_row, distress_sora)
        self.assertEqual(pd.Timestamp("2025-02-15"), context["snapshot_ts"])
        self.assertEqual(pd.Timestamp("2025-02-26"), context["fomc_decision_date"])
        self.assertAlmostEqual(-0.20, context["predicted_change"])
        self.assertAlmostEqual(2.8, context["predicted_level"])
        self.assertAlmostEqual(compute_sora_distress_score(-0.20), context["distress_sora"])


if __name__ == "__main__":
    unittest.main()
