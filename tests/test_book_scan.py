"""Focused coverage for deterministic RM book screening."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

from priscilla.book_scan import AS_OF_DATE, build_book_scan
from priscilla.evidence import build_client_evidence


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
APP_PATH = REPO_ROOT / "streamlit_app.py"


class OfficialBookScanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scan = build_book_scan(DATA_DIR)
        cls.by_client = {row["client_id"]: row for row in cls.scan["clients"]}

    def test_official_book_has_exactly_20_clients(self) -> None:
        self.assertEqual(20, self.scan["client_count"])
        self.assertEqual(20, len(self.scan["clients"]))

    def test_all_24_portfolio_relationships_are_represented(self) -> None:
        self.assertEqual(24, self.scan["portfolio_relationship_count"])
        self.assertEqual(24, len(self.scan["portfolio_mandate_evidence"]))
        self.assertEqual(
            24, sum(row["portfolio_count"] for row in self.scan["clients"])
        )

    def test_multi_portfolio_clients_are_preserved(self) -> None:
        self.assertEqual(2, self.by_client["CL-0001"]["portfolio_count"])
        self.assertEqual(2, self.by_client["CL-0002"]["portfolio_count"])
        self.assertEqual(3, self.by_client["CL-0017"]["portfolio_count"])

    def test_cl0014_deep_evidence_path_still_returns_locked_result(self) -> None:
        packet = build_client_evidence(DATA_DIR)
        self.assertEqual("CL-0014", packet["source_facts"]["client"]["client_id"])
        self.assertEqual(
            206_878_860.0,
            packet["calculated_results"]["current_snapshot"][
                "total_portfolio_market_value"
            ],
        )

    def test_existing_cl0014_tabs_and_new_book_view_are_reachable(self) -> None:
        app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
        self.assertEqual([], app.exception)
        self.assertEqual(
            ["Overview", "Evidence", "RM notes", "AI briefing"],
            [tab.label for tab in app.tabs],
        )
        visible_subheaders = [item.value for item in app.subheader]
        for heading in (
            "Why now",
            "Deterministic evidence",
            "RM source context",
            "AI briefing",
            "RM Book",
            "Deep investigation — CL-0014 Lau Chi Ming",
        ):
            self.assertIn(heading, visible_subheaders)
        self.assertEqual(
            "Generate AI RM Briefing",
            app.button(key="generate_ai_briefing").label,
        )

    def test_review_queue_is_flag_derived_and_book_first(self) -> None:
        app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
        self.assertEqual([], app.exception)
        source = APP_PATH.read_text(encoding="utf-8")
        self.assertLess(
            source.index('st.subheader("RM Book")'),
            source.index("**Funding pressure meets concentrated property exposure**"),
        )
        self.assertLess(
            source.index('st.markdown("#### Full RM Book")'),
            source.index("overview_tab, evidence_tab, notes_tab, ai_tab = st.tabs"),
        )
        queue_columns = [
            "Client ID / name",
            "Review signals",
            "Why surfaced",
            "Deep investigation",
        ]
        full_book_columns = [
            "Client ID",
            "Client name",
            "Portfolio count",
            "Mandate code(s)",
            "Confirmed cash need",
            "Credit facility",
            "Commitment",
            "Mandate allocation deviation",
            "Evidence flags",
        ]
        frames = [element.value for element in app.dataframe]
        queue = next(frame for frame in frames if list(frame.columns) == queue_columns)
        full_book = next(
            frame for frame in frames if list(frame.columns) == full_book_columns
        )

        expected_rows = sorted(
            (
                row
                for row in self.scan["clients"]
                if row["evidence_flag_count"] > 0
            ),
            key=lambda row: (-row["evidence_flag_count"], row["client_id"]),
        )
        flag_labels = (
            ("confirmed_cash_need_present", "Cash need"),
            ("credit_facility_present", "Credit facility"),
            ("commitment_present", "Commitment"),
            ("mandate_allocation_deviation_present", "Mandate deviation"),
        )
        expected_reasons = [
            ", ".join(
                label for key, label in flag_labels if row[key] is True
            )
            for row in expected_rows
        ]

        self.assertEqual(
            [
                f"{row['client_id']} — {row['client_name']}"
                for row in expected_rows
            ],
            queue["Client ID / name"].tolist(),
        )
        self.assertEqual(
            [row["evidence_flag_count"] for row in expected_rows],
            queue["Review signals"].tolist(),
        )
        self.assertEqual(expected_reasons, queue["Why surfaced"].tolist())
        investigation_values = queue.set_index("Client ID / name")[
            "Deep investigation"
        ]
        self.assertEqual("Available", investigation_values["CL-0014 — Lau Chi Ming"])
        self.assertEqual(
            {"—"},
            set(investigation_values.drop("CL-0014 — Lau Chi Ming").tolist()),
        )
        self.assertEqual(20, len(full_book))
        self.assertEqual(24, int(full_book["Portfolio count"].sum()))
        self.assertIn(
            "Ordered by number of independent deterministic review signals. "
            "Signal count indicates breadth of evidence, not risk severity or "
            "investment priority.",
            [item.value for item in app.caption],
        )


class MissingEvidenceTests(unittest.TestCase):
    def test_missing_optional_sources_produce_unknown_not_false(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "clients.csv").write_text(
                "client_id,client_name\nCL-X,Test Client\n", encoding="utf-8"
            )
            (root / "portfolios.csv").write_text(
                "portfolio_id,client_id,mandate_code\nPF-X,CL-X,BAL\n",
                encoding="utf-8",
            )
            scan = build_book_scan(root)

        row = scan["clients"][0]
        self.assertIsNone(row["confirmed_cash_need_present"])
        self.assertIsNone(row["credit_facility_present"])
        self.assertIsNone(row["commitment_present"])
        self.assertIsNone(row["mandate_allocation_deviation_present"])
        self.assertEqual(0, row["evidence_flag_count"])


class MandateComparisonTests(unittest.TestCase):
    def test_only_supplied_min_max_ranges_drive_deviation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "clients.csv").write_text(
                "client_id,client_name\nCL-X,Test Client\n", encoding="utf-8"
            )
            (root / "portfolios.csv").write_text(
                "portfolio_id,client_id,mandate_code\nPF-X,CL-X,TEST\n",
                encoding="utf-8",
            )
            (root / "holdings.csv").write_text(
                "snapshot_date,portfolio_id,asset_class,market_value_base\n"
                f"{AS_OF_DATE},PF-X,Equity,80\n"
                f"{AS_OF_DATE},PF-X,Cash and Equivalents,20\n",
                encoding="utf-8",
            )
            (root / "mandates.csv").write_text(
                "mandate_code,asset_class,min_pct,max_pct\n"
                "TEST,Equity,0,70\n"
                "TEST,Cash and Equivalents,0,30\n"
                "TEST,Fixed Income,10,50\n",
                encoding="utf-8",
            )
            scan = build_book_scan(root)

        client = scan["clients"][0]
        evidence = scan["portfolio_mandate_evidence"][0]
        equity = next(
            item for item in evidence["comparisons"] if item["asset_class"] == "Equity"
        )
        fixed_income = next(
            item
            for item in evidence["comparisons"]
            if item["asset_class"] == "Fixed Income"
        )
        self.assertTrue(client["mandate_allocation_deviation_present"])
        self.assertEqual(80.0, equity["actual_pct"])
        self.assertEqual((0.0, 70.0), (equity["min_pct"], equity["max_pct"]))
        self.assertTrue(equity["outside_range"])
        self.assertEqual(0.0, fixed_income["actual_pct"])
        self.assertTrue(fixed_income["outside_range"])
        self.assertEqual(
            ["holdings.csv", "portfolios.csv", "mandates.csv"],
            scan["provenance"]["mandate_sources"],
        )


if __name__ == "__main__":
    unittest.main()
