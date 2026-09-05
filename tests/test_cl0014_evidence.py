"""Locked expectations for the deterministic CL-0014 evidence packet."""

from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

from priscilla.evidence import (
    REQUIRED_CSV_FILES,
    REQUIRED_JSON_FILES,
    build_client_evidence,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
OFFICIAL_PATHS = (
    *(DATA_DIR / name for name in REQUIRED_CSV_FILES + REQUIRED_JSON_FILES),
    REPO_ROOT / ".gitignore",
    REPO_ROOT / "README.md",
    REPO_ROOT / "requirements.txt",
    REPO_ROOT / "docs" / "DATA_DICTIONARY.md",
    REPO_ROOT / "starter" / "quickstart.py",
)


def _hash_files(paths: tuple[Path, ...]) -> dict[Path, str]:
    return {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


class CL0014EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.before_hashes = _hash_files(OFFICIAL_PATHS)
        cls.packet = build_client_evidence(DATA_DIR)

    @classmethod
    def tearDownClass(cls) -> None:
        after_hashes = _hash_files(OFFICIAL_PATHS)
        if cls.before_hashes != after_hashes:
            raise AssertionError("Evidence analysis modified an official source file")

    def test_all_required_official_data_files_load(self) -> None:
        self.assertEqual(17, len(OFFICIAL_PATHS))
        self.assertTrue(all(path.is_file() for path in OFFICIAL_PATHS))

    def test_cl0014_exists(self) -> None:
        client = self.packet["source_facts"]["client"]
        self.assertEqual("CL-0014", client["client_id"])
        self.assertEqual("Lau Chi Ming", client["client_name"])

    def test_current_portfolio_total(self) -> None:
        current = self.packet["calculated_results"]["current_snapshot"]
        self.assertEqual(206_878_860.0, current["total_portfolio_market_value"])

    def test_current_property_linked_exposure(self) -> None:
        current = self.packet["calculated_results"]["current_snapshot"]
        self.assertEqual(101_431_480.0, current["property_linked_market_value"])
        component_ids = {
            item["instrument_id"] for item in current["property_linked_components"]
        }
        self.assertIn("SYN-SP-0503", component_ids)

    def test_current_property_percentage(self) -> None:
        current = self.packet["calculated_results"]["current_snapshot"]
        self.assertEqual(49.03, round(current["property_linked_percentage"], 2))

    def test_complete_property_percentage_series(self) -> None:
        series = self.packet["calculated_results"]["portfolio_snapshot_series"]
        actual = {
            item["snapshot_date"]: round(item["property_linked_percentage"], 2)
            for item in series
        }
        self.assertEqual(
            {
                "2025-12-31": 48.54,
                "2026-02-27": 53.45,
                "2026-03-31": 51.92,
                "2026-06-30": 49.96,
                "2026-08-26": 49.03,
            },
            actual,
        )

    def test_current_facility_metrics(self) -> None:
        facility = self.packet["source_facts"]["credit_facility"]
        self.assertEqual(69.41, facility["current_ltv_percentage"])
        self.assertEqual(70.0, facility["margin_call_trigger_percentage"])
        self.assertAlmostEqual(
            0.59,
            self.packet["calculated_results"][
                "facility_ltv_distance_to_trigger_percentage_points"
            ],
            places=10,
        )
        self.assertEqual(
            [53.93, 53.53, 65.62, 67.96, 69.41],
            [item["ltv_percentage"] for item in facility["ltv_series"]],
        )

    def test_confirmed_cash_requirement(self) -> None:
        needs = self.packet["source_facts"]["confirmed_cash_needs"]
        self.assertEqual(1, len(needs))
        self.assertEqual(60_000_000, needs[0]["amount"])
        self.assertEqual("Confirmed", needs[0]["certainty"])

    def test_current_daily_liquidity(self) -> None:
        current = self.packet["calculated_results"]["current_snapshot"]
        self.assertEqual(
            88_847_450.0,
            current["daily_liquidity_gross_market_value"],
        )
        self.assertEqual(
            "UNRESOLVED",
            self.packet["calculated_results"]["funding_suitability"],
        )

    def test_rm_notes_are_loaded_verbatim_from_source(self) -> None:
        notes = self.packet["source_facts"]["rm_notes"]
        self.assertEqual(["N-018", "N-019"], [note["note_id"] for note in notes])
        self.assertIn("Drew a further HKD 4m", notes[0]["source_note_text"])
        self.assertIn("Redevelopment project needs", notes[1]["source_note_text"])

    def test_facility_drawdown_tension(self) -> None:
        tension = self.packet["data_tensions"][0]
        self.assertEqual("UNRESOLVED_DATA_TENSION", tension["label"])
        self.assertEqual(6_000_000.0, tension["facility_balance_increase"])
        self.assertEqual(4_000_000.0, tension["logged_facility_drawdown_transactions"])
        self.assertEqual(2_000_000.0, tension["unreconciled_difference"])
        self.assertIsNone(tension["explanation"])

    def test_analysis_does_not_modify_official_sources(self) -> None:
        before = _hash_files(OFFICIAL_PATHS)
        build_client_evidence(DATA_DIR)
        self.assertEqual(before, _hash_files(OFFICIAL_PATHS))


if __name__ == "__main__":
    unittest.main()
