"""Lightweight tests for the Streamlit workbench boundary."""

from __future__ import annotations

import py_compile
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest


REPO_ROOT = Path(__file__).resolve().parent.parent
APP_PATH = REPO_ROOT / "streamlit_app.py"


class StreamlitWorkbenchTests(unittest.TestCase):
    def test_app_compiles(self) -> None:
        py_compile.compile(str(APP_PATH), doraise=True)

    def test_evidence_loads_without_credentials_and_ai_is_click_gated(self) -> None:
        controlled_failure = {
            "status": "unavailable",
            "model": None,
            "latency_seconds": None,
            "validation_passed": False,
            "structural_validation_passed": False,
            "semantic_validation_passed": False,
            "deterministic_evidence": {"sentinel": "preserved"},
            "model_synthesis": None,
            "error": {
                "code": "MISSING_API_KEY",
                "message": "DEEPSEEK_API_KEY is not configured",
            },
        }

        with patch(
            "priscilla.synthesis.synthesize_evidence",
            return_value=controlled_failure,
        ) as synthesis:
            app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
            self.assertEqual([], app.exception)
            synthesis.assert_not_called()
            self.assertIn("HKD 206.88m", [metric.value for metric in app.metric])

            app.button(key="generate_ai_briefing").click().run(timeout=30)
            self.assertEqual([], app.exception)
            synthesis.assert_called_once()
            self.assertTrue(
                any(
                    element.value
                    == "AI briefing unavailable. Deterministic evidence remains available."
                    for element in app.warning
                )
            )


if __name__ == "__main__":
    unittest.main()
