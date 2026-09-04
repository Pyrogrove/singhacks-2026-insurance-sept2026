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

    def _run_controlled_failure(self, code: str, message: str) -> AppTest:
        controlled_failure = {
            "status": "unavailable" if code.startswith("MISSING_") else "failed",
            "model": "deepseek-v4-flash-internal",
            "latency_seconds": None,
            "validation_passed": False,
            "structural_validation_passed": False,
            "semantic_validation_passed": False,
            "deterministic_evidence": {"sentinel": "preserved"},
            "model_synthesis": None,
            "error": {"code": code, "message": message},
        }

        with patch(
            "priscilla.synthesis.synthesize_evidence",
            return_value=controlled_failure,
        ) as synthesis:
            app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
            self.assertEqual([], app.exception)
            synthesis.assert_not_called()
            app.button(key="generate_ai_briefing").click().run(timeout=30)
            self.assertEqual([], app.exception)
            synthesis.assert_called_once()
        return app

    @staticmethod
    def _visible_text(app: AppTest) -> str:
        collections = (
            app.caption,
            app.info,
            app.markdown,
            app.subheader,
            app.success,
            app.warning,
        )
        return "\n".join(
            str(element.value)
            for collection in collections
            for element in collection
        )

    def _assert_deterministic_evidence_without_synthesis(self, app: AppTest) -> None:
        self.assertIn("HKD 206.88m", [metric.value for metric in app.metric])
        self.assertNotIn(
            "EVIDENCE-GROUNDED RM BRIEFING",
            [element.value for element in app.subheader],
        )

    def test_missing_api_key_is_provider_neutral_and_preserves_evidence(self) -> None:
        app = self._run_controlled_failure(
            "MISSING_API_KEY", "DEEPSEEK_API_KEY is not configured"
        )
        self._assert_deterministic_evidence_without_synthesis(app)
        visible_text = self._visible_text(app)
        self.assertIn(
            "AI briefing service is not configured in this environment.",
            visible_text,
        )
        self.assertNotIn("deepseek", visible_text.casefold())
        self.assertNotIn("DEEPSEEK_API_KEY", visible_text)

    def test_http_failure_is_provider_neutral_and_preserves_evidence(self) -> None:
        app = self._run_controlled_failure(
            "SYNTHESIS_FAILED", "DeepSeek HTTP error 503"
        )
        self._assert_deterministic_evidence_without_synthesis(app)
        visible_text = self._visible_text(app)
        self.assertIn(
            "AI briefing service is temporarily unavailable.", visible_text
        )
        self.assertNotIn("deepseek", visible_text.casefold())
        self.assertNotIn("503", visible_text)

    def test_validation_failures_render_safe_bounded_messages(self) -> None:
        cases = {
            "STRUCTURAL_VALIDATION_FAILED": (
                "DeepSeek returned malformed JSON",
                "AI briefing response did not pass structural validation.",
            ),
            "SEMANTIC_VALIDATION_FAILED": (
                "Prohibited semantic condition in headline",
                "AI briefing response did not pass evidence-safety validation.",
            ),
        }
        for code, (raw_message, expected_message) in cases.items():
            with self.subTest(code=code):
                app = self._run_controlled_failure(code, raw_message)
                self._assert_deterministic_evidence_without_synthesis(app)
                visible_text = self._visible_text(app)
                self.assertIn(expected_message, visible_text)
                self.assertNotIn(raw_message, visible_text)
                self.assertNotIn("deepseek", visible_text.casefold())


if __name__ == "__main__":
    unittest.main()
