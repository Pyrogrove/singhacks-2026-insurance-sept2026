"""Lightweight tests for the Streamlit workbench boundary."""

from __future__ import annotations

import py_compile
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest


REPO_ROOT = Path(__file__).resolve().parent.parent
APP_PATH = REPO_ROOT / "streamlit_app.py"
ENGLISH_DISCLAIMER = (
    "The Relationship Manager remains responsible for advice and action."
)
ENGLISH_AUTHORITY = (
    "Priscilla supports evidence-led investigation; it does not provide autonomous "
    "advice or execute trades."
)


def _successful_english_synthesis() -> dict:
    synthesis = {
        "headline": "Validated English headline",
        "why_it_matters": "Validated English explanation.",
        "evidence_used": [
            "Property-linked exposure is 49.03%.",
            "Facility LTV is 69.41%.",
        ],
        "uncertainties": [
            "The HKD2m difference remains unexplained.",
            "The source does not establish funding suitability.",
        ],
        "rm_questions": [
            "What should the RM clarify?",
            "What funding sources are available?",
        ],
        "rm_review_options": [
            "Review the evidence with the client.",
            "Reconcile the HKD2m difference.",
            "Review facility terms and timing.",
        ],
        "disclaimer": "The Relationship Manager remains responsible for advice and action.",
    }
    return {
        "status": "available",
        "model": "internal-provider-model",
        "latency_seconds": 1.0,
        "validation_passed": True,
        "structural_validation_passed": True,
        "semantic_validation_passed": True,
        "deterministic_evidence": {"sentinel": "preserved"},
        "model_synthesis": synthesis,
        "error": None,
    }


def _successful_translation_result() -> dict:
    return {
        "status": "available",
        "model": "internal-provider-model",
        "latency_seconds": 1.0,
        "english_synthesis": _successful_english_synthesis()["model_synthesis"],
        "translation": {
            "headline": "已驗證的英文標題",
            "why_it_matters": "已驗證的英文說明。",
            "evidence_used": [
                "物業相關風險承擔為 49.03%。",
                "融資 LTV 為 69.41%。",
            ],
            "uncertainties": [
                "HKD2m 的差異仍未解釋。",
                "資料來源並未確立資金適合性。",
            ],
            "rm_questions": [
                "RM 應澄清甚麼？",
                "有哪些可用資金來源？",
            ],
            "rm_review_options": [
                "與客戶檢視證據。",
                "核對 HKD2m 的差異。",
                "檢視融資條款及時間。",
            ],
        },
        "error": None,
    }


def _failed_translation_result() -> dict:
    english_synthesis = _successful_english_synthesis()["model_synthesis"]
    return {
        "status": "failed",
        "model": "deepseek-provider-model-id",
        "latency_seconds": 1.0,
        "english_synthesis": english_synthesis,
        "translation": None,
        "error": {
            "code": "TRANSLATION_FAILED",
            "message": "DeepSeek HTTP error 503: DEEPSEEK_API_KEY",
        },
    }


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
            "RM INTELLIGENCE BRIEFING",
            [element.value for element in app.subheader],
        )

    def test_rm_authority_is_visible_before_ai_generation(self) -> None:
        app = AppTest.from_file(str(APP_PATH)).run(timeout=30)

        self.assertEqual([], app.exception)
        visible_text = self._visible_text(app)
        self.assertEqual(1, visible_text.count(ENGLISH_DISCLAIMER))
        self.assertEqual(1, visible_text.count(ENGLISH_AUTHORITY))

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

    def test_english_success_is_one_complete_rm_decision_brief(self) -> None:
        english_result = _successful_english_synthesis()
        original_synthesis = _successful_english_synthesis()["model_synthesis"]
        with (
            patch(
                "priscilla.synthesis.synthesize_evidence",
                return_value=english_result,
            ) as synthesis,
            patch(
                "priscilla.translation.translate_validated_synthesis",
            ) as translation,
        ):
            app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
            app.button(key="generate_ai_briefing").click().run(timeout=30)

        synthesis.assert_called_once()
        translation.assert_not_called()
        self.assertEqual([], app.segmented_control)
        self.assertNotIn(
            "generate_traditional_chinese_translation",
            [button.key for button in app.button],
        )
        self.assertEqual(original_synthesis, english_result["model_synthesis"])
        self.assertEqual([], app.exception)
        subheadings = [element.value for element in app.subheader]
        self.assertEqual(1, subheadings.count("RM INTELLIGENCE BRIEFING"))
        visible_text = self._visible_text(app)
        self.assertNotIn("EVIDENCE-GROUNDED RM BRIEFING", visible_text)
        self.assertNotIn("AI SYNTHESIS]", visible_text)
        expected_context = {
            "WHY NOW": (
                "The facility is only 0.59 percentage points from its 70.00% "
                "trigger (current LTV: 69.41%) while the confirmed HKD60m "
                "Mid-Levels redevelopment funding requirement approaches in the "
                "Nov 2026–Jun 2027 window."
            ),
            "WHY THIS CLIENT": (
                "This is client-specific: 49.03% of portfolio value is "
                "property-linked, liquidity needs are recorded as High, and the "
                "client expects to fund part of the HKD60m requirement from the "
                "portfolio. The RM note records that he was surprised how little "
                "was liquid."
            ),
            "RISK IF UNRESOLVED": (
                "If the funding plan remains unresolved and lending values weaken, "
                "the RM may have less flexibility to address the HKD60m requirement "
                "before or during the Nov 2026–Jun 2027 funding window. The existing "
                "HKD2m facility discrepancy should be reconciled before relying on "
                "the facility records for the client discussion."
            ),
        }
        for label, body in expected_context.items():
            self.assertIn(label, visible_text)
            self.assertIn(body, visible_text)
        captions = [element.value for element in app.caption]
        self.assertLess(
            captions.index("WHY NOW"),
            captions.index("WHAT NEEDS ATTENTION"),
        )
        for prohibited_claim in (
            "margin call will occur",
            "forced liquidation will occur",
            "client cannot fund the requirement",
            "facility will breach",
            "portfolio must be sold",
            "loss will occur",
        ):
            self.assertNotIn(prohibited_claim, visible_text.casefold())
        for field in (
            "headline",
            "why_it_matters",
            "uncertainties",
            "rm_questions",
            "rm_review_options",
            "evidence_used",
        ):
            value = english_result["model_synthesis"][field]
            for item in value if isinstance(value, list) else (value,):
                self.assertIn(item, visible_text)
        self.assertIn("WHAT NEEDS ATTENTION", visible_text)
        self.assertIn("WHY IT MATTERS", visible_text)
        self.assertIn("WHAT WE DON'T KNOW", visible_text)
        self.assertIn("WHAT PRISCILLA SHOULD ASK", visible_text)
        self.assertIn("REVIEW OPTIONS", visible_text)
        self.assertIn(
            "Evidence supporting this briefing",
            [element.label for element in app.expander],
        )
        english_authority = (
            "The Relationship Manager remains responsible for advice and action."
        )
        supporting_authority = (
            "Priscilla supports evidence-led investigation; it does not provide "
            "autonomous advice or execute trades."
        )
        self.assertEqual(2, visible_text.count(english_authority))
        self.assertEqual(2, visible_text.count(supporting_authority))
        self.assertNotIn("客戶經理仍對建議及後續行動負責。", visible_text)

    @unittest.skip("Traditional Chinese UI is intentionally hidden in the public demo")
    def test_translation_is_explicit_once_and_reused_across_language_switches(
        self,
    ) -> None:
        english_result = _successful_english_synthesis()
        translation_result = _successful_translation_result()
        with (
            patch(
                "priscilla.synthesis.synthesize_evidence",
                return_value=english_result,
            ) as synthesis,
            patch(
                "priscilla.translation.translate_validated_synthesis",
                return_value=translation_result,
            ) as translation,
        ):
            app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
            synthesis.assert_not_called()
            translation.assert_not_called()

            app.button(key="generate_ai_briefing").click().run(timeout=30)
            synthesis.assert_called_once()
            translation.assert_not_called()
            self.assertIn("Validated English headline", self._visible_text(app))

            app.segmented_control(key="ai_briefing_language").select(
                "繁體中文"
            ).run(timeout=30)
            translation.assert_not_called()
            self.assertEqual(
                "產生繁體中文版本",
                app.button(key="generate_traditional_chinese_translation").label,
            )

            app.button(key="generate_traditional_chinese_translation").click().run(
                timeout=30
            )
            translation.assert_called_once_with(english_result["model_synthesis"])
            cached_translation = app.session_state["cl0014_translation_result"]
            chinese_text = self._visible_text(app)
            for label in (
                "為何是現在",
                "為何是這位客戶",
                "若仍未解決的風險",
            ):
                self.assertIn(label, chinese_text)
            for protected_value in (
                "69.41%",
                "70.00%",
                "0.59",
                "HKD60m",
                "49.03%",
                "HKD2m",
                "Nov 2026",
                "Jun 2027",
            ):
                self.assertIn(protected_value, chinese_text)
            for field in (
                "headline",
                "why_it_matters",
                "uncertainties",
                "rm_questions",
                "rm_review_options",
                "evidence_used",
            ):
                value = translation_result["translation"][field]
                for item in value if isinstance(value, list) else (value,):
                    self.assertIn(item, chinese_text)
            for label in (
                "客戶經理情報簡報",
                "AI 協助 · 以驗證英文版本為依據",
                "需由客戶經理檢視",
                "需要關注",
                "為何重要",
                "仍待確認",
                "PRISCILLA 應確認",
                "可檢視選項",
            ):
                self.assertIn(label, chinese_text)
            self.assertIn(
                "支持此簡報的證據",
                [element.label for element in app.expander],
            )
            chinese_disclaimer = "客戶經理仍對建議及後續行動負責。"
            chinese_authority = (
                "Priscilla 支援以證據為本的調查；不提供自主投資建議或執行交易。"
            )
            self.assertEqual(1, chinese_text.count(chinese_disclaimer))
            self.assertEqual(1, chinese_text.count(chinese_authority))
            self.assertEqual(
                1,
                chinese_text.count(
                    "The Relationship Manager remains responsible for advice and action."
                ),
            )
            self.assertNotIn(
                "generate_traditional_chinese_translation",
                [button.key for button in app.button],
            )

            app.segmented_control(key="ai_briefing_language").select("English").run(
                timeout=30
            )
            app.segmented_control(key="ai_briefing_language").select(
                "繁體中文"
            ).run(timeout=30)
            translation.assert_called_once()
            self.assertEqual(
                cached_translation,
                app.session_state["cl0014_translation_result"],
            )
            self.assertIn("已驗證的英文標題", self._visible_text(app))

    @unittest.skip("Traditional Chinese UI is intentionally hidden in the public demo")
    def test_new_english_synthesis_clears_stored_translation(self) -> None:
        old_english_result = _successful_english_synthesis()
        new_english_result = _successful_english_synthesis()
        new_english_result["model_synthesis"]["headline"] = (
            "New validated English headline"
        )
        old_translation_result = _successful_translation_result()
        new_translation_result = _successful_translation_result()
        new_translation_result["english_synthesis"] = new_english_result[
            "model_synthesis"
        ]
        new_translation_result["translation"]["headline"] = "新的已驗證英文標題"

        with (
            patch(
                "priscilla.synthesis.synthesize_evidence",
                side_effect=(old_english_result, new_english_result),
            ) as synthesis,
            patch(
                "priscilla.translation.translate_validated_synthesis",
                side_effect=(old_translation_result, new_translation_result),
            ) as translation,
        ):
            app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
            app.button(key="generate_ai_briefing").click().run(timeout=30)
            app.segmented_control(key="ai_briefing_language").select(
                "繁體中文"
            ).run(timeout=30)
            app.button(key="generate_traditional_chinese_translation").click().run(
                timeout=30
            )
            self.assertIn("已驗證的英文標題", self._visible_text(app))

            app.button(key="generate_ai_briefing").click().run(timeout=30)
            self.assertEqual(2, synthesis.call_count)
            translation.assert_called_once()
            visible_text = self._visible_text(app)
            self.assertNotIn("已驗證的英文標題", visible_text)
            self.assertEqual(
                "產生繁體中文版本",
                app.button(key="generate_traditional_chinese_translation").label,
            )

            app.segmented_control(key="ai_briefing_language").select("English").run(
                timeout=30
            )
            self.assertIn("New validated English headline", self._visible_text(app))
            app.segmented_control(key="ai_briefing_language").select(
                "繁體中文"
            ).run(timeout=30)
            translation.assert_called_once()
            app.button(key="generate_traditional_chinese_translation").click().run(
                timeout=30
            )
            translation.assert_called_with(new_english_result["model_synthesis"])
            self.assertEqual(2, translation.call_count)
            self.assertIn("新的已驗證英文標題", self._visible_text(app))

    @unittest.skip("Traditional Chinese UI is intentionally hidden in the public demo")
    def test_translation_failure_is_neutral_and_keeps_english_visible(self) -> None:
        english_result = _successful_english_synthesis()
        failed_translation = _failed_translation_result()
        with (
            patch(
                "priscilla.synthesis.synthesize_evidence",
                return_value=english_result,
            ),
            patch(
                "priscilla.translation.translate_validated_synthesis",
                return_value=failed_translation,
            ) as translation,
        ):
            app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
            app.button(key="generate_ai_briefing").click().run(timeout=30)
            app.segmented_control(key="ai_briefing_language").select(
                "繁體中文"
            ).run(timeout=30)
            app.button(key="generate_traditional_chinese_translation").click().run(
                timeout=30
            )

        translation.assert_called_once()
        self.assertIn("HKD 206.88m", [metric.value for metric in app.metric])
        self.assertIn(
            "RM INTELLIGENCE BRIEFING",
            [element.value for element in app.subheader],
        )
        visible_text = self._visible_text(app)
        self.assertIn(
            "繁體中文翻譯暫時無法使用。英文驗證版本仍然可用。",
            visible_text,
        )
        self.assertIn("Validated English headline", visible_text)
        self.assertNotIn("已驗證的英文標題", visible_text)
        self.assertEqual(
            "重試繁體中文翻譯",
            app.button(key="generate_traditional_chinese_translation").label,
        )
        self.assertEqual(
            failed_translation,
            app.session_state["cl0014_translation_result"],
        )
        self.assertNotIn("deepseek", visible_text.casefold())
        self.assertNotIn("DEEPSEEK_API_KEY", visible_text)
        self.assertNotIn("503", visible_text)
        self.assertEqual(
            2,
            visible_text.count(
                "The Relationship Manager remains responsible for advice and action."
            ),
        )
        self.assertNotIn("客戶經理仍對建議及後續行動負責。", visible_text)

    @unittest.skip("Traditional Chinese UI is intentionally hidden in the public demo")
    def test_failed_translation_survives_switching_and_explicit_retry_succeeds(
        self,
    ) -> None:
        english_result = _successful_english_synthesis()
        failed_translation = _failed_translation_result()
        successful_translation = _successful_translation_result()
        with (
            patch(
                "priscilla.synthesis.synthesize_evidence",
                return_value=english_result,
            ),
            patch(
                "priscilla.translation.translate_validated_synthesis",
                side_effect=(failed_translation, successful_translation),
            ) as translation,
        ):
            app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
            app.button(key="generate_ai_briefing").click().run(timeout=30)
            app.segmented_control(key="ai_briefing_language").select(
                "繁體中文"
            ).run(timeout=30)
            translation.assert_not_called()

            app.button(key="generate_traditional_chinese_translation").click().run(
                timeout=30
            )
            self.assertEqual(1, translation.call_count)
            self.assertEqual(
                failed_translation,
                app.session_state["cl0014_translation_result"],
            )
            self.assertEqual(
                "重試繁體中文翻譯",
                app.button(key="generate_traditional_chinese_translation").label,
            )

            app.segmented_control(key="ai_briefing_language").select("English").run(
                timeout=30
            )
            self.assertEqual(1, translation.call_count)
            self.assertEqual(
                failed_translation,
                app.session_state["cl0014_translation_result"],
            )
            self.assertIn("Validated English headline", self._visible_text(app))

            app.segmented_control(key="ai_briefing_language").select(
                "繁體中文"
            ).run(timeout=30)
            self.assertEqual(1, translation.call_count)
            self.assertEqual(
                failed_translation,
                app.session_state["cl0014_translation_result"],
            )
            self.assertEqual(
                "重試繁體中文翻譯",
                app.button(key="generate_traditional_chinese_translation").label,
            )

            app.button(key="generate_traditional_chinese_translation").click().run(
                timeout=30
            )

        self.assertEqual(2, translation.call_count)
        self.assertEqual(
            successful_translation,
            app.session_state["cl0014_translation_result"],
        )
        self.assertIn("已驗證的英文標題", self._visible_text(app))
        self.assertNotIn(
            "generate_traditional_chinese_translation",
            [button.key for button in app.button],
        )
        visible_text = self._visible_text(app)
        self.assertNotIn("deepseek", visible_text.casefold())
        self.assertNotIn("DEEPSEEK_API_KEY", visible_text)
        self.assertNotIn("503", visible_text)


if __name__ == "__main__":
    unittest.main()
