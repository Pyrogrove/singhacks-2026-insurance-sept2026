"""Tests for the bounded DeepSeek synthesis layer."""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path
from typing import Any

from priscilla.evidence import build_client_evidence
from priscilla.synthesis import REQUIRED_DISCLAIMER, synthesize_evidence


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TEST_ENVIRONMENT = {
    "DEEPSEEK_API_KEY": "test-secret-not-real",
    "DEEPSEEK_MODEL": "test-model",
}


def _valid_synthesis() -> dict[str, Any]:
    return {
        "headline": "Concentrated collateral and near-trigger LTV require RM review",
        "why_it_matters": "The supplied evidence presents connected liquidity and collateral facts.",
        "evidence_used": ["Property-linked exposure is 49.03%."],
        "uncertainties": ["The HKD 2m drawdown difference remains unexplained."],
        "rm_questions": ["What funding sources does the client expect to use?"],
        "rm_review_options": ["Review collateral and liquidity facts with the client."],
        "disclaimer": REQUIRED_DISCLAIMER,
    }


def _response(content: str) -> dict[str, Any]:
    return {"choices": [{"message": {"content": content}}]}


class SynthesisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.evidence = build_client_evidence(DATA_DIR)

    def test_valid_model_json_is_accepted_without_changing_evidence(self) -> None:
        before = copy.deepcopy(self.evidence)
        result = synthesize_evidence(
            self.evidence,
            environment=TEST_ENVIRONMENT,
            transport=lambda *_: _response(json.dumps(_valid_synthesis())),
        )
        self.assertEqual("available", result["status"])
        self.assertTrue(result["validation_passed"])
        self.assertEqual(_valid_synthesis(), result["model_synthesis"])
        self.assertEqual(before, self.evidence)
        self.assertEqual(before, result["deterministic_evidence"])

    def test_malformed_json_is_rejected_without_fallback(self) -> None:
        result = synthesize_evidence(
            self.evidence,
            environment=TEST_ENVIRONMENT,
            transport=lambda *_: _response("not-json"),
        )
        self.assertEqual("failed", result["status"])
        self.assertFalse(result["validation_passed"])
        self.assertIsNone(result["model_synthesis"])

    def test_missing_required_fields_are_rejected(self) -> None:
        incomplete = _valid_synthesis()
        del incomplete["headline"]
        result = synthesize_evidence(
            self.evidence,
            environment=TEST_ENVIRONMENT,
            transport=lambda *_: _response(json.dumps(incomplete)),
        )
        self.assertEqual("failed", result["status"])
        self.assertIsNone(result["model_synthesis"])

    def test_missing_api_key_fails_safely(self) -> None:
        called = False

        def transport(*_: Any) -> dict[str, Any]:
            nonlocal called
            called = True
            return _response(json.dumps(_valid_synthesis()))

        result = synthesize_evidence(
            self.evidence,
            environment={"DEEPSEEK_MODEL": "test-model"},
            transport=transport,
        )
        self.assertEqual("unavailable", result["status"])
        self.assertEqual("MISSING_API_KEY", result["error"]["code"])
        self.assertFalse(called)
        self.assertIsNone(result["model_synthesis"])

    def test_missing_model_fails_safely(self) -> None:
        result = synthesize_evidence(
            self.evidence,
            environment={"DEEPSEEK_API_KEY": "test-secret-not-real"},
            transport=lambda *_: self.fail("transport should not be called"),
        )
        self.assertEqual("unavailable", result["status"])
        self.assertEqual("MISSING_MODEL", result["error"]["code"])
        self.assertIsNone(result["model_synthesis"])

    def test_http_api_error_fails_safely_without_fallback(self) -> None:
        def failing_transport(*_: Any) -> dict[str, Any]:
            raise RuntimeError("DeepSeek HTTP error 503")

        result = synthesize_evidence(
            self.evidence,
            environment=TEST_ENVIRONMENT,
            transport=failing_transport,
        )
        self.assertEqual("failed", result["status"])
        self.assertIn("503", result["error"]["message"])
        self.assertIsNone(result["model_synthesis"])
        self.assertEqual(self.evidence, result["deterministic_evidence"])

    def test_empty_model_response_fails_without_synthetic_text(self) -> None:
        result = synthesize_evidence(
            self.evidence,
            environment=TEST_ENVIRONMENT,
            transport=lambda *_: _response(""),
        )
        self.assertEqual("failed", result["status"])
        self.assertIsNone(result["model_synthesis"])

    def test_rm_authority_disclaimer_is_required(self) -> None:
        invalid = _valid_synthesis()
        invalid["disclaimer"] = "Review this output carefully."
        result = synthesize_evidence(
            self.evidence,
            environment=TEST_ENVIRONMENT,
            transport=lambda *_: _response(json.dumps(invalid)),
        )
        self.assertEqual("failed", result["status"])
        self.assertIsNone(result["model_synthesis"])

    def test_authoritative_calculated_metrics_are_not_altered(self) -> None:
        metrics_before = copy.deepcopy(self.evidence["calculated_results"])
        result = synthesize_evidence(
            self.evidence,
            environment=TEST_ENVIRONMENT,
            transport=lambda *_: _response(json.dumps(_valid_synthesis())),
        )
        self.assertEqual(
            metrics_before,
            result["deterministic_evidence"]["calculated_results"],
        )

    def test_model_input_is_bounded_and_category_labelled(self) -> None:
        captured_payload: dict[str, Any] = {}

        def capture_transport(
            _endpoint: str,
            payload: dict[str, Any],
            _api_key: str,
            _timeout: float,
        ) -> dict[str, Any]:
            captured_payload.update(payload)
            return _response(json.dumps(_valid_synthesis()))

        result = synthesize_evidence(
            self.evidence,
            environment=TEST_ENVIRONMENT,
            transport=capture_transport,
        )
        self.assertEqual("available", result["status"])
        model_input = json.loads(captured_payload["messages"][1]["content"])
        self.assertEqual(
            {"source_facts", "calculated_results", "data_tensions", "hypotheses"},
            set(model_input),
        )
        self.assertEqual(
            "CL-0014", model_input["source_facts"]["client_context"]["client_id"]
        )
        self.assertNotIn("clients", model_input)

    def test_timeout_fails_safely(self) -> None:
        def timing_out(*_: Any) -> dict[str, Any]:
            raise TimeoutError("request timed out")

        result = synthesize_evidence(
            self.evidence,
            environment=TEST_ENVIRONMENT,
            transport=timing_out,
        )
        self.assertEqual("failed", result["status"])
        self.assertIsNone(result["model_synthesis"])
        self.assertEqual(self.evidence, result["deterministic_evidence"])

    def _run_model_output(self, synthesis: dict[str, Any]) -> dict[str, Any]:
        return synthesize_evidence(
            self.evidence,
            environment=TEST_ENVIRONMENT,
            transport=lambda *_: _response(json.dumps(synthesis)),
        )

    def _assert_semantically_rejected(self, synthesis: dict[str, Any]) -> None:
        evidence_before = copy.deepcopy(self.evidence)
        result = self._run_model_output(synthesis)
        self.assertEqual("failed", result["status"])
        self.assertEqual("SEMANTIC_VALIDATION_FAILED", result["error"]["code"])
        self.assertTrue(result["structural_validation_passed"])
        self.assertFalse(result["semantic_validation_passed"])
        self.assertIsNone(result["model_synthesis"])
        self.assertEqual(evidence_before, result["deterministic_evidence"])
        self.assertEqual(evidence_before, self.evidence)

    def test_held_to_maturity_is_rejected(self) -> None:
        invalid = _valid_synthesis()
        invalid["headline"] = "Held-to-maturity assets create urgency"
        self._assert_semantically_rejected(invalid)

    def test_forced_liquidation_is_rejected(self) -> None:
        invalid = _valid_synthesis()
        invalid["why_it_matters"] = "The facility could result in forced liquidation."
        self._assert_semantically_rejected(invalid)

    def test_rm_must_act_is_rejected(self) -> None:
        invalid = _valid_synthesis()
        invalid["headline"] = "The RM must act before the cash need"
        self._assert_semantically_rejected(invalid)

    def test_illiquid_collateral_is_rejected(self) -> None:
        invalid = _valid_synthesis()
        invalid["why_it_matters"] = "The illiquid collateral mix increases concern."
        self._assert_semantically_rejected(invalid)

    def test_speculative_hkd2m_causes_are_rejected(self) -> None:
        for cause in ("fees", "accrued interest", "undisclosed transactions"):
            with self.subTest(cause=cause):
                invalid = _valid_synthesis()
                invalid["uncertainties"] = [f"The HKD2m could be {cause}."]
                self._assert_semantically_rejected(invalid)

    def test_unexplained_hkd2m_wording_passes(self) -> None:
        valid = _valid_synthesis()
        valid["uncertainties"] = [
            "The HKD2m remains unexplained by the supplied evidence."
        ]
        result = self._run_model_output(valid)
        self.assertEqual("available", result["status"])
        self.assertTrue(result["semantic_validation_passed"])

    def test_external_resources_are_allowed_only_as_rm_question(self) -> None:
        valid = _valid_synthesis()
        valid["rm_questions"] = [
            "Does the client have external business cash resources?"
        ]
        self.assertEqual("available", self._run_model_output(valid)["status"])

        invalid = _valid_synthesis()
        invalid["why_it_matters"] = "The client has external business cash resources."
        self._assert_semantically_rejected(invalid)

    def test_non_binding_review_option_is_allowed(self) -> None:
        valid = _valid_synthesis()
        valid["rm_review_options"] = ["Review the collateral position."]
        result = self._run_model_output(valid)
        self.assertEqual("available", result["status"])
        self.assertTrue(result["semantic_validation_passed"])

    def test_directive_review_option_is_rejected(self) -> None:
        invalid = _valid_synthesis()
        invalid["rm_review_options"] = ["The RM should review the collateral position."]
        self._assert_semantically_rejected(invalid)

    def test_safe_funding_claim_is_rejected(self) -> None:
        invalid = _valid_synthesis()
        invalid["headline"] = "The HKD60m cash need is safely funded"
        self._assert_semantically_rejected(invalid)

    def test_margin_call_occurrence_claim_is_rejected(self) -> None:
        invalid = _valid_synthesis()
        invalid["why_it_matters"] = "A margin call has already occurred."
        self._assert_semantically_rejected(invalid)


if __name__ == "__main__":
    unittest.main()
