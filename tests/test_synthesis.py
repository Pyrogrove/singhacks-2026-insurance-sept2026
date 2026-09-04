"""Tests for the bounded DeepSeek synthesis layer."""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

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
        self.assertEqual(
            {**_valid_synthesis(), "disclaimer": REQUIRED_DISCLAIMER},
            result["model_synthesis"],
        )
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

    def test_model_cannot_override_deterministic_disclaimer(self) -> None:
        invalid = {**_valid_synthesis(), "disclaimer": "Model-controlled text."}
        result = synthesize_evidence(
            self.evidence,
            environment=TEST_ENVIRONMENT,
            transport=lambda *_: _response(json.dumps(invalid)),
        )
        self.assertEqual("failed", result["status"])
        self.assertEqual("STRUCTURAL_VALIDATION_FAILED", result["error"]["code"])
        self.assertIsNone(result["model_synthesis"])

    def test_python_appends_deterministic_disclaimer(self) -> None:
        result = self._run_model_output(_valid_synthesis())
        self.assertEqual("available", result["status"])
        self.assertEqual(REQUIRED_DISCLAIMER, result["model_synthesis"]["disclaimer"])

    def test_deterministic_disclaimer_is_not_semantically_scanned(self) -> None:
        sentinel = "The cash requirement is safely funded."
        with patch("priscilla.synthesis.REQUIRED_DISCLAIMER", sentinel):
            result = self._run_model_output(_valid_synthesis())
        self.assertEqual("available", result["status"])
        self.assertTrue(result["semantic_validation_passed"])
        self.assertEqual(sentinel, result["model_synthesis"]["disclaimer"])

    def test_valid_concise_output_passes(self) -> None:
        result = self._run_model_output(_valid_synthesis())
        self.assertEqual("available", result["status"])
        self.assertTrue(result["validation_passed"])

    def test_oversized_why_it_matters_fails(self) -> None:
        invalid = _valid_synthesis()
        invalid["why_it_matters"] = " ".join(["word"] * 81)
        result = self._run_model_output(invalid)
        self.assertEqual("failed", result["status"])
        self.assertEqual("STRUCTURAL_VALIDATION_FAILED", result["error"]["code"])
        self.assertIn("80 words", result["error"]["message"])
        self.assertIsNone(result["model_synthesis"])

    def test_too_many_evidence_items_fail(self) -> None:
        invalid = _valid_synthesis()
        invalid["evidence_used"] = [f"Evidence {number}" for number in range(5)]
        result = self._run_model_output(invalid)
        self.assertEqual("failed", result["status"])
        self.assertIn("evidence_used exceeds 4", result["error"]["message"])

    def test_too_many_uncertainties_fail(self) -> None:
        invalid = _valid_synthesis()
        invalid["uncertainties"] = [f"Uncertainty {number}" for number in range(4)]
        result = self._run_model_output(invalid)
        self.assertEqual("failed", result["status"])
        self.assertIn("uncertainties exceeds 3", result["error"]["message"])

    def test_too_many_questions_fail(self) -> None:
        invalid = _valid_synthesis()
        invalid["rm_questions"] = [f"Question {number}?" for number in range(4)]
        result = self._run_model_output(invalid)
        self.assertEqual("failed", result["status"])
        self.assertIn("rm_questions exceeds 3", result["error"]["message"])

    def test_too_many_review_options_fail(self) -> None:
        invalid = _valid_synthesis()
        invalid["rm_review_options"] = [f"Review option {number}" for number in range(4)]
        result = self._run_model_output(invalid)
        self.assertEqual("failed", result["status"])
        self.assertIn("rm_review_options exceeds 3", result["error"]["message"])

    def test_lending_value_headroom_terminology_passes(self) -> None:
        valid = _valid_synthesis()
        valid["evidence_used"] = ["Lending-value headroom is HKD 25.57m."]
        result = self._run_model_output(valid)
        self.assertEqual("available", result["status"])
        self.assertTrue(result["semantic_validation_passed"])

    def test_margin_call_headroom_terminology_fails(self) -> None:
        invalid = _valid_synthesis()
        invalid["evidence_used"] = ["Margin-call headroom is HKD 25.57m."]
        self._assert_semantically_rejected(invalid)

    def test_trigger_buffer_terminology_fails(self) -> None:
        invalid = _valid_synthesis()
        invalid["evidence_used"] = ["The trigger buffer is HKD 25.57m."]
        self._assert_semantically_rejected(invalid)

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
        facility = model_input["source_facts"]["facility"]
        self.assertEqual(58_000_000.0, facility["current_drawn"])
        self.assertEqual(69.41, facility["current_ltv_percentage"])
        self.assertEqual(70.0, facility["margin_call_trigger_percentage"])
        self.assertEqual(82.86, facility["utilisation_current_percentage"])
        self.assertAlmostEqual(
            0.59,
            model_input["calculated_results"][
                "facility_ltv_distance_to_trigger_percentage_points"
            ],
            places=10,
        )
        self.assertTrue(
            all("headroom" not in snapshot for snapshot in facility["ltv_series"])
        )
        serialized_input = json.dumps(model_input)
        self.assertNotIn('"headroom"', serialized_input)
        self.assertNotIn("25565930", serialized_input)
        self.assertIn(
            "headroom",
            self.evidence["source_facts"]["credit_facility"]["ltv_series"][0],
        )

    def test_model_facing_prompt_excludes_headroom_wording(self) -> None:
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
        model_facing_text = "\n".join(
            message["content"] for message in captured_payload["messages"]
        )
        self.assertNotIn("headroom", model_facing_text.casefold())
        for required_signal in ("current_drawn", "current_ltv_percentage", "70.0", "0.59"):
            self.assertIn(required_signal, model_facing_text)

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

    def test_evidence_bounded_external_funding_uncertainty_passes(self) -> None:
        valid = _valid_synthesis()
        valid["uncertainties"] = [
            "The supplied evidence does not establish whether external funding "
            "sources are available."
        ]
        result = self._run_model_output(valid)
        self.assertEqual("available", result["status"])
        self.assertTrue(result["semantic_validation_passed"])

    def test_external_funding_rm_question_passes(self) -> None:
        valid = _valid_synthesis()
        valid["rm_questions"] = [
            "Are there any external funding sources available for the HKD60m requirement?"
        ]
        result = self._run_model_output(valid)
        self.assertEqual("available", result["status"])
        self.assertTrue(result["semantic_validation_passed"])

    def test_asserted_external_business_cash_flow_in_uncertainties_is_rejected(
        self,
    ) -> None:
        invalid = _valid_synthesis()
        invalid["uncertainties"] = ["The client may have external business cash flow."]
        self._assert_semantically_rejected(invalid)

    def test_asserted_fresh_borrowing_in_uncertainties_is_rejected(self) -> None:
        invalid = _valid_synthesis()
        invalid["uncertainties"] = ["Fresh borrowing could fund the requirement."]
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

    def test_default_timeout_is_sixty_seconds(self) -> None:
        captured_timeout: list[float] = []

        def capture_timeout(
            _endpoint: str,
            _payload: dict[str, Any],
            _api_key: str,
            timeout: float,
        ) -> dict[str, Any]:
            captured_timeout.append(timeout)
            return _response(json.dumps(_valid_synthesis()))

        result = synthesize_evidence(
            self.evidence,
            environment=TEST_ENVIRONMENT,
            transport=capture_timeout,
        )
        self.assertEqual("available", result["status"])
        self.assertEqual([60.0], captured_timeout)

    def test_configured_timeout_is_used(self) -> None:
        captured_timeout: list[float] = []
        environment = {
            **TEST_ENVIRONMENT,
            "DEEPSEEK_TIMEOUT_SECONDS": "75.5",
        }

        def capture_timeout(
            _endpoint: str,
            _payload: dict[str, Any],
            _api_key: str,
            timeout: float,
        ) -> dict[str, Any]:
            captured_timeout.append(timeout)
            return _response(json.dumps(_valid_synthesis()))

        result = synthesize_evidence(
            self.evidence,
            environment=environment,
            transport=capture_timeout,
        )
        self.assertEqual("available", result["status"])
        self.assertEqual([75.5], captured_timeout)

    def test_invalid_timeout_fails_safely(self) -> None:
        for invalid_timeout in ("", "not-a-number", "0", "-1", "nan", "inf"):
            with self.subTest(timeout=invalid_timeout):
                evidence_before = copy.deepcopy(self.evidence)
                environment = {
                    **TEST_ENVIRONMENT,
                    "DEEPSEEK_TIMEOUT_SECONDS": invalid_timeout,
                }
                result = synthesize_evidence(
                    self.evidence,
                    environment=environment,
                    transport=lambda *_: self.fail("transport should not be called"),
                )
                self.assertEqual("unavailable", result["status"])
                self.assertEqual("INVALID_TIMEOUT", result["error"]["code"])
                self.assertIn("positive numeric", result["error"]["message"])
                self.assertIsNone(result["model_synthesis"])
                self.assertEqual(evidence_before, result["deterministic_evidence"])

    def test_explicitly_negated_funding_language_passes(self) -> None:
        statements = (
            "The HKD60m need is not represented as safely funded.",
            "Funding suitability remains unresolved.",
            "The evidence does not establish that the requirement is safely funded.",
        )
        for statement in statements:
            with self.subTest(statement=statement):
                valid = _valid_synthesis()
                valid["uncertainties"] = [statement]
                result = self._run_model_output(valid)
                self.assertEqual("available", result["status"])
                self.assertTrue(result["semantic_validation_passed"])

    def test_affirmative_funding_sufficiency_language_fails(self) -> None:
        field_and_statement = (
            ("headline", "The HKD60m requirement is safely funded."),
            (
                "why_it_matters",
                "The client has sufficient liquidity to fully fund the HKD60m requirement.",
            ),
        )
        for field, statement in field_and_statement:
            with self.subTest(field=field, statement=statement):
                invalid = _valid_synthesis()
                invalid[field] = statement
                self._assert_semantically_rejected(invalid)


if __name__ == "__main__":
    unittest.main()
