"""Tests for the bounded Traditional Chinese translation derivative."""

from __future__ import annotations

import copy
import json
import unittest
from typing import Any

from priscilla.translation import (
    TRANSLATION_FIELDS,
    TRANSLATION_SYSTEM_INSTRUCTION,
    TranslationValidationError,
    translate_validated_synthesis,
    validate_translation,
)


TEST_ENVIRONMENT = {
    "DEEPSEEK_API_KEY": "test-secret-not-real",
    "DEEPSEEK_MODEL": "test-model",
}


def _english_synthesis() -> dict[str, Any]:
    return {
        "headline": "CL-0014 has 69.41% LTV versus a 70.00% trigger.",
        "why_it_matters": (
            "The HKD60m need runs from 2026-11-01 to 2027-06-30, while the "
            "facility is 0.59 percentage-point from its trigger."
        ),
        "evidence_used": [
            "Golden Harbour Properties exposure remains material.",
            "Mid-Levels is part of the confirmed cash need.",
        ],
        "uncertainties": ["The HKD2m difference remains unexplained."],
        "rm_questions": ["What should the RM clarify with the client?"],
        "rm_review_options": ["Review CN-013 with the client."],
        "disclaimer": (
            "The Relationship Manager remains responsible for advice and action."
        ),
    }


def _traditional_chinese_translation() -> dict[str, Any]:
    return {
        "headline": "CL-0014 的 LTV 為 69.41%，相對於 70.00% 觸發點。",
        "why_it_matters": (
            "HKD60m 的資金需要期為 2026-11-01 至 2027-06-30，而該融資距離觸發點為 "
            "0.59 個百分點。"
        ),
        "evidence_used": [
            "Golden Harbour Properties 的風險承擔仍然重大。",
            "Mid-Levels 是已確認現金需要的一部分。",
        ],
        "uncertainties": ["HKD2m 的差異仍未解釋。"],
        "rm_questions": ["RM 應向客戶澄清甚麼？"],
        "rm_review_options": ["與客戶檢視 CN-013。"],
    }


def _response(content: str) -> dict[str, Any]:
    return {"choices": [{"message": {"content": content}}]}


def _translation_transport(
    translated: dict[str, Any],
    captured: dict[str, Any] | None = None,
) -> Any:
    def transport(
        _endpoint: str,
        payload: dict[str, Any],
        _api_key: str,
        _timeout: float,
    ) -> dict[str, Any]:
        if captured is not None:
            captured.update(copy.deepcopy(payload))
        return _response(json.dumps(translated, ensure_ascii=False))

    return transport


class TranslationTests(unittest.TestCase):
    def test_valid_chinese_translation_with_exact_critical_tokens_passes(self) -> None:
        english = _english_synthesis()
        translated = _traditional_chinese_translation()

        result = validate_translation(english, translated)

        self.assertEqual(translated, result)

    def test_hkd_token_with_inserted_space_fails(self) -> None:
        translated = _traditional_chinese_translation()
        translated["why_it_matters"] = translated["why_it_matters"].replace(
            "HKD60m", "HKD 60m"
        )
        with self.assertRaises(TranslationValidationError):
            validate_translation(_english_synthesis(), translated)

    def test_hkd_token_converted_to_chinese_value_fails(self) -> None:
        translated = _traditional_chinese_translation()
        translated["why_it_matters"] = translated["why_it_matters"].replace(
            "HKD60m", "6,000萬港元"
        )
        with self.assertRaises(TranslationValidationError):
            validate_translation(_english_synthesis(), translated)

    def test_percentage_with_full_width_symbol_fails(self) -> None:
        translated = _traditional_chinese_translation()
        translated["headline"] = translated["headline"].replace(
            "70.00%", "70.00％"
        )
        with self.assertRaises(TranslationValidationError):
            validate_translation(_english_synthesis(), translated)

    def test_changed_numeric_value_fails(self) -> None:
        translated = _traditional_chinese_translation()
        translated["why_it_matters"] = translated["why_it_matters"].replace(
            "0.59", "0.60"
        )
        with self.assertRaises(TranslationValidationError):
            validate_translation(_english_synthesis(), translated)

    def test_missing_critical_acronym_fails(self) -> None:
        translated = _traditional_chinese_translation()
        translated["headline"] = translated["headline"].replace("LTV", "按揭成數")
        with self.assertRaises(TranslationValidationError):
            validate_translation(_english_synthesis(), translated)

    def test_missing_or_extra_field_fails(self) -> None:
        for mutate in (
            lambda value: value.pop("headline"),
            lambda value: value.update({"extra_field": "不得新增欄位"}),
        ):
            with self.subTest(mutate=mutate):
                translated = _traditional_chinese_translation()
                mutate(translated)
                with self.assertRaises(TranslationValidationError):
                    validate_translation(_english_synthesis(), translated)

    def test_added_or_removed_list_item_fails(self) -> None:
        for mutate in (
            lambda value: value["evidence_used"].append("新增項目。"),
            lambda value: value["evidence_used"].pop(),
        ):
            with self.subTest(mutate=mutate):
                translated = _traditional_chinese_translation()
                mutate(translated)
                with self.assertRaises(TranslationValidationError):
                    validate_translation(_english_synthesis(), translated)

    def test_malformed_json_fails_closed(self) -> None:
        result = translate_validated_synthesis(
            _english_synthesis(),
            environment=TEST_ENVIRONMENT,
            transport=lambda *_: _response("not-json"),
        )

        self.assertEqual("failed", result["status"])
        self.assertEqual("TRANSLATION_VALIDATION_FAILED", result["error"]["code"])
        self.assertIsNone(result["translation"])

    def test_transport_receives_only_six_validated_english_fields(self) -> None:
        english = _english_synthesis()
        captured: dict[str, Any] = {}

        result = translate_validated_synthesis(
            english,
            environment=TEST_ENVIRONMENT,
            transport=_translation_transport(
                _traditional_chinese_translation(), captured
            ),
        )

        self.assertEqual("available", result["status"])
        model_input = json.loads(captured["messages"][1]["content"])
        expected = {field: english[field] for field in TRANSLATION_FIELDS}
        self.assertEqual(expected, model_input)
        self.assertNotIn("disclaimer", model_input)
        serialized = json.dumps(model_input)
        for prohibited in (
            "holdings",
            "source_facts",
            "credit_facility",
            "market_context",
            "event_log",
            "rm_notes",
        ):
            self.assertNotIn(prohibited, serialized)

    def test_prompt_is_concise_translation_only_and_has_no_placeholders(self) -> None:
        normalized = " ".join(TRANSLATION_SYSTEM_INSTRUCTION.split())
        for instruction in (
            "Translate the supplied validated English RM briefing into Traditional Chinese.",
            "Translate prose only.",
            "Do not analyse, infer, add facts or recommendations",
            "remove uncertainty or qualifications",
            "Preserve the same six fields",
            "Preserve every critical financial/value token exactly",
            "Do not convert HKD60m to HKD 60m or 6,000萬港元",
            "70.00% to 70.00％",
            "0.59 to 零點五九",
        ):
            self.assertIn(instruction, normalized)
        self.assertNotIn("[[PG_T", TRANSLATION_SYSTEM_INSTRUCTION)


if __name__ == "__main__":
    unittest.main()
