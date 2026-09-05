"""Translation-only Traditional Chinese rendering of validated RM synthesis."""

from __future__ import annotations

import copy
import json
import os
import re
import time
from collections import Counter
from collections.abc import Mapping
from typing import Any

from .synthesis import (
    DEFAULT_BASE_URL,
    Transport,
    _post_json,
    _resolve_timeout_seconds,
    validate_model_synthesis,
    validate_synthesis_semantics,
)


TRANSLATION_FIELDS = (
    "headline",
    "why_it_matters",
    "evidence_used",
    "uncertainties",
    "rm_questions",
    "rm_review_options",
)
LIST_FIELDS = (
    "evidence_used",
    "uncertainties",
    "rm_questions",
    "rm_review_options",
)

# These names are deliberately kept in their source form. Other critical values,
# identifiers, and acronyms are extracted generically by CRITICAL_TOKEN_PATTERN.
PROTECTED_TERMS = (
    "Mid-Levels",
    "Golden Harbour Properties",
)
CRITICAL_TOKEN_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])(?:"
    r"\d{4}-\d{2}-\d{2}|"
    r"[A-Z]{3}\d[\d,]*(?:\.\d+)?(?:m|bn)?|"
    r"[A-Z]{2,}-\d+|"
    r"\d[\d,]*(?:\.\d+)?%|"
    r"\d[\d,]*(?:\.\d+)?|"
    r"[A-Z]{2,}"
    r")(?![A-Za-z0-9_])"
)

TRANSLATION_SYSTEM_INSTRUCTION = """Translate the supplied validated English RM briefing into Traditional Chinese.
Translate prose only. Do not analyse, infer, add facts or recommendations, or remove uncertainty or qualifications.
Preserve the same six fields and every list's length and order. Return only one JSON object.
Preserve every critical financial/value token exactly as it appears in English, including monetary/value notation, percentages, numbers, dates, identifiers, material acronyms, and explicit instrument or proper names.
For example, keep HKD60m, 70.00%, 0.59, 2026-11-01, CL-0014, CN-013, LTV, and RM exact.
Do not convert HKD60m to HKD 60m or 6,000萬港元; 70.00% to 70.00％; or 0.59 to 零點五九.
Mixed Traditional Chinese prose and English financial notation is acceptable.
"""


class TranslationValidationError(ValueError):
    """Raised when translated content violates the translation-only contract."""


def _validated_english_fields(model_synthesis: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(model_synthesis, Mapping):
        raise TranslationValidationError("English synthesis must be a mapping")
    missing = sorted(set(TRANSLATION_FIELDS) - model_synthesis.keys())
    unexpected = sorted(
        model_synthesis.keys() - set(TRANSLATION_FIELDS) - {"disclaimer"}
    )
    if missing or unexpected:
        raise TranslationValidationError(
            "English synthesis does not match the validated synthesis boundary"
        )
    selected = {
        field: copy.deepcopy(model_synthesis[field]) for field in TRANSLATION_FIELDS
    }
    try:
        selected = validate_model_synthesis(selected)
        validate_synthesis_semantics(selected)
    except ValueError as exc:
        raise TranslationValidationError(
            "English synthesis did not pass canonical validation"
        ) from exc
    return selected


def _ordered_text_items(value: Mapping[str, Any]) -> list[tuple[str, str]]:
    items = [
        ("headline", value["headline"]),
        ("why_it_matters", value["why_it_matters"]),
    ]
    for field in LIST_FIELDS:
        items.extend(
            (f"{field}[{index}]", text)
            for index, text in enumerate(value[field])
        )
    return items


def _critical_tokens(text: str) -> Counter[str]:
    tokens = Counter(match.group(0) for match in CRITICAL_TOKEN_PATTERN.finditer(text))
    for term in PROTECTED_TERMS:
        tokens[term] += text.count(term)
    return +tokens


def validate_translation(
    english_synthesis: Mapping[str, Any],
    translated: Any,
) -> dict[str, Any]:
    """Check structure and exact critical tokens against corresponding English items."""
    english = _validated_english_fields(english_synthesis)
    if not isinstance(translated, dict):
        raise TranslationValidationError("Translation must be a JSON object")
    if set(translated) != set(TRANSLATION_FIELDS):
        raise TranslationValidationError("Translation fields differ from English source")

    for field in ("headline", "why_it_matters"):
        if not isinstance(translated[field], str) or not translated[field].strip():
            raise TranslationValidationError(f"Translated {field} must be non-empty text")
    for field in LIST_FIELDS:
        items = translated[field]
        if (
            not isinstance(items, list)
            or len(items) != len(english[field])
            or any(not isinstance(item, str) or not item.strip() for item in items)
        ):
            raise TranslationValidationError(
                f"Translated {field} must preserve list length and non-empty items"
            )

    translated_items = dict(_ordered_text_items(translated))
    for location, source_text in _ordered_text_items(english):
        if _critical_tokens(source_text) != _critical_tokens(
            translated_items[location]
        ):
            raise TranslationValidationError(
                f"Translated {location} changed or removed a critical token"
            )

    return {field: copy.deepcopy(translated[field]) for field in TRANSLATION_FIELDS}


def _failure_result(
    english_synthesis: Mapping[str, Any],
    code: str,
    message: str,
    *,
    model: str | None = None,
    latency_seconds: float | None = None,
) -> dict[str, Any]:
    return {
        "status": "unavailable" if code.startswith("MISSING_") else "failed",
        "model": model,
        "latency_seconds": latency_seconds,
        "english_synthesis": copy.deepcopy(dict(english_synthesis)),
        "translation": None,
        "error": {"code": code, "message": message},
    }


def translate_validated_synthesis(
    model_synthesis: Mapping[str, Any],
    *,
    environment: Mapping[str, str] | None = None,
    transport: Transport | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Translate only validated English synthesis and fail closed on any drift."""
    try:
        english = _validated_english_fields(model_synthesis)
    except TranslationValidationError as exc:
        return _failure_result(
            model_synthesis,
            "INVALID_ENGLISH_SYNTHESIS",
            str(exc),
        )

    env = os.environ if environment is None else environment
    api_key = env.get("DEEPSEEK_API_KEY", "").strip()
    model = env.get("DEEPSEEK_MODEL", "").strip()
    if not api_key:
        return _failure_result(
            english,
            "MISSING_API_KEY",
            "Translation API key is not configured",
            model=model or None,
        )
    if not model:
        return _failure_result(
            english,
            "MISSING_MODEL",
            "Translation model is not configured",
        )

    base_url = env.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL).strip().rstrip("/")
    if not base_url:
        return _failure_result(
            english,
            "MISSING_BASE_URL",
            "Translation base URL is empty",
            model=model,
        )
    try:
        request_timeout = _resolve_timeout_seconds(env, timeout)
    except ValueError as exc:
        return _failure_result(
            english,
            "INVALID_TIMEOUT",
            str(exc),
            model=model,
        )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": TRANSLATION_SYSTEM_INSTRUCTION},
            {
                "role": "user",
                "content": json.dumps(english, ensure_ascii=False),
            },
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0,
        "stream": False,
    }
    request_transport = _post_json if transport is None else transport
    started = time.perf_counter()
    try:
        response = request_transport(
            f"{base_url}/chat/completions", payload, api_key, request_timeout
        )
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise TranslationValidationError("Translation response has no choices")
        message = choices[0].get("message")
        if not isinstance(message, Mapping):
            raise TranslationValidationError("Translation response has no message")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise TranslationValidationError("Translation response is empty")
        try:
            decoded = json.loads(content)
        except json.JSONDecodeError as exc:
            raise TranslationValidationError(
                "Translation response is malformed JSON"
            ) from exc
        translation = validate_translation(english, decoded)
    except TranslationValidationError as exc:
        return _failure_result(
            english,
            "TRANSLATION_VALIDATION_FAILED",
            str(exc),
            model=model,
            latency_seconds=time.perf_counter() - started,
        )
    except Exception as exc:
        return _failure_result(
            english,
            "TRANSLATION_FAILED",
            str(exc),
            model=model,
            latency_seconds=time.perf_counter() - started,
        )

    return {
        "status": "available",
        "model": model,
        "latency_seconds": time.perf_counter() - started,
        "english_synthesis": copy.deepcopy(dict(model_synthesis)),
        "translation": translation,
        "error": None,
    }
