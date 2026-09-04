"""Bounded DeepSeek interpretation over deterministic client evidence."""

from __future__ import annotations

import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .evidence import build_client_evidence


REQUIRED_DISCLAIMER = "The Relationship Manager remains responsible for advice and action."
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_TIMEOUT_SECONDS = 60.0

SYSTEM_INSTRUCTION = f"""You are a bounded interpretation layer for a relationship manager.
Use only the supplied CL-0014 evidence. Python has already calculated every authoritative metric.

Your tasks:
- identify the most important client-specific issue;
- connect already-calculated evidence and explain why the RM should care;
- surface uncertainty or contradiction;
- propose questions and review options for the RM.

You must not recompute or change metrics, invent facts or quotations, invent market or geopolitical
causes, provide definitive investment advice, claim the HKD 60m need is safely funded, assign a
monetary value to the external property-development business, recommend autonomous action, or
claim a margin call has already occurred.

Specific claim discipline:
- Do not use "held-to-maturity" unless that phrase appears in the evidence; it does not appear in
  this supplied packet.
- Do not state or imply "forced liquidation"; facility enforcement mechanics are not supplied.
- Do not say the RM "must", "should", or is "required to" take an action. Use non-binding language
  such as "merits review", "could be reviewed", "question for the RM", or "review option".
- Do not call illiquid portfolio positions "illiquid collateral". Distinguish gross portfolio
  composition from lending-value or collateral contribution.
- Do not speculate about the HKD 2m unreconciled difference. State only that it remains unexplained
  by the supplied evidence.
- Do not invent external business cash flow, fresh borrowing, sale restrictions, fees, accrued
  interest, or settlement mechanics as facts. Unknown external funding sources may appear only as
  questions for the RM.
- Do not claim the HKD 60m need is safely funded or that a margin call has occurred.

Return one JSON object only, with exactly these fields:
headline (string), why_it_matters (string), evidence_used (array of strings), uncertainties
(array of strings), rm_questions (array of strings), rm_review_options (array of strings), and
disclaimer (string). The disclaimer must include this exact sentence:
{REQUIRED_DISCLAIMER}
"""

Transport = Callable[[str, dict[str, Any], str, float], dict[str, Any]]


class SynthesisValidationError(ValueError):
    """Raised when provider content does not satisfy the synthesis contract."""


class SynthesisSemanticValidationError(SynthesisValidationError):
    """Raised when structurally valid content violates narrow claim rules."""


PROHIBITED_GENERAL_PATTERNS = (
    (re.compile(r"\bheld[ -]to[ -]maturity\b", re.IGNORECASE), "held-to-maturity"),
    (re.compile(r"\bforced liquidation\b", re.IGNORECASE), "forced liquidation"),
    (re.compile(r"\billiquid collateral\b", re.IGNORECASE), "illiquid collateral"),
    (
        re.compile(r"\b(?:safe|safely|fully) funded\b", re.IGNORECASE),
        "funding asserted as safe",
    ),
    (
        re.compile(
            r"\bmargin call (?:has |had )?(?:already )?"
            r"(?:occurred|happened|triggered|been triggered)\b",
            re.IGNORECASE,
        ),
        "margin call asserted as having occurred",
    ),
    (
        re.compile(r"\b(?:must|should|required to)\s+(?:act|take)\b", re.IGNORECASE),
        "directive language",
    ),
)
DIRECTIVE_PATTERN = re.compile(
    r"\b(?:the\s+)?(?:rm|relationship manager|priscilla)\s+"
    r"(?:must|should|needs to|has to|is required to)\s+"
    r"(?:act|take|review|assess|reconcile|discuss|consider|sell|buy|reduce|increase|"
    r"liquidate|raise|obtain|arrange|contact|advise)\b",
    re.IGNORECASE,
)
DISCREPANCY_PATTERN = re.compile(
    r"(?:hkd\s*2\s*m|2[,.]?000[,.]?000|unreconcil|discrepanc|difference)",
    re.IGNORECASE,
)
SPECULATIVE_CAUSE_PATTERN = re.compile(
    r"\b(?:undisclosed transactions?|fees?|accrued interest)\b",
    re.IGNORECASE,
)
EXTERNAL_FUNDING_PATTERN = re.compile(
    r"\b(?:external(?: business)?(?: cash)? (?:resources?|funding|funds)|"
    r"external business cash flow|fresh borrowing|sale restrictions?|settlement mechanics)\b",
    re.IGNORECASE,
)


def build_compact_model_input(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Select only relevant CL-0014 evidence for the model."""
    facts = evidence["source_facts"]
    calculated = evidence["calculated_results"]
    current = calculated["current_snapshot"]
    facility = facts["credit_facility"]
    return {
        "source_facts": {
            "client_context": facts["client"],
            "facility": {
                "facility_id": facility["facility_id"],
                "facility_type": facility["facility_type"],
                "collateral_portfolio_id": facility["collateral_portfolio_id"],
                "currency": facility["currency"],
                "current_drawn": facility["current_drawn"],
                "current_ltv_percentage": facility["current_ltv_percentage"],
                "margin_call_trigger_percentage": facility[
                    "margin_call_trigger_percentage"
                ],
                "utilisation_current_percentage": facility[
                    "utilisation_current_percentage"
                ],
                "ltv_series": facility["ltv_series"],
            },
            "confirmed_cash_needs": facts["confirmed_cash_needs"],
            "rm_notes_verbatim": facts["rm_notes"],
        },
        "calculated_results": {
            "property_exposure_series": calculated["portfolio_snapshot_series"],
            "current_property_linked_components": current[
                "property_linked_components"
            ],
            "current_total_portfolio_market_value": current[
                "total_portfolio_market_value"
            ],
            "current_property_linked_market_value": current[
                "property_linked_market_value"
            ],
            "current_property_linked_percentage": current[
                "property_linked_percentage"
            ],
            "facility_ltv_distance_to_trigger_percentage_points": calculated[
                "facility_ltv_distance_to_trigger_percentage_points"
            ],
            "daily_liquidity_gross_market_value": current[
                "daily_liquidity_gross_market_value"
            ],
            "funding_suitability": calculated["funding_suitability"],
        },
        "data_tensions": evidence["data_tensions"],
        "hypotheses": evidence["hypotheses"],
    }


def validate_model_synthesis(value: Any) -> dict[str, Any]:
    """Validate and return model JSON without inventing missing content."""
    if not isinstance(value, dict):
        raise SynthesisValidationError("Model synthesis must be a JSON object")

    required_fields = (
        "headline",
        "why_it_matters",
        "evidence_used",
        "uncertainties",
        "rm_questions",
        "rm_review_options",
        "disclaimer",
    )
    missing = sorted(set(required_fields) - value.keys())
    if missing:
        raise SynthesisValidationError(
            "Model synthesis is missing required field(s): " + ", ".join(missing)
        )
    unexpected = sorted(value.keys() - set(required_fields))
    if unexpected:
        raise SynthesisValidationError(
            "Model synthesis contains unexpected field(s): " + ", ".join(unexpected)
        )

    for field in ("headline", "why_it_matters", "disclaimer"):
        if not isinstance(value[field], str) or not value[field].strip():
            raise SynthesisValidationError(f"{field} must be a non-empty string")
    for field in (
        "evidence_used",
        "uncertainties",
        "rm_questions",
        "rm_review_options",
    ):
        items = value[field]
        if (
            not isinstance(items, list)
            or not items
            or any(not isinstance(item, str) or not item.strip() for item in items)
        ):
            raise SynthesisValidationError(
                f"{field} must be a non-empty array of non-empty strings"
            )
    if REQUIRED_DISCLAIMER not in value["disclaimer"]:
        raise SynthesisValidationError("Required RM authority disclaimer is missing")

    return {field: value[field] for field in required_fields}


def validate_synthesis_semantics(value: Mapping[str, Any]) -> None:
    """Fail closed on narrow, demonstrated unsupported claim patterns."""
    text_fields = ("headline", "why_it_matters", "disclaimer")
    list_fields = (
        "evidence_used",
        "uncertainties",
        "rm_questions",
        "rm_review_options",
    )
    field_items = {
        **{field: [value[field]] for field in text_fields},
        **{field: value[field] for field in list_fields},
    }
    discrepancy_mentioned = any(
        DISCREPANCY_PATTERN.search(text)
        for items in field_items.values()
        for text in items
    )

    for field, items in field_items.items():
        for text in items:
            for pattern, condition in PROHIBITED_GENERAL_PATTERNS:
                if pattern.search(text):
                    raise SynthesisSemanticValidationError(
                        f"Prohibited semantic condition in {field}: {condition}"
                    )
            if DIRECTIVE_PATTERN.search(text):
                raise SynthesisSemanticValidationError(
                    f"Prohibited semantic condition in {field}: RM directive"
                )
            if discrepancy_mentioned and SPECULATIVE_CAUSE_PATTERN.search(text):
                raise SynthesisSemanticValidationError(
                    f"Prohibited semantic condition in {field}: speculative HKD 2m cause"
                )
            if EXTERNAL_FUNDING_PATTERN.search(text):
                is_rm_question = field == "rm_questions" and text.strip().endswith("?")
                if not is_rm_question:
                    raise SynthesisSemanticValidationError(
                        f"Prohibited semantic condition in {field}: asserted external funding"
                    )


def _post_json(
    endpoint: str,
    payload: dict[str, Any],
    api_key: str,
    timeout: float,
) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"DeepSeek HTTP error {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"DeepSeek connection error: {exc.reason}") from exc
    parsed = json.loads(response_body)
    if not isinstance(parsed, dict):
        raise RuntimeError("DeepSeek API response was not a JSON object")
    return parsed


def _failure_result(
    evidence: Mapping[str, Any],
    status: str,
    code: str,
    message: str,
    model: str | None,
    latency_seconds: float | None = None,
    structural_validation_passed: bool = False,
    semantic_validation_passed: bool = False,
) -> dict[str, Any]:
    return {
        "status": status,
        "model": model,
        "latency_seconds": latency_seconds,
        "validation_passed": False,
        "structural_validation_passed": structural_validation_passed,
        "semantic_validation_passed": semantic_validation_passed,
        "deterministic_evidence": evidence,
        "model_synthesis": None,
        "error": {"code": code, "message": message},
    }


def _resolve_timeout_seconds(
    environment: Mapping[str, str],
    explicit_timeout: float | None,
) -> float:
    value: Any
    if explicit_timeout is not None:
        value = explicit_timeout
    elif "DEEPSEEK_TIMEOUT_SECONDS" in environment:
        value = environment["DEEPSEEK_TIMEOUT_SECONDS"]
    else:
        value = DEFAULT_TIMEOUT_SECONDS
    try:
        timeout = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "DEEPSEEK_TIMEOUT_SECONDS must be a positive numeric value"
        ) from exc
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("DEEPSEEK_TIMEOUT_SECONDS must be a positive numeric value")
    return timeout


def synthesize_evidence(
    evidence: Mapping[str, Any],
    *,
    environment: Mapping[str, str] | None = None,
    transport: Transport | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Request and validate one bounded synthesis while preserving evidence."""
    env = os.environ if environment is None else environment
    api_key = env.get("DEEPSEEK_API_KEY", "").strip()
    model = env.get("DEEPSEEK_MODEL", "").strip()
    if not api_key:
        return _failure_result(
            evidence,
            "unavailable",
            "MISSING_API_KEY",
            "DEEPSEEK_API_KEY is not configured",
            model or None,
        )
    if not model:
        return _failure_result(
            evidence,
            "unavailable",
            "MISSING_MODEL",
            "DEEPSEEK_MODEL is not configured",
            None,
        )

    base_url = env.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL).strip().rstrip("/")
    if not base_url:
        return _failure_result(
            evidence,
            "unavailable",
            "MISSING_BASE_URL",
            "DEEPSEEK_BASE_URL is empty",
            model,
        )
    try:
        request_timeout = _resolve_timeout_seconds(env, timeout)
    except ValueError as exc:
        return _failure_result(
            evidence,
            "unavailable",
            "INVALID_TIMEOUT",
            str(exc),
            model,
        )
    endpoint = f"{base_url}/chat/completions"
    compact_input = build_compact_model_input(evidence)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {
                "role": "user",
                "content": json.dumps(compact_input, ensure_ascii=False),
            },
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0,
        "stream": False,
    }

    request_transport = _post_json if transport is None else transport
    started = time.perf_counter()
    structural_validation_passed = False
    try:
        provider_response = request_transport(endpoint, payload, api_key, request_timeout)
        latency = time.perf_counter() - started
        choices = provider_response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise SynthesisValidationError("DeepSeek returned no choices")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise SynthesisValidationError("DeepSeek returned no message")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise SynthesisValidationError("DeepSeek returned empty model content")
        try:
            decoded = json.loads(content)
        except json.JSONDecodeError as exc:
            raise SynthesisValidationError("DeepSeek returned malformed JSON") from exc
        synthesis = validate_model_synthesis(decoded)
        structural_validation_passed = True
        validate_synthesis_semantics(synthesis)
    except SynthesisSemanticValidationError as exc:
        latency = time.perf_counter() - started
        return _failure_result(
            evidence,
            "failed",
            "SEMANTIC_VALIDATION_FAILED",
            str(exc),
            model,
            latency,
            structural_validation_passed=True,
        )
    except SynthesisValidationError as exc:
        latency = time.perf_counter() - started
        return _failure_result(
            evidence,
            "failed",
            "STRUCTURAL_VALIDATION_FAILED",
            str(exc),
            model,
            latency,
        )
    except Exception as exc:
        latency = time.perf_counter() - started
        return _failure_result(
            evidence,
            "failed",
            "SYNTHESIS_FAILED",
            str(exc),
            model,
            latency,
            structural_validation_passed=structural_validation_passed,
        )

    return {
        "status": "available",
        "model": model,
        "latency_seconds": latency,
        "validation_passed": True,
        "structural_validation_passed": True,
        "semantic_validation_passed": True,
        "deterministic_evidence": evidence,
        "model_synthesis": synthesis,
        "error": None,
    }


def synthesize_client_briefing(
    data_dir: str | Path,
    *,
    client_id: str = "CL-0014",
    as_of: str = "2026-08-26",
    environment: Mapping[str, str] | None = None,
    transport: Transport | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Build deterministic evidence, then request a bounded interpretation."""
    evidence = build_client_evidence(data_dir, client_id=client_id, as_of=as_of)
    return synthesize_evidence(
        evidence,
        environment=environment,
        transport=transport,
        timeout=timeout,
    )


def _money(currency: str, amount: float) -> str:
    return f"{currency} {amount:,.0f}"


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    result = synthesize_client_briefing(repo_root / "data")
    evidence = result["deterministic_evidence"]
    client = evidence["source_facts"]["client"]
    current = evidence["calculated_results"]["current_snapshot"]
    facility = evidence["source_facts"]["credit_facility"]
    tension = evidence["data_tensions"][0]
    currency = client["base_currency"]

    print("DETERMINISTIC EVIDENCE")
    print(f"Client: {client['client_id']} - {client['client_name']}")
    print(f"Portfolio total: {_money(currency, current['total_portfolio_market_value'])}")
    print(
        f"Property-linked: {_money(currency, current['property_linked_market_value'])} "
        f"({current['property_linked_percentage']:.2f}%)"
    )
    print(
        f"Facility LTV: {facility['current_ltv_percentage']:.2f}% / "
        f"{facility['margin_call_trigger_percentage']:.2f}% trigger"
    )
    print(
        "Daily liquidity (gross): "
        f"{_money(currency, current['daily_liquidity_gross_market_value'])}"
    )
    print(
        f"Data tension: {_money(currency, tension['unreconciled_difference'])} "
        "unreconciled"
    )

    print("\nMODEL SYNTHESIS")
    synthesis = result["model_synthesis"]
    if result["model"]:
        print(f"Model: {result['model']}")
    if result["latency_seconds"] is not None:
        print(f"Latency: {result['latency_seconds']:.3f} seconds")
    print(
        "Structural validation: "
        f"{'passed' if result['structural_validation_passed'] else 'not passed'}"
    )
    print(
        "Semantic validation: "
        f"{'passed' if result['semantic_validation_passed'] else 'not passed'}"
    )
    if synthesis is None:
        print(f"UNAVAILABLE ({result['error']['code']}): {result['error']['message']}")
        return
    print(f"Headline: {synthesis['headline']}")
    print(f"Why it matters: {synthesis['why_it_matters']}")
    print("Evidence used:")
    for item in synthesis["evidence_used"]:
        print(f"- {item}")
    print("Uncertainties:")
    for item in synthesis["uncertainties"]:
        print(f"- {item}")
    print("RM questions:")
    for item in synthesis["rm_questions"]:
        print(f"- {item}")
    print("RM review options:")
    for item in synthesis["rm_review_options"]:
        print(f"- {item}")
    print(f"Disclaimer: {synthesis['disclaimer']}")


if __name__ == "__main__":
    main()
