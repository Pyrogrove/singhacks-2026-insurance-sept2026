"""Single-page RM investigation workbench for the CL-0014 demo."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import streamlit as st

from priscilla.book_scan import build_book_scan
from priscilla.evidence import build_client_evidence
from priscilla.synthesis import synthesize_evidence
from priscilla.translation import translate_validated_synthesis


REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
AI_UNAVAILABLE_MESSAGE = (
    "AI briefing unavailable. Deterministic evidence remains available."
)
AI_FAILURE_MESSAGES = {
    "MISSING_API_KEY": "AI briefing service is not configured in this environment.",
    "MISSING_MODEL": "AI briefing service is not configured in this environment.",
    "MISSING_BASE_URL": "AI briefing service is not configured in this environment.",
    "INVALID_TIMEOUT": "AI briefing service is not configured in this environment.",
    "SYNTHESIS_FAILED": "AI briefing service is temporarily unavailable.",
    "STRUCTURAL_VALIDATION_FAILED": (
        "AI briefing response did not pass structural validation."
    ),
    "SEMANTIC_VALIDATION_FAILED": (
        "AI briefing response did not pass evidence-safety validation."
    ),
}
UNKNOWN_AI_FAILURE_MESSAGE = (
    "AI briefing is unavailable. Deterministic evidence remains available."
)
TRANSLATION_UNAVAILABLE_MESSAGE = (
    "繁體中文翻譯暫時無法使用。英文驗證版本仍然可用。"
)
ENGLISH_DISCLAIMER = (
    "The Relationship Manager remains responsible for advice and action."
)
ENGLISH_AUTHORITY = (
    "Priscilla supports evidence-led investigation; it does not provide autonomous "
    "advice or execute trades."
)
TRADITIONAL_CHINESE_DISCLAIMER = "客戶經理仍對建議及後續行動負責。"
TRADITIONAL_CHINESE_AUTHORITY = (
    "Priscilla 支援以證據為本的調查；不提供自主投資建議或執行交易。"
)
TRANSLATION_TEXT_FIELDS = ("headline", "why_it_matters")
TRANSLATION_LIST_FIELDS = (
    "evidence_used",
    "uncertainties",
    "rm_questions",
    "rm_review_options",
)


@st.cache_data(show_spinner=False)
def load_evidence(data_dir: str) -> dict[str, Any]:
    """Load the deterministic packet once per app process."""
    return build_client_evidence(data_dir)


@st.cache_data(show_spinner=False)
def load_book_scan(data_dir: str) -> dict[str, Any]:
    """Load the deterministic official-book screening view once per process."""
    return build_book_scan(data_dir)


def _money_millions(currency: str, amount: float) -> str:
    return f"{currency} {amount / 1_000_000:.2f}m"


def _date_label(value: str) -> str:
    return pd.Timestamp(value).strftime("%d %b %Y")


def _render_signal_card(
    column: Any,
    *,
    provenance: str,
    label: str,
    value: str,
    detail: str,
) -> None:
    with column:
        badge_color = {
            "CALCULATED RESULT": "blue",
            "VERIFIED SOURCE FACT": "green",
            "UNRESOLVED DATA TENSION": "orange",
        }[provenance]
        st.markdown(f":{badge_color}-badge[{provenance}]")
        st.metric(label, value)
        st.caption(detail)


def _compact_money(currency: str, amount: float) -> str:
    """Format whole-million evidence using the briefing's protected notation."""
    millions = amount / 1_000_000
    if not millions.is_integer():
        raise ValueError("Decision-context amounts must be whole millions")
    return f"{currency}{millions:.0f}m"


def _render_decision_context(
    evidence: Mapping[str, Any],
    *,
    traditional_chinese: bool,
) -> None:
    """Render deterministic RM context without model-generated interpretation."""
    facts = evidence["source_facts"]
    client = facts["client"]
    facility = facts["credit_facility"]
    cash_need = facts["confirmed_cash_needs"][0]
    current = evidence["calculated_results"]["current_snapshot"]
    distance = evidence["calculated_results"][
        "facility_ltv_distance_to_trigger_percentage_points"
    ]
    tension = evidence["data_tensions"][0]
    rm_note = next(
        note for note in facts["rm_notes"] if note["note_id"] == "N-019"
    )

    funding_amount = _compact_money(cash_need["currency"], cash_need["amount"])
    discrepancy = _compact_money(tension["currency"], tension["unreconciled_difference"])
    liquidity_need_zh = {"High": "高"}.get(
        client["liquidity_needs"], client["liquidity_needs"]
    )
    funding_window = (
        f"{pd.Timestamp(cash_need['due_from']).strftime('%b %Y')}–"
        f"{pd.Timestamp(cash_need['due_to']).strftime('%b %Y')}"
    )

    # N-019 is selected explicitly so its two client-context facts remain tied to
    # the verified note rather than inferred from portfolio calculations.
    if "expects to fund it partly from the portfolio" not in rm_note["source_note_text"]:
        raise ValueError("N-019 no longer supports portfolio-funded client context")
    if "surprised how little of it is liquid" not in rm_note["source_note_text"]:
        raise ValueError("N-019 no longer supports the recorded liquidity reaction")

    cards = (
        (
            "為何是現在",
            f"融資目前的 LTV 為 {facility['current_ltv_percentage']:.2f}%，與 "
            f"{facility['margin_call_trigger_percentage']:.2f}% 觸發點僅相距 "
            f"{distance:.2f} 個百分點；已確認的 Mid-Levels 重建項目 "
            f"{funding_amount} 資金需求，其 {funding_window} 資金窗口正逐步臨近。",
        ),
        (
            "為何是這位客戶",
            f"這是針對該客戶的情況：投資組合價值的 "
            f"{current['property_linked_percentage']:.2f}% 與物業相關，流動資金需要記錄為「"
            f"{liquidity_need_zh}」，而客戶預期從投資組合撥付部分 "
            f"{funding_amount} 資金。客戶經理筆記記錄他對可變現部分如此少感到意外。",
        ),
        (
            "若仍未解決的風險",
            f"若資金計劃仍未解決且可貸款價值轉弱，客戶經理在 {funding_window} "
            f"資金窗口之前或期間處理 {funding_amount} 需求的靈活性可能較低。"
            f"在客戶討論中依賴融資記錄之前，應先核對現有 {discrepancy} 融資差異。",
        ),
    ) if traditional_chinese else (
        (
            "WHY NOW",
            f"The facility is only {distance:.2f} percentage points from its "
            f"{facility['margin_call_trigger_percentage']:.2f}% trigger (current LTV: "
            f"{facility['current_ltv_percentage']:.2f}%) while the confirmed "
            f"{funding_amount} Mid-Levels redevelopment funding requirement approaches "
            f"in the {funding_window} window.",
        ),
        (
            "WHY THIS CLIENT",
            f"This is client-specific: {current['property_linked_percentage']:.2f}% "
            f"of portfolio value is property-linked, liquidity needs are recorded as "
            f"{client['liquidity_needs']}, and the client expects to fund part of the "
            f"{funding_amount} requirement from the portfolio. The RM note records that "
            "he was surprised how little was liquid.",
        ),
        (
            "RISK IF UNRESOLVED",
            f"If the funding plan remains unresolved and lending values weaken, the RM "
            f"may have less flexibility to address the {funding_amount} requirement "
            f"before or during the {funding_window} funding window. The existing "
            f"{discrepancy} facility discrepancy should be reconciled before relying on "
            "the facility records for the client discussion.",
        ),
    )

    context_columns = st.columns(3, gap="small", border=True)
    for column, (label, body) in zip(context_columns, cards, strict=True):
        with column:
            st.caption(label)
            st.write(body)


def _render_briefing_content(
    synthesis: Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    traditional_chinese: bool,
) -> None:
    labels = (
        {
            "heading": "客戶經理情報簡報",
            "supporting": "AI 協助 · 以驗證英文版本為依據",
            "status": "需由客戶經理檢視",
            "attention": "需要關注",
            "why": "為何重要",
            "unknown": "仍待確認",
            "questions": "PRISCILLA 應確認",
            "options": "可檢視選項",
            "evidence": "支持此簡報的證據",
        }
        if traditional_chinese
        else {
            "heading": "RM INTELLIGENCE BRIEFING",
            "supporting": "AI-assisted · evidence-grounded",
            "status": "RM REVIEW REQUIRED",
            "attention": "WHAT NEEDS ATTENTION",
            "why": "WHY IT MATTERS",
            "unknown": "WHAT WE DON'T KNOW",
            "questions": "WHAT PRISCILLA SHOULD ASK",
            "options": "REVIEW OPTIONS",
            "evidence": "Evidence supporting this briefing",
        }
    )

    st.subheader(labels["heading"])
    st.markdown(
        f":gray-badge[{labels['supporting']}] "
        f":orange-badge[{labels['status']}]"
    )
    _render_decision_context(
        evidence,
        traditional_chinese=traditional_chinese,
    )

    with st.container(border=True, gap="small"):
        st.caption(labels["attention"])
        st.markdown(f"**{synthesis['headline']}**")
        st.caption(labels["why"])
        st.write(synthesis["why_it_matters"])

    left, right = st.columns(2, gap="small")
    with left:
        with st.container(border=True, height="stretch"):
            st.markdown(f":orange[**{labels['unknown']}**]")
            for item in synthesis["uncertainties"]:
                st.write(f"• {item}")
    with right:
        with st.container(border=True, height="stretch"):
            st.markdown(f"**{labels['questions']}**")
            for index, item in enumerate(synthesis["rm_questions"], start=1):
                st.markdown(f"**{index:02d}**  {item}")

    st.markdown(f"**{labels['options']}**")
    option_columns = st.columns(
        len(synthesis["rm_review_options"]), gap="small", border=True
    )
    for index, (column, item) in enumerate(
        zip(option_columns, synthesis["rm_review_options"], strict=True), start=1
    ):
        with column:
            st.caption(f"{index:02d}")
            st.write(item)

    with st.expander(labels["evidence"], expanded=False):
        for item in synthesis["evidence_used"]:
            st.write(f"• {item}")

    if traditional_chinese:
        authority_lines = (
            TRADITIONAL_CHINESE_DISCLAIMER,
            TRADITIONAL_CHINESE_AUTHORITY,
        )
    else:
        authority_lines = (ENGLISH_DISCLAIMER, ENGLISH_AUTHORITY)
    st.caption(f"{authority_lines[0]}\n\n{authority_lines[1]}")


def _available_translation_payload(
    result: Any,
) -> Mapping[str, Any] | None:
    """Return a renderable cached translation only for the available state."""
    if not isinstance(result, Mapping) or result.get("status") != "available":
        return None
    payload = result.get("translation")
    if not isinstance(payload, Mapping):
        return None
    if any(
        not isinstance(payload.get(field), str) or not payload[field].strip()
        for field in TRANSLATION_TEXT_FIELDS
    ):
        return None
    if any(
        not isinstance(payload.get(field), list)
        or not payload[field]
        or any(not isinstance(item, str) or not item.strip() for item in payload[field])
        for field in TRANSLATION_LIST_FIELDS
    ):
        return None
    return payload


def _render_translation_action(
    synthesis: Mapping[str, Any],
    *,
    label: str,
) -> None:
    """Offer one explicit translation attempt and persist its result."""
    translate_clicked = st.button(
        label,
        key="generate_traditional_chinese_translation",
        icon=":material/translate:",
        width="content",
    )
    if translate_clicked:
        with st.spinner("正在準備繁體中文翻譯…"):
            st.session_state["cl0014_translation_result"] = (
                translate_validated_synthesis(synthesis)
            )
        st.rerun()


def _render_ai_result(
    result: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> None:
    if result.get("status") != "available" or not result.get("model_synthesis"):
        st.warning(AI_UNAVAILABLE_MESSAGE)
        error = result.get("error")
        if isinstance(error, Mapping):
            code = str(error.get("code", "SYNTHESIS_UNAVAILABLE"))
            st.caption(AI_FAILURE_MESSAGES.get(code, UNKNOWN_AI_FAILURE_MESSAGE))
        return

    synthesis = result["model_synthesis"]
    language = "English"

    displayed_synthesis = synthesis
    traditional_chinese = False
    if language == "繁體中文":
        translation_result = st.session_state.get("cl0014_translation_result")
        translated_payload = _available_translation_payload(translation_result)
        if translation_result is None:
            _render_translation_action(
                synthesis,
                label="產生繁體中文版本",
            )
            st.caption("繁體中文是英文驗證版本的翻譯檢視，不會產生新的分析。")
        elif translated_payload is not None:
            displayed_synthesis = translated_payload
            traditional_chinese = True
        else:
            st.warning(TRANSLATION_UNAVAILABLE_MESSAGE)
            st.caption("Validated English briefing:")
            _render_translation_action(
                synthesis,
                label="重試繁體中文翻譯",
            )

    _render_briefing_content(
        displayed_synthesis,
        evidence,
        traditional_chinese=traditional_chinese,
    )


st.set_page_config(
    page_title="Priscilla — RM Intelligence Investigator",
    page_icon=":material/query_stats:",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        --jb-navy: #0f2138;
        --jb-navy-2: #17324f;
        --jb-gold: #a9822f;
        --jb-bg: #f5f4f0;
        --jb-card: #ffffff;
        --jb-border: #e1ded4;
        --jb-muted: #6b7280;
    }
    [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        background: var(--jb-bg);
    }
    [data-testid="stHeader"] { background: transparent; }
    .block-container { padding-top: 1.1rem; padding-bottom: 2rem; max-width: 1240px; }

    .jb-header {
        background: linear-gradient(135deg, var(--jb-navy) 0%, var(--jb-navy-2) 100%);
        border: 1px solid rgba(169, 130, 47, 0.35);
        border-radius: 10px;
        padding: 16px 24px;
        margin-bottom: 12px;
        box-shadow: 0 2px 10px rgba(15, 33, 56, 0.15);
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 24px;
        flex-wrap: wrap;
    }
    .jb-header .jb-title {
        font-size: 1.5rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        color: #ffffff;
        margin: 0;
    }
    .jb-header .jb-role {
        font-size: 0.78rem;
        color: var(--jb-gold);
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-top: 1px;
    }
    .jb-header .jb-desc {
        font-size: 0.82rem;
        color: #c9cdd6;
        margin-top: 4px;
    }
    .jb-header .jb-context { text-align: right; }
    .jb-header .jb-meta {
        font-size: 0.85rem;
        font-weight: 600;
        color: #f5f4f0;
    }
    .jb-header .jb-sub {
        font-size: 0.7rem;
        color: #99a1b0;
        margin-top: 3px;
    }

    span.stMarkdownBadge {
        border-radius: 4px !important;
        font-size: 0.72rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em;
        padding: 2px 9px !important;
        border: 1px solid rgba(0, 0, 0, 0.06);
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--jb-card);
        border: 1px solid var(--jb-border) !important;
        border-radius: 8px !important;
        box-shadow: 0 1px 3px rgba(15, 33, 56, 0.05);
    }

    [data-testid="stMetricValue"] {
        font-size: 1.35rem;
        font-weight: 700;
        color: var(--jb-navy);
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        color: var(--jb-muted);
    }

    [data-testid="stExpander"] {
        border: 1px solid var(--jb-border);
        border-radius: 8px;
        background: var(--jb-card);
    }
    [data-testid="stDataFrame"] {
        border: 1px solid var(--jb-border);
        border-radius: 8px;
        overflow: hidden;
    }
    [data-testid="stTabs"] button[aria-selected="true"] p {
        color: var(--jb-navy);
        font-weight: 700;
    }
    [data-testid="stTabs"] [data-baseweb="tab-highlight"] {
        background-color: var(--jb-gold) !important;
    }

    h2, h3, h4 { margin-top: 0.15rem; margin-bottom: 0.35rem; }
    hr { margin: 0.5rem 0 1rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

book_scan = load_book_scan(str(DATA_DIR))

st.markdown(
    f"""
    <div class="jb-header">
        <div class="jb-brand">
            <p class="jb-title">PRISCILLA</p>
            <p class="jb-role">RM Intelligence Investigator</p>
            <p class="jb-desc">Evidence-led review across the RM book · SingHacks 2026 · Julius Baer Wealth Intelligence Challenge</p>
        </div>
        <div class="jb-context">
            <p class="jb-meta">{book_scan['client_count']} Clients · {book_scan['portfolio_relationship_count']} Portfolios · Snapshot {_date_label(book_scan['as_of'])}</p>
            <p class="jb-sub">Hackathon prototype using synthetic challenge data</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

evidence = load_evidence(str(DATA_DIR))
facts = evidence["source_facts"]
client = facts["client"]
facility = facts["credit_facility"]
cash_need = facts["confirmed_cash_needs"][0]
calculated = evidence["calculated_results"]
current = calculated["current_snapshot"]
tension = evidence["data_tensions"][0]
currency = client["base_currency"]

property_series = pd.DataFrame(calculated["portfolio_snapshot_series"])
property_series["Snapshot"] = pd.to_datetime(property_series["snapshot_date"])
property_series["Property-linked exposure (%)"] = property_series[
    "property_linked_percentage"
]
ltv_series = pd.DataFrame(facility["ltv_series"])
ltv_series["Snapshot"] = pd.to_datetime(ltv_series["snapshot_date"])
ltv_series["Facility LTV (%)"] = ltv_series["ltv_percentage"]
ltv_series["Margin-call trigger (70%)"] = facility[
    "margin_call_trigger_percentage"
]

def _flag_label(value: bool | None) -> str:
    if value is None:
        return "UNKNOWN"
    return "YES" if value else "NO"


def _why_surfaced(row: Mapping[str, Any]) -> str:
    flag_labels = (
        ("confirmed_cash_need_present", "Cash need"),
        ("credit_facility_present", "Credit facility"),
        ("commitment_present", "Commitment"),
        ("mandate_allocation_deviation_present", "Mandate deviation"),
    )
    return ", ".join(label for key, label in flag_labels if row[key] is True)


review_queue = sorted(
    (row for row in book_scan["clients"] if row["evidence_flag_count"] > 0),
    key=lambda row: (-row["evidence_flag_count"], row["client_id"]),
)
top_review_queue = review_queue[:5]

st.markdown("#### Review Queue")
st.caption(
    "Independent deterministic review signals. Counts indicate evidence breadth, "
    "not risk severity or investment priority."
)
if top_review_queue:
    queue_columns = st.columns(len(top_review_queue), gap="small", border=True)
    for column, row in zip(queue_columns, top_review_queue, strict=True):
        with column:
            st.markdown(f"**{row['client_id']}**")
            st.caption(row["client_name"] or "UNKNOWN")
            st.metric("Signals", row["evidence_flag_count"])
            st.caption(_why_surfaced(row))
            if row["client_id"] == "CL-0014":
                st.markdown(":violet-badge[Validated deep dive available]")

with st.expander(
    f"View complete RM Book ({book_scan['client_count']} clients)", expanded=False
):
    book_view = pd.DataFrame(
        [
            {
                "Client ID": row["client_id"],
                "Client name": row["client_name"] or "UNKNOWN",
                "Portfolio count": row["portfolio_count"],
                "Mandate code(s)": ", ".join(row["mandate_codes"]) or "UNKNOWN",
                "Confirmed cash need": _flag_label(
                    row["confirmed_cash_need_present"]
                ),
                "Credit facility": _flag_label(row["credit_facility_present"]),
                "Commitment": _flag_label(row["commitment_present"]),
                "Mandate allocation deviation": _flag_label(
                    row["mandate_allocation_deviation_present"]
                ),
                "Evidence flags": row["evidence_flag_count"],
            }
            for row in book_scan["clients"]
        ]
    )
    st.dataframe(book_view, hide_index=True, width="stretch")
    st.caption(
        "YES means supplied evidence supports the signal; NO means the available "
        "official source does not; UNKNOWN means the evidence is unavailable or "
        "insufficient."
    )

with st.expander("Evidence sources & provenance", expanded=False):
    st.write(
        "Client and portfolio relationships: clients.csv, portfolios.csv. "
        "Cash needs: planned_cash_needs.csv (Confirmed only). Credit facilities: "
        "credit_facilities.csv. Commitments: commitments.csv. Allocation comparison: "
        "holdings.csv at 2026-08-26 against explicit mandates.csv min/max ranges."
    )

st.divider()
deep_dive_heading, deep_dive_context = st.columns([2, 3], vertical_alignment="center")
with deep_dive_heading:
    st.subheader("CL-0014 — Deep investigation")
    st.caption("Lau Chi Ming")
with deep_dive_context:
    st.markdown(
        ":violet-badge[Validated deep dive available] "
        ":gray-badge[RM Priscilla Ong] "
        ":gray-badge[As of 26 Aug 2026]",
        text_alignment="right",
    )
st.caption(
    "The only client in this book with a validated deep investigation. The "
    "sections below connect the review signals to portfolio evidence, RM notes "
    "and a grounded AI briefing."
)

st.info(
    "**Funding pressure meets concentrated property exposure**",
    icon=":material/priority_high:",
)
critical_signals = st.columns(4, gap="small", border=True)
_render_signal_card(
    critical_signals[0],
    provenance="CALCULATED RESULT",
    label="Property-linked exposure",
    value=f"{current['property_linked_percentage']:.2f}%",
    detail=_money_millions(currency, current["property_linked_market_value"]),
)
_render_signal_card(
    critical_signals[1],
    provenance="CALCULATED RESULT",
    label="Facility LTV",
    value=f"{facility['current_ltv_percentage']:.2f}%",
    detail=(
        f"{facility['margin_call_trigger_percentage']:.2f}% trigger · "
        f"{calculated['facility_ltv_distance_to_trigger_percentage_points']:.2f} "
        "percentage-point distance"
    ),
)
_render_signal_card(
    critical_signals[2],
    provenance="VERIFIED SOURCE FACT",
    label="Confirmed cash need",
    value=_money_millions(currency, cash_need["amount"]),
    detail=f"{cash_need['description']} · Nov 2026–Jun 2027",
)
_render_signal_card(
    critical_signals[3],
    provenance="UNRESOLVED DATA TENSION",
    label="Unexplained difference",
    value=_money_millions(currency, tension["unreconciled_difference"]),
    detail="Facility increase vs logged drawdowns",
)

overview_tab, evidence_tab, notes_tab, ai_tab = st.tabs(
    ["Overview", "Evidence", "RM notes", "AI briefing"]
)

with overview_tab:
    why_now, investigation = st.columns([3, 2], gap="medium")
    with why_now:
        st.subheader("Why now")
        property_chart, ltv_chart = st.columns(2)
        with property_chart:
            st.markdown("**Property-linked exposure**")
            st.line_chart(
                property_series,
                x="Snapshot",
                y="Property-linked exposure (%)",
                height=170,
            )
            st.caption("48.54% → 49.03% · five verified snapshots")
        with ltv_chart:
            st.markdown("**Facility LTV versus trigger**")
            st.line_chart(
                ltv_series,
                x="Snapshot",
                y=["Facility LTV (%)", "Margin-call trigger (70%)"],
                height=170,
            )
            st.caption("53.93% → 69.41% · margin-call trigger = 70%")

    with investigation:
        with st.container(border=True):
            st.subheader("Investigation trail")
            st.markdown(
                f"**1 · Concentration** — Property-linked exposure is "
                f"**{current['property_linked_percentage']:.2f}%**.\n\n"
                f"**2 · Facility pressure** — LTV is "
                f"**{facility['current_ltv_percentage']:.2f}%** versus the "
                f"**{facility['margin_call_trigger_percentage']:.2f}%** trigger.\n\n"
                f"**3 · Future cash need** — "
                f"**{_money_millions(currency, cash_need['amount'])}** is confirmed.\n\n"
                f"**4 · Contradiction** — Facility balance rose "
                f"**{_money_millions(currency, tension['facility_balance_increase'])}**, "
                f"but logged drawdowns total "
                f"**{_money_millions(currency, tension['logged_facility_drawdown_transactions'])}**. "
                f"**{_money_millions(currency, tension['unreconciled_difference'])} "
                "remains unexplained.**"
            )
            st.caption(
                "Optional bounded AI synthesis. Deterministic evidence above remains authoritative."
            )
            generate_clicked = st.button(
                "Generate AI RM Briefing",
                type="primary",
                key="generate_ai_briefing",
                icon=":material/auto_awesome:",
                width="stretch",
            )
            if generate_clicked:
                with st.spinner("Generating RM intelligence briefing…"):
                    new_synthesis_result = synthesize_evidence(evidence)
                    st.session_state["cl0014_synthesis_result"] = new_synthesis_result
                    if (
                        new_synthesis_result.get("status") == "available"
                        and new_synthesis_result.get("model_synthesis")
                    ):
                        st.session_state.pop("cl0014_translation_result", None)

            overview_result = st.session_state.get("cl0014_synthesis_result")
            if overview_result:
                if overview_result.get("status") == "available":
                    st.success("AI synthesis is ready in the AI briefing tab.")
                else:
                    st.warning(AI_UNAVAILABLE_MESSAGE)

with evidence_tab:
    st.subheader("Deterministic evidence")
    st.caption("Authoritative portfolio calculations and source records")
    secondary_metrics = st.columns(4, border=True)
    secondary_metrics[0].metric(
        "Portfolio total",
        _money_millions(currency, current["total_portfolio_market_value"]),
    )
    secondary_metrics[0].caption("CALCULATED RESULT")
    secondary_metrics[1].metric(
        "Daily liquidity — gross",
        _money_millions(currency, current["daily_liquidity_gross_market_value"]),
    )
    secondary_metrics[1].caption(
        "CALCULATED RESULT · Funding suitability unresolved"
    )
    secondary_metrics[2].metric(
        "Facility drawn", _money_millions(currency, facility["current_drawn"])
    )
    secondary_metrics[2].caption("VERIFIED SOURCE FACT")
    secondary_metrics[3].metric(
        "Facility utilisation", f"{facility['utilisation_current_percentage']:.2f}%"
    )
    secondary_metrics[3].caption("VERIFIED SOURCE FACT")

    look_through, reconciliation = st.columns([3, 2], gap="medium")
    with look_through:
        st.markdown("#### Property exposure look-through")
        st.info("Four different instruments, one dominant economic theme")
        components = pd.DataFrame(current["property_linked_components"])
        component_view = components[
            ["instrument_name", "asset_class", "market_value_base"]
        ].rename(
            columns={
                "instrument_name": "Instrument",
                "asset_class": "Asset class",
                "market_value_base": f"Market value ({currency})",
            }
        )
        st.dataframe(
            component_view,
            hide_index=True,
            column_config={
                f"Market value ({currency})": st.column_config.NumberColumn(
                    format="accounting"
                )
            },
        )

    with reconciliation:
        st.markdown("#### Facility reconciliation")
        st.markdown(":orange-badge[UNRESOLVED DATA TENSION]")
        tension_columns = st.columns(3)
        tension_values = (
            ("Facility increase", tension["facility_balance_increase"]),
            ("Logged drawdowns", tension["logged_facility_drawdown_transactions"]),
            ("Unexplained", tension["unreconciled_difference"]),
        )
        for column, (label, value) in zip(
            tension_columns, tension_values, strict=True
        ):
            with column:
                st.metric(label, _money_millions(currency, value))
        st.warning(
            "The supplied evidence does not explain the HKD 2.00m difference."
        )
        st.caption("27 Feb–31 Mar 2026 · No cause is asserted")

with notes_tab:
    st.subheader("RM source context")
    st.caption("Verbatim records from the supplied RM notes")
    note_columns = st.columns(len(facts["rm_notes"]), border=True)
    for column, note in zip(note_columns, facts["rm_notes"], strict=True):
        with column:
            st.markdown(":green-badge[VERIFIED SOURCE FACT — RM NOTE]")
            st.caption(
                f"{_date_label(note['date'])} · {note['channel']} · {note['note_id']}"
            )
            st.markdown(f"> {note['source_note_text']}")

with ai_tab:
    if "cl0014_synthesis_result" in st.session_state:
        _render_ai_result(st.session_state["cl0014_synthesis_result"], evidence)
    else:
        st.subheader("AI briefing")
        st.markdown(":violet-badge[AI SYNTHESIS · OPTIONAL]")
        st.info(
            "Use **Generate AI RM Briefing** in Overview to request the bounded "
            "interpretation. No model call has been made."
        )
        st.caption("Deterministic evidence remains authoritative and available.")

st.caption(ENGLISH_DISCLAIMER)
st.caption(ENGLISH_AUTHORITY)
