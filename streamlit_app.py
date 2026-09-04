"""Single-page RM investigation workbench for the CL-0014 demo."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import streamlit as st

from priscilla.evidence import build_client_evidence
from priscilla.synthesis import synthesize_evidence


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


@st.cache_data(show_spinner=False)
def load_evidence(data_dir: str) -> dict[str, Any]:
    """Load the deterministic packet once per app process."""
    return build_client_evidence(data_dir)


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


def _render_list(title: str, items: list[str]) -> None:
    st.markdown(f"**{title}**")
    for item in items:
        st.write(f"• {item}")


def _render_ai_result(result: Mapping[str, Any]) -> None:
    if result.get("status") != "available" or not result.get("model_synthesis"):
        st.warning(AI_UNAVAILABLE_MESSAGE)
        error = result.get("error")
        if isinstance(error, Mapping):
            code = str(error.get("code", "SYNTHESIS_UNAVAILABLE"))
            st.caption(AI_FAILURE_MESSAGES.get(code, UNKNOWN_AI_FAILURE_MESSAGE))
        return

    synthesis = result["model_synthesis"]
    st.markdown(":violet-badge[AI SYNTHESIS]")
    st.subheader("EVIDENCE-GROUNDED RM BRIEFING")
    st.markdown(f"### {synthesis['headline']}")
    st.write(synthesis["why_it_matters"])

    left, right = st.columns(2)
    with left:
        _render_list("Uncertainties", synthesis["uncertainties"])
        _render_list("Questions for the RM", synthesis["rm_questions"])
    with right:
        _render_list("RM review options", synthesis["rm_review_options"])

    with st.expander("Evidence used by the model"):
        for item in synthesis["evidence_used"]:
            st.write(f"• {item}")
    st.info(synthesis["disclaimer"])


st.set_page_config(
    page_title="Priscilla — RM Intelligence Investigator",
    page_icon=":material/query_stats:",
    layout="wide",
)

header_brand, header_context = st.columns([1, 3], vertical_alignment="center")
with header_brand:
    st.markdown("## Priscilla")
    st.caption("RM Intelligence Investigator")
with header_context:
    st.markdown(
        ":blue-badge[Client CL-0014] "
        ":violet-badge[RM Priscilla Ong] "
        ":gray-badge[As of 26 Aug 2026] "
        ":gray-badge[Synthetic challenge dataset]",
        text_alignment="right",
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
                    st.session_state["cl0014_synthesis_result"] = synthesize_evidence(
                        evidence
                    )

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
    st.subheader("AI briefing")
    if "cl0014_synthesis_result" in st.session_state:
        _render_ai_result(st.session_state["cl0014_synthesis_result"])
    else:
        st.markdown(":violet-badge[AI SYNTHESIS · OPTIONAL]")
        st.info(
            "Use **Generate AI RM Briefing** in Overview to request the bounded "
            "interpretation. No model call has been made."
        )
        st.caption("Deterministic evidence remains authoritative and available.")

st.caption("The Relationship Manager remains responsible for advice and action.")
st.caption(
    "Priscilla supports evidence-led investigation; it does not provide autonomous "
    "advice or execute trades."
)
