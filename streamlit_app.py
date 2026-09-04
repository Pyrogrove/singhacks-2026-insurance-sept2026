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


@st.cache_data(show_spinner=False)
def load_evidence(data_dir: str) -> dict[str, Any]:
    """Load the deterministic packet once per app process."""
    return build_client_evidence(data_dir)


def _money_millions(currency: str, amount: float) -> str:
    return f"{currency} {amount / 1_000_000:.2f}m"


def _date_label(value: str) -> str:
    return pd.Timestamp(value).strftime("%d %b %Y")


def _render_metric_card(
    column: Any,
    *,
    provenance: str,
    label: str,
    value: str,
    detail: str,
) -> None:
    with column:
        st.caption(provenance)
        st.metric(label, value, border=True)
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
            message = str(error.get("message", "No model output is available."))
            st.caption(f"{code}: {message}")
        return

    synthesis = result["model_synthesis"]
    st.success("MODEL SYNTHESIS — DeepSeek V4 Flash")
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
    layout="wide",
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

st.title("Priscilla")
st.subheader("RM Intelligence Investigator")
st.caption("Evidence-first client investigation for Relationship Managers")
st.markdown(
    ":blue-badge[Client CL-0014] "
    ":violet-badge[RM Priscilla Ong] "
    ":gray-badge[As of 26 Aug 2026] "
    ":gray-badge[Data: Synthetic challenge dataset]"
)

st.info("### Funding pressure meets concentrated property exposure")

row_one = st.columns(3)
_render_metric_card(
    row_one[0],
    provenance="CALCULATED RESULT",
    label="Portfolio",
    value=_money_millions(currency, current["total_portfolio_market_value"]),
    detail="Current portfolio market value",
)
_render_metric_card(
    row_one[1],
    provenance="CALCULATED RESULT",
    label="Property-linked exposure",
    value=f"{current['property_linked_percentage']:.2f}%",
    detail=_money_millions(currency, current["property_linked_market_value"]),
)
_render_metric_card(
    row_one[2],
    provenance="SOURCE FACT + CALCULATED RESULT",
    label="Facility LTV",
    value=f"{facility['current_ltv_percentage']:.2f}%",
    detail=(
        f"{facility['margin_call_trigger_percentage']:.2f}% trigger · "
        f"{calculated['facility_ltv_distance_to_trigger_percentage_points']:.2f} pp distance · "
        f"{_money_millions(currency, facility['current_drawn'])} drawn · "
        f"{facility['utilisation_current_percentage']:.2f}% utilisation"
    ),
)

row_two = st.columns(3)
_render_metric_card(
    row_two[0],
    provenance="VERIFIED SOURCE FACT",
    label="Confirmed cash need",
    value=_money_millions(currency, cash_need["amount"]),
    detail=f"{cash_need['description']} · Nov 2026–Jun 2027",
)
_render_metric_card(
    row_two[1],
    provenance="CALCULATED RESULT",
    label="Daily liquidity (gross)",
    value=_money_millions(currency, current["daily_liquidity_gross_market_value"]),
    detail="Funding suitability remains unresolved",
)
_render_metric_card(
    row_two[2],
    provenance="UNRESOLVED DATA TENSION",
    label="Unexplained difference",
    value=_money_millions(currency, tension["unreconciled_difference"]),
    detail="Facility balance vs logged drawdowns",
)

st.subheader("Exposure and facility trajectory")
chart_left, chart_right = st.columns(2)
property_series = pd.DataFrame(calculated["portfolio_snapshot_series"])
property_series["Snapshot"] = pd.to_datetime(property_series["snapshot_date"])
property_series["Property-linked exposure (%)"] = property_series[
    "property_linked_percentage"
]
with chart_left:
    st.markdown("**Property-linked exposure**")
    st.line_chart(
        property_series,
        x="Snapshot",
        y="Property-linked exposure (%)",
        height=260,
    )
    st.caption("Five official portfolio snapshots · CALCULATED RESULT")

ltv_series = pd.DataFrame(facility["ltv_series"])
ltv_series["Snapshot"] = pd.to_datetime(ltv_series["snapshot_date"])
ltv_series["Facility LTV (%)"] = ltv_series["ltv_percentage"]
ltv_series["Margin-call trigger (70%)"] = facility[
    "margin_call_trigger_percentage"
]
with chart_right:
    st.markdown("**Facility LTV and trigger**")
    st.line_chart(
        ltv_series,
        x="Snapshot",
        y=["Facility LTV (%)", "Margin-call trigger (70%)"],
        height=260,
    )
    st.caption(
        f"Current distance to the 70% trigger: "
        f"{calculated['facility_ltv_distance_to_trigger_percentage_points']:.2f} "
        "percentage points"
    )

with st.expander("Property exposure look-through — current snapshot"):
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
    st.info("Four different instruments, one dominant economic theme")

context_left, context_right = st.columns(2)
with context_left:
    st.subheader("RM context")
    for note in facts["rm_notes"]:
        with st.container(border=True):
            st.caption("VERIFIED SOURCE FACT — RM NOTE")
            st.caption(f"{_date_label(note['date'])} · {note['channel']} · {note['note_id']}")
            st.markdown(f"> {note['source_note_text']}")

with context_right:
    st.subheader("Unresolved reconciliation")
    st.warning("**UNRESOLVED DATA TENSION**")
    tension_columns = st.columns(3)
    tension_values = (
        ("Facility increase", tension["facility_balance_increase"]),
        ("Logged drawdowns", tension["logged_facility_drawdown_transactions"]),
        ("Unexplained", tension["unreconciled_difference"]),
    )
    for column, (label, value) in zip(tension_columns, tension_values, strict=True):
        with column:
            st.metric(label, _money_millions(currency, value))
    st.caption(
        "27 Feb–31 Mar 2026. The supplied evidence does not explain the "
        f"{_money_millions(currency, tension['unreconciled_difference'])} difference."
    )

st.divider()
st.subheader("Optional AI RM briefing")
st.write(
    "Optional bounded AI synthesis. Deterministic evidence above remains authoritative."
)
generate_clicked = st.button(
    "Generate AI RM Briefing",
    type="primary",
    key="generate_ai_briefing",
)
if generate_clicked:
    with st.spinner("Generating bounded DeepSeek synthesis…"):
        st.session_state["cl0014_synthesis_result"] = synthesize_evidence(evidence)

if "cl0014_synthesis_result" in st.session_state:
    _render_ai_result(st.session_state["cl0014_synthesis_result"])

st.divider()
st.caption("The Relationship Manager remains responsible for advice and action.")
st.caption(
    "Priscilla supports evidence-led investigation; it does not provide autonomous "
    "advice or execute trades."
)
