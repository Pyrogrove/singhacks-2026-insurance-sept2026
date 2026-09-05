"""Build deterministic, JSON-serializable evidence from the official dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_CSV_FILES = (
    "clients.csv",
    "portfolios.csv",
    "holdings.csv",
    "instruments.csv",
    "mandates.csv",
    "transactions.csv",
    "credit_facilities.csv",
    "commitments.csv",
    "planned_cash_needs.csv",
    "market_context.csv",
    "event_log.csv",
)
REQUIRED_JSON_FILES = ("rm_notes.json",)
SNAPSHOT_DATES = (
    "2025-12-31",
    "2026-02-27",
    "2026-03-31",
    "2026-06-30",
    "2026-08-26",
)


def _load_official_data(data_dir: Path) -> dict[str, Any]:
    required = REQUIRED_CSV_FILES + REQUIRED_JSON_FILES
    missing = [name for name in required if not (data_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing required official data file(s) in {data_dir}: "
            + ", ".join(missing)
        )

    loaded: dict[str, Any] = {
        Path(name).stem: pd.read_csv(data_dir / name)
        for name in REQUIRED_CSV_FILES
    }
    with (data_dir / "rm_notes.json").open(encoding="utf-8") as handle:
        notes = json.load(handle)
    if not isinstance(notes, list):
        raise ValueError("rm_notes.json must contain a JSON list")
    loaded["rm_notes"] = notes
    return loaded


def _one_row(frame: pd.DataFrame, description: str) -> pd.Series:
    if len(frame) != 1:
        raise ValueError(f"Expected exactly one {description}; found {len(frame)}")
    return frame.iloc[0]


def _native(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _selected_record(row: pd.Series, fields: tuple[str, ...]) -> dict[str, Any]:
    return {field: _native(row[field]) for field in fields}


def _is_property_linked(row: pd.Series) -> bool:
    """Classify using joined instrument metadata, independent of asset class."""
    sector = str(row["instrument_sector"]).strip().casefold()
    descriptive_metadata = " ".join(
        str(row[field]).casefold()
        for field in ("instrument_name_instrument", "underlying_reference")
        if pd.notna(row[field])
    )
    return sector == "real estate" or any(
        marker in descriptive_metadata for marker in ("property", "properties", "real estate")
    )


def build_client_evidence(
    data_dir: str | Path,
    client_id: str = "CL-0014",
    as_of: str = "2026-08-26",
) -> dict[str, Any]:
    """Return a deterministic evidence packet composed only from official data."""
    data_path = Path(data_dir)
    data = _load_official_data(data_path)

    clients = data["clients"]
    client = _one_row(
        clients.loc[clients["client_id"] == client_id],
        f"client row for {client_id}",
    )
    portfolios = data["portfolios"].loc[
        data["portfolios"]["client_id"] == client_id
    ].copy()
    if portfolios.empty:
        raise ValueError(f"No portfolios found for client {client_id}")
    base_currencies = portfolios["base_currency"].dropna().unique().tolist()
    if len(base_currencies) != 1:
        raise ValueError(
            f"Expected one portfolio base currency for {client_id}; found {base_currencies}"
        )
    base_currency = str(base_currencies[0])

    client_holdings = data["holdings"].loc[
        data["holdings"]["client_id"] == client_id
    ].copy()
    if client_holdings.empty:
        raise ValueError(f"No holdings found for client {client_id}")
    instruments = data["instruments"][
        [
            "instrument_id",
            "instrument_name",
            "asset_class",
            "sector",
            "underlying_reference",
            "liquidity_tier",
        ]
    ].rename(
        columns={
            "instrument_name": "instrument_name_instrument",
            "asset_class": "instrument_asset_class",
            "sector": "instrument_sector",
            "liquidity_tier": "instrument_liquidity_tier",
        }
    )
    joined = client_holdings.merge(
        instruments,
        on="instrument_id",
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    unmatched = sorted(joined.loc[joined["_merge"] != "both", "instrument_id"].unique())
    if unmatched:
        raise ValueError(f"Holdings reference missing instrument metadata: {unmatched}")
    inconsistent_currency = joined.loc[joined["portfolio_ccy"] != base_currency]
    if not inconsistent_currency.empty:
        raise ValueError(f"Holdings contain a portfolio currency other than {base_currency}")
    joined["is_property_linked"] = joined.apply(_is_property_linked, axis=1)

    available_snapshots = sorted(joined["snapshot_date"].astype(str).unique().tolist())
    if available_snapshots != list(SNAPSHOT_DATES):
        raise ValueError(
            f"Expected official snapshots {list(SNAPSHOT_DATES)}; found {available_snapshots}"
        )
    if as_of not in available_snapshots:
        raise ValueError(f"No holdings snapshot for {client_id} as of {as_of}")

    portfolio_snapshots: list[dict[str, Any]] = []
    for snapshot_date in SNAPSHOT_DATES:
        snapshot = joined.loc[joined["snapshot_date"] == snapshot_date]
        total = float(snapshot["market_value_base"].sum())
        property_value = float(
            snapshot.loc[snapshot["is_property_linked"], "market_value_base"].sum()
        )
        portfolio_snapshots.append(
            {
                "snapshot_date": snapshot_date,
                "currency": base_currency,
                "total_portfolio_market_value": total,
                "property_linked_market_value": property_value,
                "property_linked_percentage": property_value / total * 100,
            }
        )

    current = joined.loc[joined["snapshot_date"] == as_of].copy()
    property_components = current.loc[current["is_property_linked"]].sort_values(
        "instrument_id"
    )
    component_records = [
        {
            "instrument_id": str(row["instrument_id"]),
            "instrument_name": str(row["instrument_name_instrument"]),
            "asset_class": str(row["instrument_asset_class"]),
            "sector": str(row["instrument_sector"]),
            "underlying_reference": _native(row["underlying_reference"]),
            "market_value_base": float(row["market_value_base"]),
            "liquidity_tier": str(row["instrument_liquidity_tier"]),
        }
        for _, row in property_components.iterrows()
    ]
    current_snapshot = next(
        item for item in portfolio_snapshots if item["snapshot_date"] == as_of
    )
    current_snapshot = {
        **current_snapshot,
        "property_linked_components": component_records,
        "daily_liquidity_gross_market_value": float(
            current.loc[
                current["instrument_liquidity_tier"] == "Daily", "market_value_base"
            ].sum()
        ),
    }

    facilities = data["credit_facilities"].loc[
        data["credit_facilities"]["client_id"] == client_id
    ]
    facility = _one_row(facilities, f"credit facility for {client_id}")
    if str(facility["facility_ccy"]) != base_currency:
        raise ValueError("Facility and portfolio currencies do not match")
    if as_of not in SNAPSHOT_DATES:
        raise ValueError(f"Unsupported facility as-of date: {as_of}")
    ltv_series = [
        {
            "snapshot_date": date,
            "drawn": float(facility[f"drawn_{date}"]),
            "collateral_market_value": float(
                facility[f"collateral_market_value_{date}"]
            ),
            "lending_value": float(facility[f"lending_value_{date}"]),
            "ltv_percentage": float(facility[f"ltv_pct_{date}"]),
            "headroom": float(facility[f"headroom_{date}"]),
        }
        for date in SNAPSHOT_DATES
    ]
    current_ltv = float(facility[f"ltv_pct_{as_of}"])
    trigger_ltv = float(facility["margin_call_ltv_pct"])

    needs = data["planned_cash_needs"].loc[
        data["planned_cash_needs"]["client_id"] == client_id
    ].sort_values("need_id")
    confirmed_needs = needs.loc[needs["certainty"] == "Confirmed"]
    cash_need_records = [
        _selected_record(
            row,
            (
                "need_id",
                "description",
                "currency",
                "amount",
                "due_from",
                "due_to",
                "recurrence",
                "certainty",
            ),
        )
        for _, row in confirmed_needs.iterrows()
    ]

    note_records = [
        {
            "note_id": str(note["note_id"]),
            "date": str(note["note_date"]),
            "channel": str(note["channel"]),
            "source_note_text": str(note["note"]),
        }
        for note in data["rm_notes"]
        if note.get("client_id") == client_id
    ]
    note_records.sort(key=lambda note: (note["date"], note["note_id"]))

    period_start = "2026-02-27"
    period_end = "2026-03-31"
    facility_balance_increase = float(
        facility[f"drawn_{period_end}"] - facility[f"drawn_{period_start}"]
    )
    transactions = data["transactions"]
    drawdowns = transactions.loc[
        (transactions["client_id"] == client_id)
        & (transactions["transaction_type"] == "Facility Drawdown")
        & (transactions["trade_date"] > period_start)
        & (transactions["trade_date"] <= period_end)
    ].sort_values(["trade_date", "transaction_id"])
    drawdown_total = float(drawdowns["amount"].sum())
    drawdown_records = [
        _selected_record(
            row,
            (
                "transaction_id",
                "trade_date",
                "settlement_date",
                "currency",
                "amount",
                "narrative",
            ),
        )
        for _, row in drawdowns.iterrows()
    ]

    return {
        "client_id": client_id,
        "as_of": as_of,
        "source_facts": {
            "label": "SOURCE_FACTS",
            "client": _selected_record(
                client,
                (
                    "client_id",
                    "client_name",
                    "base_currency",
                    "source_of_wealth",
                    "risk_profile",
                    "liquidity_needs",
                    "objectives",
                ),
            ),
            "portfolios": [
                _selected_record(
                    row,
                    (
                        "portfolio_id",
                        "portfolio_name",
                        "mandate_code",
                        "service_model",
                        "base_currency",
                    ),
                )
                for _, row in portfolios.sort_values("portfolio_id").iterrows()
            ],
            "credit_facility": {
                "facility_id": str(facility["facility_id"]),
                "facility_type": str(facility["facility_type"]),
                "collateral_portfolio_id": str(facility["collateral_portfolio_id"]),
                "currency": str(facility["facility_ccy"]),
                "current_drawn": float(facility[f"drawn_{as_of}"]),
                "current_ltv_percentage": current_ltv,
                "margin_call_trigger_percentage": trigger_ltv,
                "utilisation_current_percentage": float(
                    facility["utilisation_pct_current"]
                ),
                "ltv_series": ltv_series,
            },
            "confirmed_cash_needs": cash_need_records,
            "rm_notes": note_records,
        },
        "calculated_results": {
            "portfolio_snapshot_series": portfolio_snapshots,
            "current_snapshot": current_snapshot,
            "facility_ltv_distance_to_trigger_percentage_points": trigger_ltv
            - current_ltv,
            "funding_suitability": "UNRESOLVED",
        },
        "data_tensions": [
            {
                "label": "UNRESOLVED_DATA_TENSION",
                "period_start": period_start,
                "period_end": period_end,
                "currency": str(facility["facility_ccy"]),
                "facility_balance_increase": facility_balance_increase,
                "logged_facility_drawdown_transactions": drawdown_total,
                "unreconciled_difference": facility_balance_increase - drawdown_total,
                "transactions": drawdown_records,
                "explanation": None,
            }
        ],
        "hypotheses": [],
    }


def _format_money(currency: str, amount: float) -> str:
    return f"{currency} {amount:,.0f}"


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    packet = build_client_evidence(repo_root / "data")
    client = packet["source_facts"]["client"]
    current = packet["calculated_results"]["current_snapshot"]
    facility = packet["source_facts"]["credit_facility"]
    cash_need = packet["source_facts"]["confirmed_cash_needs"][0]
    tension = packet["data_tensions"][0]
    currency = client["base_currency"]

    print(f"{client['client_id']} - {client['client_name']} evidence ({packet['as_of']})")
    print(f"Portfolio total: {_format_money(currency, current['total_portfolio_market_value'])}")
    print(
        "Property-linked: "
        f"{_format_money(currency, current['property_linked_market_value'])} "
        f"({current['property_linked_percentage']:.2f}%)"
    )
    print(
        "Daily-liquidity gross market value: "
        f"{_format_money(currency, current['daily_liquidity_gross_market_value'])}"
    )
    print(
        f"Facility {facility['facility_id']}: drawn "
        f"{_format_money(currency, facility['current_drawn'])}; LTV "
        f"{facility['current_ltv_percentage']:.2f}% / "
        f"{facility['margin_call_trigger_percentage']:.2f}% trigger"
    )
    print(
        f"Confirmed cash need: {_format_money(cash_need['currency'], cash_need['amount'])} "
        f"({cash_need['due_from']} to {cash_need['due_to']})"
    )
    print(
        f"{tension['label']}: facility increase "
        f"{_format_money(currency, tension['facility_balance_increase'])}, logged drawdowns "
        f"{_format_money(currency, tension['logged_facility_drawdown_transactions'])}, "
        f"difference {_format_money(currency, tension['unreconciled_difference'])}"
    )
    print("Funding suitability: UNRESOLVED")


if __name__ == "__main__":
    main()
