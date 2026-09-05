"""Deterministic screening flags for the official RM client book."""

from __future__ import annotations

import csv
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


AS_OF_DATE = "2026-08-26"
CORE_FILES = ("clients.csv", "portfolios.csv")
OPTIONAL_FLAG_FILES = (
    "planned_cash_needs.csv",
    "credit_facilities.csv",
    "commitments.csv",
)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_optional_rows(path: Path) -> list[dict[str, str]] | None:
    """Return ``None`` when an optional evidence source is unavailable."""
    try:
        return _read_rows(path)
    except (FileNotFoundError, OSError, csv.Error):
        return None


def _presence_by_client(
    rows: list[dict[str, str]] | None,
    *,
    confirmed_only: bool = False,
) -> dict[str, bool] | None:
    if rows is None or any("client_id" not in row for row in rows):
        return None
    if confirmed_only and any("certainty" not in row for row in rows):
        return None
    present: dict[str, bool] = {}
    for row in rows:
        if confirmed_only and row.get("certainty") != "Confirmed":
            continue
        client_id = row.get("client_id", "").strip()
        if client_id:
            present[client_id] = True
    return present


def _decimal(value: str | None) -> Decimal | None:
    try:
        return Decimal(value) if value not in (None, "") else None
    except InvalidOperation:
        return None


def _portfolio_mandate_evidence(
    portfolio: dict[str, str],
    holdings: list[dict[str, str]] | None,
    mandates: list[dict[str, str]] | None,
    *,
    as_of: str,
) -> dict[str, Any]:
    """Compare one latest portfolio allocation only with explicit mandate ranges."""
    portfolio_id = portfolio["portfolio_id"]
    mandate_code = portfolio.get("mandate_code", "").strip()
    provenance = {
        "snapshot_date": as_of,
        "holdings_source": "holdings.csv",
        "portfolio_source": "portfolios.csv",
        "mandate_source": "mandates.csv",
    }
    unknown = {
        "portfolio_id": portfolio_id,
        "mandate_code": mandate_code or None,
        "deviation_present": None,
        "comparisons": [],
        "provenance": provenance,
    }
    if holdings is None or mandates is None or not mandate_code:
        return unknown

    portfolio_holdings = [
        row
        for row in holdings
        if row.get("portfolio_id") == portfolio_id
        and row.get("snapshot_date") == as_of
    ]
    mandate_rows = [
        row for row in mandates if row.get("mandate_code") == mandate_code
    ]
    if not portfolio_holdings or not mandate_rows:
        return unknown

    totals: defaultdict[str, Decimal] = defaultdict(Decimal)
    for row in portfolio_holdings:
        asset_class = row.get("asset_class", "").strip()
        value = _decimal(row.get("market_value_base"))
        if not asset_class or value is None:
            return unknown
        totals[asset_class] += value
    total_value = sum(totals.values(), Decimal("0"))
    if total_value <= 0:
        return unknown

    comparisons: list[dict[str, Any]] = []
    ranged_classes: set[str] = set()
    incomplete_range = False
    for row in mandate_rows:
        asset_class = row.get("asset_class", "").strip()
        minimum = _decimal(row.get("min_pct"))
        maximum = _decimal(row.get("max_pct"))
        if not asset_class or minimum is None or maximum is None:
            incomplete_range = True
            continue
        ranged_classes.add(asset_class)
        actual = totals[asset_class] * Decimal("100") / total_value
        outside = actual < minimum or actual > maximum
        comparisons.append(
            {
                "asset_class": asset_class,
                "actual_pct": float(actual),
                "min_pct": float(minimum),
                "max_pct": float(maximum),
                "outside_range": outside,
            }
        )

    if not comparisons:
        return unknown
    supplied_range_deviation = any(item["outside_range"] for item in comparisons)
    held_classes_without_ranges = sorted(set(totals) - ranged_classes)
    deviation_present: bool | None
    if supplied_range_deviation:
        deviation_present = True
    elif held_classes_without_ranges or incomplete_range:
        deviation_present = None
    else:
        deviation_present = False
    return {
        "portfolio_id": portfolio_id,
        "mandate_code": mandate_code,
        "deviation_present": deviation_present,
        "comparisons": comparisons,
        "held_classes_without_ranges": held_classes_without_ranges,
        "provenance": provenance,
    }


def build_book_scan(
    data_dir: str | Path,
    as_of: str = AS_OF_DATE,
) -> dict[str, Any]:
    """Build a deterministic, non-advisory screening view of the official book."""
    root = Path(data_dir)
    missing_core = [name for name in CORE_FILES if not (root / name).is_file()]
    if missing_core:
        raise FileNotFoundError(f"Missing core book data: {', '.join(missing_core)}")

    clients = _read_rows(root / "clients.csv")
    portfolios = _read_rows(root / "portfolios.csv")
    holdings = _read_optional_rows(root / "holdings.csv")
    mandates = _read_optional_rows(root / "mandates.csv")
    optional_rows = {
        name: _read_optional_rows(root / name) for name in OPTIONAL_FLAG_FILES
    }
    confirmed_cash = _presence_by_client(
        optional_rows["planned_cash_needs.csv"], confirmed_only=True
    )
    facilities = _presence_by_client(optional_rows["credit_facilities.csv"])
    commitments = _presence_by_client(optional_rows["commitments.csv"])

    portfolios_by_client: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for portfolio in portfolios:
        portfolios_by_client[portfolio.get("client_id", "")].append(portfolio)

    scan_rows: list[dict[str, Any]] = []
    portfolio_evidence: list[dict[str, Any]] = []
    for client in sorted(clients, key=lambda row: row.get("client_id", "")):
        client_id = client.get("client_id", "").strip()
        client_portfolios = sorted(
            portfolios_by_client.get(client_id, []),
            key=lambda row: row.get("portfolio_id", ""),
        )
        mandate_evidence = [
            _portfolio_mandate_evidence(
                portfolio, holdings, mandates, as_of=as_of
            )
            for portfolio in client_portfolios
        ]
        portfolio_evidence.extend(mandate_evidence)
        deviation_values = [item["deviation_present"] for item in mandate_evidence]
        if any(value is True for value in deviation_values):
            mandate_deviation: bool | None = True
        elif deviation_values and all(value is False for value in deviation_values):
            mandate_deviation = False
        else:
            mandate_deviation = None

        cash_flag = (
            confirmed_cash.get(client_id, False)
            if confirmed_cash is not None
            else None
        )
        facility_flag = (
            facilities.get(client_id, False) if facilities is not None else None
        )
        commitment_flag = (
            commitments.get(client_id, False)
            if commitments is not None
            else None
        )
        evidence_flags = (
            cash_flag,
            facility_flag,
            commitment_flag,
            mandate_deviation,
        )
        scan_rows.append(
            {
                "client_id": client_id,
                "client_name": client.get("client_name", "").strip() or None,
                "portfolio_count": len(client_portfolios),
                "portfolio_ids": [
                    item.get("portfolio_id") for item in client_portfolios
                ],
                "mandate_codes": sorted(
                    {
                        item["mandate_code"]
                        for item in client_portfolios
                        if item.get("mandate_code")
                    }
                ),
                "confirmed_cash_need_present": cash_flag,
                "credit_facility_present": facility_flag,
                "commitment_present": commitment_flag,
                "mandate_allocation_deviation_present": mandate_deviation,
                "evidence_flag_count": sum(value is True for value in evidence_flags),
            }
        )

    return {
        "as_of": as_of,
        "client_count": len(scan_rows),
        "portfolio_relationship_count": sum(
            row["portfolio_count"] for row in scan_rows
        ),
        "clients": scan_rows,
        "portfolio_mandate_evidence": portfolio_evidence,
        "provenance": {
            "data_dir": str(root),
            "client_source": "clients.csv",
            "portfolio_source": "portfolios.csv",
            "confirmed_cash_need_source": "planned_cash_needs.csv",
            "credit_facility_source": "credit_facilities.csv",
            "commitment_source": "commitments.csv",
            "mandate_sources": ["holdings.csv", "portfolios.csv", "mandates.csv"],
        },
    }
