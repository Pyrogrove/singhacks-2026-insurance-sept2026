# AGENTS.md

Provider-neutral instructions for coding agents working in this repository.

## Repository purpose

SingHacks 2026 hackathon submission for Julius Baer's wealth-intelligence challenge. The repository
contains the challenge dataset/brief plus **Priscilla**, an RM Intelligence Workbench (Streamlit app)
built on top of it. See [README.md](README.md) for full context and [PROJECT.md](PROJECT.md) for
current lifecycle state.

## Status: FROZEN

This repository is a retained release candidate on `feature/final-visual-system`. **Do not change
application behaviour or visual design.** Only documentation, repository hygiene, secret-safety and
pre-publication work is in scope unless the human maintainer explicitly authorizes otherwise.

## Source of truth

* [README.md](README.md) — challenge brief, dataset description, and this repository's
  implementation (setup, architecture, verification, deployment, limitations).
* [docs/DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md) — field-level reference for every file in
  `data/`.
* `data/event_log.csv` is the **authoritative source** for any 2026 market/geopolitical event used in
  generated text — never let a model substitute its own recollection of events.

## Key directories

* `data/` — synthetic dataset (clients, portfolios, holdings, instruments, mandates, transactions,
  credit facilities, commitments, planned cash needs, market context, event log, RM notes). All data
  is synthetic; treat it as real client data anyway per the README's stated exercise rule.
* `priscilla/` — application logic: `book_scan.py` (deterministic book screening), `evidence.py`
  (deterministic per-client evidence assembly), `synthesis.py` (AI briefing generation + structural/
  semantic validation), `translation.py` (Traditional Chinese translation + critical-token
  validation).
* `streamlit_app.py` — the single-page workbench UI; entry point for `streamlit run`.
* `tests/` — `unittest`-based tests for each module above.
* `starter/quickstart.py` — minimal, dependency-light script that loads and prints the raw dataset.

## Verified commands

* Install: `pip install -r requirements-demo.txt` (adds `streamlit` on top of `requirements.txt`'s
  `pandas`).
* Run app: `streamlit run streamlit_app.py`.
* Run tests: `python -m unittest discover -s tests -v` — verified passing (86 tests, 4 intentionally
  skipped for the hidden Chinese UI path) as of `last_verified` in [PROJECT.md](PROJECT.md).
* `pytest` is **not** installed in this repository's `.venv`; use `unittest` as above unless a
  maintainer adds and documents `pytest` explicitly.
* No linter or type-checker (ruff, flake8, mypy, black, etc.) is configured. Do not assume one exists
  or invent a command for it.

## Architecture constraints and invariants

* **Deterministic evidence must stand alone.** `book_scan.py` and `evidence.py` must never depend on
  the AI provider being configured or reachable — the UI shows deterministic evidence first and the
  AI briefing as an optional, explicit, RM-triggered addition.
* **AI output is validated, not trusted.** `synthesis.py` structurally and semantically validates
  every model response before display, rejecting invented numbers, autonomous-advice framing, and
  disallowed terminology (e.g. unqualified margin-call/headroom language). Do not weaken or bypass
  this validation.
* **Translation must preserve critical tokens.** `translation.py` checks that numbers, currencies,
  percentages, and identifiers are unchanged between English and Traditional Chinese output before
  accepting a translation. Do not relax this check.
* **The RM remains the decision-maker.** UI copy (see `streamlit_app.py`'s `ENGLISH_AUTHORITY` /
  `ENGLISH_DISCLAIMER` constants) states that Priscilla supports investigation and does not provide
  autonomous advice or execute trades. Preserve this framing in any UI text changes.

## Non-obvious conventions and gotchas

* The Traditional Chinese UI path is fully implemented and tested but **intentionally hidden** in the
  public demo build — see the skip reason `"Traditional Chinese UI is intentionally hidden in the
  public demo"` in `tests/test_streamlit_app.py`. This is deliberate, not a missing feature.
* The dataset has **five dated snapshots** (2025-12-31 through 2026-08-26); most real analysis
  requires comparing snapshots, not reading one in isolation. See README's "Five snapshots" section.
* CSV reads use `encoding="utf-8-sig"` (see `evidence.py`/`book_scan.py`) — preserve this when
  touching data-loading code, as the source files may carry a BOM.

## Security and secret handling

* The only secret this repository's code reads is `DEEPSEEK_API_KEY`, always via an environment
  variable (`os.environ` / an injected `environment` mapping in tests). **Never** hardcode a key,
  commit a `.env` file, or add a `.streamlit/secrets.toml` with real values — `.gitignore` already
  excludes `.env`, `.env.*`, and `.streamlit/secrets.toml`.
* Test fixtures use the literal placeholder `"test-secret-not-real"` for `DEEPSEEK_API_KEY` — this is
  intentional and is not a real credential.
* If you find what looks like a real API key, token, password, or private key staged, committed, or
  hardcoded anywhere in this repository: **stop, do not remove or rewrite history silently, and
  report the affected file and credential type to the human maintainer** so it can be rotated/revoked.

## Definition of done / verification expectations

* Any change to `priscilla/*.py` or `streamlit_app.py` should be accompanied by running
  `python -m unittest discover -s tests -v` and confirming no new failures.
* Never weaken a test or a validation rule (structural/semantic/translation) merely to make it pass.
* Given the FROZEN status above, most agent work here should be documentation, dependency hygiene, or
  explicitly human-authorized changes — not new features.

## Authority boundaries

* Do not deploy, publish, make the GitHub repository public, or push to shared branches without the
  human maintainer's explicit approval for that specific action.
* Do not install new tooling/dependencies (linters, scanners, etc.) without approval.
* Do not modify application behavior or visual design while the FROZEN status in this file and in
  [PROJECT.md](PROJECT.md) holds.
