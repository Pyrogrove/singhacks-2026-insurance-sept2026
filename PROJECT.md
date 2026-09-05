---
project_name: Priscilla — RM Intelligence Workbench (SingHacks 2026 Julius Baer)
repository: https://github.com/Pyrogrove/singhacks-2026-insurance-sept2026
project_type: hackathon-submission
lifecycle: frozen-release-candidate
stack: python, pandas, streamlit, deepseek-api
tags: [wealth-management, ai-advisory, streamlit, hackathon, singhacks-2026]
deployment: streamlit-community-cloud
deployment_url: https://priscilla-rm-intelligence-v4.streamlit.app
last_verified: 2026-09-05
---

## Purpose

An AI-powered RM (Relationship Manager) intelligence workbench built for the SingHacks 2026 /
Julius Baer wealth-advisory challenge. It turns a synthetic 20-client private-banking dataset into
deterministic, auditable evidence plus an optional, validated AI-generated decision brief, so an RM
can move from "what does the portfolio look like" to "what should I know and do next."

## Current State

**Frozen.** HEAD of `feature/final-visual-system` is `718a1c6c69a4d0e338aa1f42e52787c0d8409bcc` as of
this milestone. This run is documentation, repository hygiene, and secret-safety only — no
application behavior or visual design changes are authorized at this time.

## Demonstrated Capabilities

* Deterministic book-wide screening across the full 20-client book (`priscilla/book_scan.py`).
* Deterministic, auditable per-client evidence assembly from five dated portfolio snapshots plus
  mandates, credit facilities, commitments, market context, and RM notes (`priscilla/evidence.py`).
* AI-generated decision briefs with structural and semantic validation against invented figures,
  autonomous-advice framing, and disallowed terminology (`priscilla/synthesis.py`).
* Traditional Chinese translation with critical-token-preservation checks (`priscilla/translation.py`),
  implemented and tested but intentionally hidden from the current public demo UI.
* Single-page Streamlit workbench (`streamlit_app.py`) that surfaces deterministic evidence first and
  the AI briefing as an explicit, RM-triggered action.

## Verification

* `python -m unittest discover -s tests -v` — 86 tests passing, 4 intentionally skipped (hidden
  Chinese UI path). Run and observed on 2026-09-05.
* `py_compile` over `streamlit_app.py`, `priscilla/*.py`, `starter/quickstart.py` — no errors.
  Observed on 2026-09-05.
* Secret-safety check: no `.env`/credential files tracked or staged; the only key/secret/token
  pattern matches in the codebase are the `DEEPSEEK_API_KEY` environment-variable name (never a
  value) and unrelated NLP "token" usage. Observed on 2026-09-05.
* Public deployment URL redirects through a Streamlit Community Cloud auth/wake gate rather than
  rendering directly when fetched programmatically; this is UNVERIFIED as a live rendering check and
  should be confirmed by a human opening the link in a browser.
* No linter or type-checker is configured in this repository — UNVERIFIED / not applicable at this
  milestone.

## Known Limitations

* No automated lint or static type-checking configured.
* Traditional Chinese UI path is present and tested but hidden from the public demo build.
* AI briefing requires `DEEPSEEK_API_KEY` in the deployment environment; the app degrades gracefully
  to deterministic-only evidence without it.
* Synthetic hackathon dataset; not a production banking system.

## Next Action

Human maintainer to independently confirm the public deployment URL renders correctly in a browser,
then decide when to make the GitHub repository public (see the retained-milestone pre-push gate
output for this run's explicit READY/REVISE/STOP assessment).
