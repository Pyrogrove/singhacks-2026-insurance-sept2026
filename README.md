# Julius Baer — Wealth Intelligence

> **From Portfolio Monitoring to Intelligence: Reimagining Wealth Advisory** — Build an AI-powered wealth intelligence experience that transforms traditional portfolio monitoring into proactive, personalised, and explainable advisory insights.

**SingHacks 2026**

---

## 🧭 This Repository's Implementation: Priscilla

**Priscilla** is the RM Intelligence Workbench built for this challenge — a single-page Streamlit
app that turns the dataset above into an evidence-led investigation tool for one relationship
manager's book of 20 clients.

**Status: frozen.** This is a retained release candidate. Application behaviour and visual design
are not being changed; this run only concerns documentation, repository hygiene and secret safety.

### What it does

* **Deterministic RM book scan** ([priscilla/book_scan.py](priscilla/book_scan.py)) — screens the
  full client book for liquidity, credit-facility and commitment flags without any AI involved, so
  there is always a trustworthy starting view even if the AI briefing is unavailable.
* **Deterministic client evidence** ([priscilla/evidence.py](priscilla/evidence.py)) — assembles
  JSON-serializable, auditable evidence for a single client from the raw CSV/JSON dataset (positions
  across the five snapshots, mandate bands, credit facilities, commitments, RM notes, market
  context).
* **AI-generated decision briefing** ([priscilla/synthesis.py](priscilla/synthesis.py)) — sends the
  bounded evidence to an LLM (DeepSeek, via a Chat Completions–style HTTP API) to produce a
  structured brief (headline, why-it-matters, evidence used, uncertainties, RM questions, review
  options), then **validates the response both structurally and semantically** before it is shown —
  rejecting outputs that invent numbers, imply autonomous advice, or use disallowed terminology
  (e.g. margin-call/headroom language not grounded in the evidence). Deterministic evidence remains
  visible even when the AI briefing fails or is not configured.
* **Traditional Chinese translation** ([priscilla/translation.py](priscilla/translation.py)) — an
  optional translation pass over the validated English brief, checked for critical-token
  preservation (numbers, currencies, percentages, identifiers) so nothing material is lost or altered
  in translation. **This UI path is intentionally hidden in the public demo** (see
  `tests/test_streamlit_app.py`, tests skipped with `"Traditional Chinese UI is intentionally hidden
  in the public demo"`), though the underlying module and its tests remain in the repository.
* **Single-page workbench UI** ([streamlit_app.py](streamlit_app.py)) — presents the book scan, lets
  the RM drill into one client, shows deterministic evidence first, and offers the AI briefing as an
  explicit, RM-triggered action. The RM's authority and the tool's non-autonomous scope are stated in
  the UI before any AI content is generated.

### Architecture and stack

* **Language/runtime**: Python (see [requirements.txt](requirements.txt) /
  [requirements-demo.txt](requirements-demo.txt) — `pandas>=2.0`, `streamlit>=1.57`)
* **UI**: Streamlit, single page (`streamlit_app.py`)
* **AI provider**: DeepSeek chat-completions API, configured solely via the `DEEPSEEK_API_KEY`
  environment variable (see [priscilla/synthesis.py](priscilla/synthesis.py) and
  [priscilla/translation.py](priscilla/translation.py)) — no key is hardcoded or committed anywhere
  in this repository
* **Data**: static CSV/JSON files in [`data/`](data/) (see [The Dataset](#-the-dataset) below and
  [docs/DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md)); no database
* **Tests**: `unittest` suite in [`tests/`](tests/) covering book scan, per-client evidence
  (`CL-0014`), synthesis validation, translation validation and the Streamlit app's control flow

### Setup and local run

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate | macOS/Linux: source .venv/bin/activate
pip install -r requirements-demo.txt

# Optional — enables the AI briefing feature; without it, deterministic evidence still works
export DEEPSEEK_API_KEY=your-key-here   # Windows PowerShell: $env:DEEPSEEK_API_KEY = "your-key-here"

streamlit run streamlit_app.py
```

`starter/quickstart.py` remains available as the original, dependency-light way to see the raw
dataset shape (`pip install -r requirements.txt && python starter/quickstart.py`).

### Verified deployment

Public demo: **https://priscilla-rm-intelligence-v4.streamlit.app** (Streamlit Community Cloud).

### Verification actually performed (this milestone)

* Full test suite run locally: `python -m unittest discover -s tests -v` — **86 tests, all passing**
  (4 intentionally skipped, corresponding to the hidden Chinese UI path noted above).
* `py_compile` over `streamlit_app.py`, `priscilla/*.py` and `starter/quickstart.py` — no syntax
  errors.
* Repository secret scan: no `.env` or credential files tracked; grep for
  key/secret/password/token patterns across the codebase confirmed all matches are the
  `DEEPSEEK_API_KEY` environment-variable name (never a value) plus unrelated uses of the word
  "token" for text-processing tokens.
* No linter or type-checker (ruff/flake8/mypy/black, etc.) is configured in this repository at this
  milestone.

### Known limitations

* No automated lint or static type-checking is configured.
* The Traditional Chinese translation UI path is implemented and tested but hidden from the public
  demo build.
* The AI briefing feature requires a `DEEPSEEK_API_KEY` to be configured in the deployment
  environment; without it, the app still functions with deterministic evidence only.
* This is a hackathon-scope build over synthetic data, not a production banking system — see
  Trust, Governance & Explainability below for what a production path would need to address.

---

> ## ⚠️ All data in this repository is synthetic
>
> Every client, portfolio, holding, transaction and relationship-manager note in `data/` was
> **generated for this hackathon**. No real client data is present. No instrument identifier
> corresponds to a real security, and all company and individual names are invented.
>
> Market levels, exchange rates and the event log *are* calibrated to real 2026 market history, so
> that portfolio behaviour is explainable against events that actually happened.
>
> Treat the files as you would real client data anyway. That habit is part of the exercise.

---

## Challenge Summary

**Goal**: Design a next-generation digital wealth advisory experience that helps Relationship Managers understand **what is happening in a client's portfolio → what could happen next → what actions should be considered**.

**Build path**: Create an AI-powered wealth intelligence layer that continuously monitors portfolios, identifies risks and opportunities, generates personalised recommendations, and supports better RM-client conversations.

> **📖 IMPORTANT**: Read this README before you start building. The dataset has a time dimension and a governance constraint that are easy to miss, and both change what a good solution looks like.

---

## 📋 The Problem We're Solving

### Current State

* Julius Baer continues to modernise its digital channels while maintaining a relationship-driven private banking model
* Clients and Relationship Managers can already access portfolio valuations, performance, asset allocations, and market information digitally
* Existing tools are often **descriptive rather than advisory**
* RMs must manually interpret portfolio risks, market implications, tax considerations, and potential actions
* Wealth portfolios are increasingly complex across asset classes, jurisdictions, currencies, mandates, and client objectives

There is an opportunity to create an **AI-powered wealth intelligence layer** that helps RMs understand and explain portfolio performance, anticipate potential developments, and identify actions worth considering.

### Who Benefits

* **Primary users**: Relationship Managers
* **Clients**: More timely, personalised, and informed advisory conversations
* **Internal stakeholders**: Product, digital-channel, technology, risk, and compliance teams evaluating how such a solution could fit into the Julius Baer ecosystem

---

## 🎯 What You're Building

The challenge is to move from:

> **"What does my client's portfolio look like?"**

to:

> **"What should I know, and what should I do next?"**

```text
┌──────────────────────────────────────────────────────────────┐
│                       Client Context                         │
│  Portfolio • Mandate • Risk Profile • Tax • Goals • Events   │
└────────────────────────────┬─────────────────────────────────┘
                             ↓
┌──────────────────────────────────────────────────────────────┐
│                  AI Wealth Intelligence Layer                │
│   Monitor • Analyse • Explain • Recommend • Stress Test      │
└────────────────────────────┬─────────────────────────────────┘
                             ↓
┌──────────────────────────────────────────────────────────────┐
│                   RM Intelligence Workbench                  │
│      Prioritise • Review • Prepare • Compare • Decide        │
└────────────────────────────┬─────────────────────────────────┘
                             ↓
┌──────────────────────────────────────────────────────────────┐
│                    Client Advisory Action                    │
│              Discuss • Rebalance • Plan • Act                │
└──────────────────────────────────────────────────────────────┘
```

---

## 🏗️ Three Building Blocks

The capabilities below sit under three headline blocks. **You are not expected to build all of
them.** Pick the ones that serve the story you want to tell, and go deep.

### 1. Intelligent Portfolio Explanations

Explain what a portfolio did and why, connecting real market and geopolitical events to the
individual holdings that moved.

* AI-powered portfolio monitoring that surfaces meaningful observations rather than more charts
* Attribution a client would actually understand

### 2. Proactive Risk & Opportunity Detection

Surface concentration, liquidity, currency and mandate risks — and event-driven ideas — before the
client has to ask.

* Client-specific risk alerts: drift, concentration, liquidity, currency, collateral
* Event-based opportunity engine connecting market developments to affected portfolios
* Portfolio stress testing and scenario analysis

### 3. RM Intelligence Workbench

Turn insight into client-ready actions, with the Relationship Manager in control.

* Personalised recommendations grounded in mandate, risk profile, tax position and objectives
* Rebalancing suggestions, with the reasoning attached
* Tax-aware optimisation opportunities
* Life-event wealth planning: retirement, business sale, philanthropy, education, succession
* A prioritised view across the whole book, so the RM knows who to call first

---

## 📊 The Dataset

Everything is in [`data/`](data/). Field-by-field definitions are in
[`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md).

### The scenario

**Priscilla Ong** is a Relationship Manager on the Asia desk, covering the Singapore and Hong Kong
booking centres. She looks after **all 20 clients** in this dataset — from a HNW individual with
around USD 8m to a multi-generational family office with USD 88m. That is a realistic book for one
RM.

**Today is 26 August 2026.** She has client meetings over the next fortnight.

### ⏱️ Five snapshots, not one

This is the most important thing to know about the data. Positions are provided at **five dated
snapshots**:

| Date | Why it's there |
|---|---|
| 2025-12-31 | Baseline, before this year's events |
| 2026-02-27 | The day before the Middle East conflict began |
| 2026-03-31 | After the Strait of Hormuz closure |
| 2026-06-30 | Half-year, after the June technology drawdown |
| 2026-08-26 | Today |

One snapshot tells you what a portfolio **is**. Comparing snapshots tells you what **happened**.
Most of the interesting work lives in the comparison — if you treat this as static data, the
explanation capability is not reachable.

### Files

| File | What's in it |
|---|---|
| `clients.csv` | The 20 clients: age, life stage, source of wealth, risk profile, tax domicile, stated objectives |
| `portfolios.csv` | 24 portfolios. Some clients have more than one — this matters |
| `holdings.csv` | Every position at all five snapshots. The centre of gravity, 1,015 rows |
| `instruments.csv` | What each instrument is, price history, and what structured products actually reference |
| `mandates.csv` | Allocation bands and concentration limits each portfolio should respect |
| `transactions.csv` | Trades, income, fees, capital calls, credit drawdowns |
| `credit_facilities.csv` | Lombard and term loans secured against portfolios, with loan-to-value history |
| `commitments.csv` | Money committed to private funds but not yet called |
| `planned_cash_needs.csv` | What clients will need money for, and when |
| `market_context.csv` | Gold, Brent, yields, FX, equity indices and volatility at the same five dates |
| `event_log.csv` | What happened in the world in 2026, and the channels through which it reached portfolios |
| `rm_notes.json` | Priscilla's own notes. Informal, subjective, and often the most useful file here |

### 🔒 `event_log.csv` is the authoritative source

For anything that happened in 2026, **use `event_log.csv` rather than what your model remembers**.
If they disagree, the file wins.

This is not bureaucracy — it is the point. A real advisory system cannot let a language model
free-associate about geopolitics in front of a client. Grounding explanations in a controlled,
auditable event source is the difference between an explanation you can defend in a compliance
review and one that merely sounds plausible.

### Quickstart

Download **`singhacks-jb-wealth-intelligence.zip`** from the challenge page and unzip it. No git
required.

```bash
unzip singhacks-jb-wealth-intelligence.zip
cd singhacks-jb-wealth-intelligence

pip install -r requirements.txt
python starter/quickstart.py
```

On Windows, right-click the zip and choose *Extract All*, then open a terminal in the extracted
folder and run the two commands above.

`starter/quickstart.py` loads every file and prints the book, the event timeline, the market table
and one worked client. It deliberately computes nothing clever — it exists so you can see the shape
of the data in 30 seconds.

### Where to start

1. **Read three files by hand before writing any code.** Open `clients.csv`, `rm_notes.json` and
   `event_log.csv` and just read them. Twenty clients is small enough to hold in your head, and the
   notes will tell you things no query will surface.
2. **Pick one client and follow them through time.** Look at what they held in December 2025 versus
   today, then work out from `event_log.csv` which events touched them. That loop — position,
   change, cause — is the core of the whole challenge.
3. **Then decide what to build.** You will have a far better sense of what would actually help
   Priscilla than if you had started from the technology.

### Things worth knowing

* **Some clients hold more than one portfolio.** A risk can be invisible in each one individually
  and obvious once you combine them.
* **`instruments.underlying_reference` tells you what a structured product is exposed to.** The
  asset class only tells you what it is called.
* **The RM notes sometimes disagree with the numbers.** That is not a bug. Where a client says one
  thing and their portfolio says another is usually where the real advice is.
* **Private markets valuations lag.** Quarterly-reported funds are normally a quarter behind. That
  is how the industry works, not an error.
* **The data contains a small number of real-world imperfections**, of the kind present in any
  bank's systems. Handling them thoughtfully counts in your favour; assuming they are absent does
  not.

---

## 🧠 Intelligence Inputs

Insights should be generated from relevant client and market context, including portfolio
composition, investment mandate, risk profile, geographic and currency exposure, tax considerations,
market conditions, client objectives and life events.

The objective is not to display more data, but to identify **what matters to the RM and why**.

---

## 🔄 Example Advisory Flow

```text
Portfolio / Market Signal
          ↓
AI Detects Relevant Change
          ↓
Assess Client-Specific Impact
          ↓
Generate Explanation or Alert
          ↓
Recommend Potential Actions
          ↓
RM Reviews Insight
          ↓
Client Conversation / Advisory Action
```

A strong solution demonstrates how the RM moves from **signal → understanding → decision → client engagement**.

---

## 🛡️ Trust, Governance & Explainability

AI-driven wealth advisory must preserve trust and the central role of the Relationship Manager.

* **Explainability** — Why was this insight or recommendation generated?
* **Suitability** — Does it respect the client's mandate, risk profile and objectives?
* **Human oversight** — Can the RM review, reject, or modify recommendations?
* **Traceability** — Can the supporting data and assumptions be inspected?
* **Compliance** — Could this workflow operate inside a regulated bank?
* **Security** — How would sensitive client and portfolio information be protected?

Recommendations should support **human decision-making rather than replace it**.

---

## 🛠️ Technology

Use **any technology stack, APIs, AI models, frameworks, software or hardware** you like.

Consider how your approach could realistically operate in a private banking environment: security,
scalability, data protection, integration, explainability, compliance.

---

## 🏆 Judging Criteria

| Criteria | Weight | Description |
| --- | --- | --- |
| **Client-Centric Innovation** | 25% | Degree to which the solution addresses real private-banking client needs and differentiates Julius Baer's digital offering |
| **User Experience & Design** | 25% | Simplicity, clarity, and actionability of wealth insights |
| **Technical & Operational Feasibility** | 25% | Realism of implementation within banking architecture, including security, scalability and compliance |
| **Strategic Impact** | 25% | Potential to strengthen Julius Baer's position as a modern, tech-enabled wealth manager while preserving the central role of the Relationship Manager |

### What we are actually assessing

**This is not a mathematics test.** We are not checking whether your percentages match ours to two
decimal places. We are assessing whether you understood what you were looking at.

A team that says *"this client's bond portfolio is down USD 5.6m"* has done arithmetic.

A team that says *"this client is 71, retired, and drawing USD 1.1m a year from a bond portfolio
that is down USD 5.6m because yields rose after the energy shock. He has told his RM he will not
sell at a loss — but his longest bond does not mature until 2045, so waiting for it to recover is
not a plan he can outlive. Here is how we would open that conversation"* has understood the client.

The second wins, even if the first number is more precise.

Specifically, we value:

* **Reasoning you can defend.** An insight an RM cannot explain to a client in a meeting is not usable.
* **Judgement about what matters.** There is far more in this dataset than you can address in a weekend. Choosing well is part of the assessment.
* **Honesty about uncertainty.** "We are not sure, and here is what we would check" beats a confident answer the data does not support. Confident fabrication scores badly.
* **The human in the loop.** Priscilla remains responsible for the advice.

**Go deep on two or three clients rather than shallow across all twenty.** A demo that genuinely
understands three clients is more convincing than a dashboard that summarises twenty.

---

## 🧭 Directions the Data Supports

A menu, not a checklist. Two or three done well beats all of them done thinly.

* **Explanation** — attribute a portfolio's year-to-date change to specific events
* **Hidden risk** — concentration that only appears when you aggregate across a client's portfolios, or look through a structured product to its underlying
* **Mandate governance** — which portfolios sit outside their bands, which breaches are drift and which were client-directed
* **Liquidity** — match commitments and planned cash needs against what is actually sellable
* **Collateral** — trace loan-to-value across the five snapshots
* **Tax-aware optimisation** — look at unrealised gains and losses together within a household, and at tax domicile rather than residence
* **Life events** — objectives and cash needs describe futures the current allocations were not built for
* **Scenario analysis** — the Middle East situation is unresolved as of today. What happens if it de-escalates? If it worsens?
* **Prioritisation** — twenty clients, one RM. Who does she call first, and can you defend the ranking?

---

## 🎤 Presentation & Demo

**Format**: Presentation + Demo

Your final presentation should include:

* Clear articulation of the problem
* Clear representation of the proposed solution
* Main functional highlights
* Explanation of how the solution addresses the challenge
* Demonstration of how AI-generated insights translate into RM actions
* Visual screens, journeys, diagrams or charts where useful

Concise, comprehensive, and easy to follow.

If something in the data looks wrong or contradictory, **say so in your presentation**. Noticing is
worth more than quietly working around it.

---

## 🚀 Challenge North Star

> **Build the intelligence layer between portfolio data and the Relationship Manager.**

Help RMs understand what matters, anticipate what may happen next, and turn complex portfolio
information into timely, personalised and trustworthy advisory conversations.
