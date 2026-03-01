# Colorado Workforce Intelligence Platform

GW Hackathon 2026 project for NLx Colorado job postings, combining structured data processing, NLP skill extraction, semantic job matching, veteran translation, market intelligence analysis, recruiter tools, and usage analytics.

## What This Project Does

- Builds reusable NLP artifacts from NLx job data.
- Runs skill-based job matching with TF-IDF + cosine similarity.
- Provides explainability via skill-gap analysis.
- Supports student pathway planning (trending and emerging skills, no-degree pathways).
- Supports veteran translation from MOS/MOC to civilian opportunities.
- Provides market intelligence diagnostics (credential inflation, salary fairness, ghost language, CIP alignment).
- Provides recruiter utilities (description quality scoring, salary benchmarking).
- Logs usage analytics persistently to SQLite + CSV.

## Core NLP / Analytics Methods

1. TF-IDF vectorization (unigram + bigram skill text)
2. Cosine similarity for profile matching
3. Skill-gap extraction (matched vs missing skills)
4. Frequency analysis for demand trends
5. MOS/MOC mapping + semantic expansion for veterans
6. Raw-vs-taxonomy confidence analysis for emerging skills
7. Ghost-vs-real posting language comparison (TF-IDF differential)
8. O*NET-grouped credential inflation detection
9. Rule-based job-description quality scoring

## Repository Layout

```text
GW_TAI_NLx_Job_Posting/
├── data/
│   ├── Colorado-Hackathon-Dataset-selected/   # zipped NLx source files
│   ├── raw/                                   # extracted CSVs
│   ├── processed/                             # generated NLP artifacts + analytics output
│   ├── interim/
│   └── external/
├── hackathon/
│   ├── app.py                                # Streamlit app
│   ├── core/
│   │   ├── data.py                           # loading, extraction, preprocessing, artifact lifecycle
│   │   ├── nlp_pipeline.py                   # NLP extraction + requirement inference
│   │   ├── matching.py                       # TF-IDF index + ranking + skill gap
│   │   ├── student.py                        # student helper utilities
│   │   ├── veterans.py                       # MOS/MOC translation utilities
│   │   ├── intelligence.py                   # market/recruiter analytics functions
│   │   └── analytics_logger.py               # persistent analytics logger (SQLite + CSV)
│   └── scripts/
│       ├── prepare_data.py                   # extract + build artifacts
│       ├── run_local.py                      # launch Streamlit locally
│       ├── run_colab.py                      # Streamlit + ngrok tunnel
│       └── run_all.py                        # prepare + run local
├── docs/                                     # project docs, assumptions, architecture diagrams
├── references/
├── reports/
├── notebooks/
├── src/                                      # baseline DS scaffold
├── requirements.txt
└── README.md
```

## Data Inputs

Expected source archives in:

- `data/Colorado-Hackathon-Dataset-selected/colorado.csv.zip`
- `data/Colorado-Hackathon-Dataset-selected/colorado_processed.csv.zip`

`prepare_data` extracts these to:

- `data/raw/colorado.csv`
- `data/raw/colorado_processed.csv`

## Generated Artifacts

Running `python -m hackathon.scripts.prepare_data` generates:

- `data/processed/nlp_skill_mentions.csv`
- `data/processed/nlp_skill_profiles.csv`
- `data/processed/nlp_requirements_profile.csv`

At app runtime, usage analytics are persisted under:

- `data/processed/analytics/usage_analytics.db`
- `data/processed/analytics/usage_analytics_events.csv`

## Quick Start

### 1) Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Prepare NLx artifacts

```bash
python -m hackathon.scripts.prepare_data
```

### 3) Run the app

```bash
python -m hackathon.scripts.run_local
```

Or directly:

```bash
python -m streamlit run hackathon/app.py
```

### 4) One-command local pipeline

```bash
python -m hackathon.scripts.run_all
```

### 5) Colab/tunnel mode (optional)

```bash
python -m hackathon.scripts.run_colab
```

## Streamlit App Workflows

- **Job Seeker**: semantic matching, filters, skill-gap, adjacent-career discovery
- **Student / Career Changer**: trending skills, field explorer, taxonomy source view, emerging skills, apprenticeship pathways
- **Veteran**: direct MOC matches + skill-based translation, optional federal filter
- **Market Intelligence**: credential inflation, salary fairness, ghost language diagnostics, CIP alignment
- **Recruiter Tools**: description quality scoring, salary benchmark tool
- **Usage Insights**: visit/search/recommendation analytics from persistent logs

## Documentation Assets

- `docs/PROJECT_DOCUMENTATION.md`
- `docs/PROJECT_DOCUMENTATION.pdf`
- `docs/MODELING_ASSUMPTIONS.md`
- `docs/MODELING_ASSUMPTIONS.pdf`
- `docs/diagrams/NLP_SKILL_EXTRACTION_ARCHITECTURE.png`
- `docs/diagrams/NLP_SKILL_EXTRACTION_ARCHITECTURE.jpg`
- `docs/diagrams/NLP_SKILL_EXTRACTION_ARCHITECTURE.pdf`

## Team-Oriented Module Ownership (Suggested)

- **Data/ML track**: `hackathon/core/data.py`, `hackathon/core/nlp_pipeline.py`, `hackathon/core/matching.py`
- **App/UI track**: `hackathon/app.py`
- **Story/Policy track**: `hackathon/core/veterans.py`, `hackathon/core/intelligence.py`, docs outputs

## Notes and Constraints

- Results are based on posted jobs, not hiring outcomes.
- Salary availability is incomplete across postings.
- Ghost-job labels depend on source labeling assumptions.
- Veteran mapping is approximate and dictionary coverage is partial.
- The dataset is a snapshot and not real-time labor-market telemetry.
