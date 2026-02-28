# Colorado Workforce Intelligence — Automated Hackathon Scripts

This folder contains a complete, role-aligned script structure for your 3-person hackathon workflow.

## Folder Layout

```text
hackathon/
├── app.py
├── core/
│   ├── data.py
│   ├── matching.py
│   ├── student.py
│   └── veterans.py
└── scripts/
    ├── prepare_data.py
    ├── run_colab.py
    └── run_local.py
```

## Team Mapping

- **Person A (Data Person):** `core/data.py`, `core/matching.py`
- **Person B (App Person):** `app.py`
- **Person C (Veterans + Story):** `core/veterans.py`, transparency section in `app.py`

## One-Time Setup

```bash
pip install -r requirements.txt
```

## Automated Run (Local)

```bash
python -m hackathon.scripts.prepare_data
python -m hackathon.scripts.run_local
```

## Automated Run (Colab / Tunnel)

```bash
python -m hackathon.scripts.prepare_data
python -m hackathon.scripts.run_colab
```

## Data Expectations

The automation script extracts:

- `data/Colorado-Hackathon-Dataset-selected/colorado.csv.zip`
- `data/Colorado-Hackathon-Dataset-selected/colorado_processed.csv.zip`

into:

- `data/raw/colorado.csv`
- `data/raw/colorado_processed.csv`

Then the NLP pipeline builds structured outputs used by the app:

- `data/processed/nlp_skill_mentions.csv`
- `data/processed/nlp_skill_profiles.csv`
