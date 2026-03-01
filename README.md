# GW_TAI_NLx_Job_Posting

## Guide & Datasets

- Case_Table = Core metadata (jurisdiction, claims, parties, outcomes)
- Docket_Table = Procedural timeline events
- Document_Table = Filed documents & attachments
- Secondary_Source_Coverage_Table = Media or external references

Structuring National Labor Exchange (NLx) Job Posting Data.

## Project Structure

```text
GW_TAI_NLx_Job_Posting/
├── data/
│   ├── external/
│   ├── interim/
│   ├── processed/
│   └── raw/
├── models/
├── notebooks/
├── references/
│   └── data_dictionary.md
├── reports/
│   └── figures/
├── src/
│   ├── data/
│   │   └── make_dataset.py
│   ├── features/
│   │   └── build_features.py
│   ├── models/
│   │   ├── predict_model.py
│   │   └── train_model.py
│   └── visualization/
│       └── visualize.py
├── .env.example
├── requirements.txt
└── README.md
```

## Quick Start

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Place raw files in `data/raw/`.
4. Run baseline data processing script:

```bash
python src/data/make_dataset.py
```

## Notes

- Keep exploratory notebooks in `notebooks/`.
- Save trained models to `models/`.
- Save plots and figures to `reports/figures/`.

## Hackathon Automation

Use the end-to-end hackathon scripts in `hackathon/`.

```bash
python -m hackathon.scripts.run_all
```

Or run step-by-step:

```bash
python -m hackathon.scripts.prepare_data
python -m hackathon.scripts.run_local
```

## Data Artifacts

Running `python -m hackathon.scripts.prepare_data` prepares these files:

- Raw inputs:
	- `data/raw/colorado.csv`
	- `data/raw/colorado_processed.csv`
- Processed NLP artifacts:
	- `data/processed/nlp_skill_mentions.csv`
	- `data/processed/nlp_skill_profiles.csv`
	- `data/processed/nlp_requirements_profile.csv` (education and experience preprocessing output)
