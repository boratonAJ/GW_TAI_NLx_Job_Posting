# GW_TAI_NLx_Job_Posting

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
