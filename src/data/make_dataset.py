from pathlib import Path
import pandas as pd


def load_raw_dataset(input_path: Path) -> pd.DataFrame:
    return pd.read_csv(input_path)


def save_processed_dataset(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    raw_path = Path("data/raw/dataset.csv")
    processed_path = Path("data/processed/dataset_processed.csv")

    if raw_path.exists():
        dataset = load_raw_dataset(raw_path)
        save_processed_dataset(dataset, processed_path)
        print(f"Saved processed dataset to {processed_path}")
    else:
        print(f"No input file found at {raw_path}")
