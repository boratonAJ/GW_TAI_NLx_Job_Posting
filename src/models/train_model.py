from pathlib import Path
import joblib
from sklearn.dummy import DummyClassifier


def train_baseline_model(X, y):
    model = DummyClassifier(strategy="most_frequent")
    model.fit(X, y)
    return model


def save_model(model, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)


if __name__ == "__main__":
    print("Add your training pipeline in src/models/train_model.py")
