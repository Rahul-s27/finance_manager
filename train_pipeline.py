import os
from pathlib import Path

import joblib
import pandas as pd

from pipeline.preprocessing import clean_data
from pipeline.feature_engineering import create_features
from pipeline.train_models import train_models
from pipeline.evaluate_models import evaluate_models


def main() -> int:
    root_dir = Path(__file__).resolve().parent
    data_path = root_dir / "data" / "training_data.csv"
    model_dir = root_dir / "models"
    model_path = model_dir / "best_model.pkl"
    vectorizer_path = model_dir / "vectorizer.pkl"

    df = pd.read_csv(data_path)

    df = clean_data(df)

    X, y, vectorizer = create_features(df)

    models, X_test, y_test = train_models(X, y)

    results = evaluate_models(models, X_test, y_test)

    best_model_name = max(results, key=results.get)

    best_model = models[best_model_name]

    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(best_model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    print("Best model:", best_model_name)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
