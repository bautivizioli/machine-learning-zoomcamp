
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier


# Configuration
BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "Student_Depression_Dataset.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "model.pkl"

TARGET = "depression"
THRESHOLD = 0.3

MODEL_PARAMS = {
    "n_estimators": 200,
    "max_depth": 10,
    "min_samples_leaf": 3,
    "max_features": 0.5,
    "random_state": 42,
    "n_jobs": -1
}


def load_data():
    df = pd.read_csv(DATA_PATH)

    # Normalize column names
    df.columns = df.columns.str.lower().str.replace(" ", "_")

    # Normalize categorical values
    string_columns = df.select_dtypes(include=["object", "string"]).columns

    for col in string_columns:
        df[col] = (
            df[col]
            .str.lower()
            .str.replace(" ", "_", regex=False)
        )

    # Encode target
    df[TARGET] = (df[TARGET] == "yes").astype(int)

    return df


def train_model(df):
    # Separate features and target
    X = df.drop(["depression", "health_condition"], axis=1)
    y = df[TARGET]

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Encode categorical variables
    train_dict = X_train.to_dict(orient="records")
    dv = DictVectorizer(sparse=True)

    X_train_encoded = dv.fit_transform(train_dict)

    # Train model
    model = RandomForestClassifier(**MODEL_PARAMS)
    model.fit(X_train_encoded, y_train)

    # Save model and vectorizer
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": model,
        "dv": dv,
        "threshold": THRESHOLD
    }

    joblib.dump(artifact, MODEL_PATH)

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Training records: {len(X_train)}")
    print(f"Test records: {len(X_test)}")
    print(f"Features: {X_train_encoded.shape[1]}")


def main():
    df = load_data()
    train_model(df)


if __name__ == "__main__":
    main()
