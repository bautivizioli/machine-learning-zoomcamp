
import joblib
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "model.pkl"


def load_model():
    return joblib.load(MODEL_PATH)


def predict(student):
    bundle = load_model()

    model = bundle["model"]
    dv = bundle["dv"]
    threshold = bundle["threshold"]

    X = dv.transform([student])

    probability = model.predict_proba(X)[0, 1]
    prediction = int(probability >= threshold)

    return {
        "depression_probability": round(float(probability), 4),
        "depression_prediction": prediction,
        "threshold": threshold,
    }


if __name__ == "__main__":
    example_student = {
        "gender": "female",
        "age": 22,
        "family_background": "middle_class",
        "relationship_status": "single",
        "marital_status": "unmarried",
        "living_style": "bachelor",
        "current_semester": 8,
        "cgpa": 3.43,
        "friend_circle": "yes",
        "cost_without_study": 10113,
        "home_division": "chittagong",
    }

    result = predict(example_student)
    print(result)
