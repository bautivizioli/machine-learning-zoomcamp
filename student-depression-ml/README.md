# Student Depression Prediction

A machine learning project that predicts whether a student belongs to the positive or negative depression class based on academic, social, and demographic information.

**Note:** This project uses a synthetic dataset for educational purposes. Predictions are not medical diagnoses.

## Problem

Student depression may be associated with different factors, including academic performance, social relationships, and personal background.

The goal is to build a binary classification model that identifies patterns in student data and predicts the depression label.

## Dataset

The dataset was obtained from [Kaggle – Student Depression Detection Dataset](https://www.kaggle.com/datasets/mkhoirulmustaqim/student-depression-detection-dataset).

It contains 4,464 records and 13 columns, including:

- **Academic:** age, current semester, CGPA.
- **Social and personal:** gender, family background, relationship status, living style, friend circle.
- **Other:** cost without study, home division, health condition.
- **Target:** depression (Yes/No).

## Exploratory Data Analysis

The EDA focused on understanding the dataset, checking data quality, and identifying relationships between features and the target.

- Checked data types, missing values, duplicates, unique values, and feature distributions.
- Compared numerical features such as age, CGPA, and current semester across the two target classes.
- Used Mutual Information to estimate the relationship between individual features and the target.
- Analyzed depression rates across categories of `relationship_status`.

One interesting finding was the relationship between relationship status and the target:

- Single students: 36.37% positive cases.
- Students in a relationship: 10.84% positive cases.
- Students with a complicated relationship: 15.08% positive cases.

### Feature selection: `health_condition`

The `health_condition` feature showed an unusually strong relationship with the target:

| Health condition | Depression: No | Depression: Yes |
|---|---:|---:|
| Good | 708 | 0 |
| Normal | 1,008 | 51 |
| Abnormal | 241 | 288 |
| Critical | 0 | 382 |

All students classified as `critical` had a positive target, while all students classified as `good` had a negative target.

This raised concerns about possible target leakage or an artificially strong relationship in the synthetic dataset. Since the meaning and origin of this feature were unclear, it was excluded from model training.

## Model Training

The dataset was divided into training, validation, and test sets using a stratified 60/20/20 split.

Categorical variables were encoded using Scikit-learn's `DictVectorizer`, while numerical variables were kept as numerical values.

Four models were compared:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

Random Forest achieved the best validation ROC AUC and was selected as the final model.

| Model | Validation ROC AUC |
|---|---:|
| Logistic Regression | 0.730 |
| Decision Tree | 0.709 |
| Random Forest | 0.773 |
| XGBoost | 0.770 |

The final model uses a threshold of 0.3, selected to improve the balance between precision and recall.

## Test Results

| Metric | Score |
|---|---:|
| ROC AUC | 0.727 |
| Accuracy | 0.667 |
| Precision | 0.423 |
| Recall | 0.654 |
| F1-score | 0.514 |

Test performance was lower than validation performance, indicating that the model has limited generalization performance on unseen data.

## API

The trained model is exposed through a REST API built with FastAPI.

- `GET /`: API health check.
- `POST /predict`: receives student information and returns a prediction and probability.

Interactive API documentation is available at `/docs`.

## Run locally

Clone the repository and navigate to the project folder.

Download the dataset from Kaggle and place the CSV file in `data/`.

Install dependencies:

```bash
uv sync
```

Train the model:

```bash
uv run python train.py
```

Start the API:

```bash
uv run uvicorn app:app --reload
```

Open `http://localhost:8000/docs` to test the API.

## Run with Docker

Build the image:

```bash
docker build -t student-depression-api .
```

Run the container:

```bash
docker run --rm -p 8000:8000 student-depression-api
```

Open `http://localhost:8000/docs`.

## Project Structure

```text
student-depression-ml/
├── data/
├── models/
│   └── model.pkl
├── notebooks/
├── app.py
├── predict.py
├── train.py
├── Dockerfile
├── .dockerignore
├── pyproject.toml
└── uv.lock
```

## Technologies

Python, Pandas, Scikit-learn, XGBoost, FastAPI, Docker, and uv.