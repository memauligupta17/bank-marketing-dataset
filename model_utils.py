"""
model_utils.py
----------------
Data + model helpers for the Bank Marketing Term Deposit Predictor.

This mirrors the preprocessing pipeline used in the project notebook
(Sample_ML_Submission_Template.ipynb):
    - One-hot encoding (drop_first=True) for:
      job, marital, education, default, housing, loan, contact, month, poutcome
    - Numeric features kept as-is: age, balance, day, duration, campaign,
      pdays, previous
    - Target: deposit (yes/no -> 1/0)
    - Final chosen model: Gradient Boosting Classifier (best CV/testing
      performance in the notebook: ~84.5% accuracy, ~85% recall)

NOTE ON DATA
------------
The original notebook loads a local file ('bank (1).csv' from the UCI Bank
Marketing dataset) that was not included with this project export. To keep
this app fully self-contained and runnable out-of-the-box, a realistic
synthetic dataset is generated here, built using the same columns, category
values, and directional relationships (e.g. longer call duration & prior
campaign success increase subscription likelihood; more campaign calls &
existing loans decrease it) that were reported in the notebook's EDA.

==> To use your REAL data, drop your bank.csv (UCI Bank Marketing format)
    into a `data/bank.csv` file and set USE_REAL_DATA_PATH below, or simply
    call `train_model(df)` with your own dataframe.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)

# If you have the real UCI 'bank.csv', put its path here and it will be used
# automatically instead of the synthetic generator.
USE_REAL_DATA_PATH = "data/bank.csv"

RANDOM_STATE = 42

CATEGORICAL_COLS = [
    "job", "marital", "education", "default",
    "housing", "loan", "contact", "month", "poutcome",
]
NUMERIC_COLS = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]

JOB_OPTIONS = [
    "management", "technician", "entrepreneur", "blue-collar", "unknown",
    "retired", "admin.", "services", "self-employed", "unemployed",
    "housemaid", "student",
]
MARITAL_OPTIONS = ["married", "single", "divorced"]
EDUCATION_OPTIONS = ["primary", "secondary", "tertiary", "unknown"]
YES_NO_OPTIONS = ["yes", "no"]
CONTACT_OPTIONS = ["cellular", "telephone", "unknown"]
MONTH_OPTIONS = ["jan", "feb", "mar", "apr", "may", "jun",
                  "jul", "aug", "sep", "oct", "nov", "dec"]
POUTCOME_OPTIONS = ["success", "failure", "other", "unknown"]


def _generate_synthetic_bank_data(n_samples: int = 6000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Generate a realistic synthetic Bank Marketing dataset that mirrors the
    structure, categories, and directional feature relationships described
    in the project notebook's EDA section."""
    rng = np.random.default_rng(seed)

    age = np.clip(rng.normal(41, 11, n_samples), 18, 95).round().astype(int)

    job = rng.choice(
        JOB_OPTIONS, n_samples,
        p=[0.21, 0.17, 0.03, 0.20, 0.01, 0.13, 0.11, 0.09, 0.03, 0.01, 0.005, 0.005],
    )
    marital = rng.choice(MARITAL_OPTIONS, n_samples, p=[0.57, 0.32, 0.11])
    education = rng.choice(EDUCATION_OPTIONS, n_samples, p=[0.15, 0.49, 0.33, 0.03])
    default = rng.choice(YES_NO_OPTIONS, n_samples, p=[0.02, 0.98])

    balance = rng.lognormal(mean=6.8, sigma=1.3, size=n_samples) - 300
    balance = np.round(balance).astype(int)
    negative_mask = rng.random(n_samples) < 0.06
    balance[negative_mask] = rng.integers(-1500, -50, size=negative_mask.sum())

    housing = rng.choice(YES_NO_OPTIONS, n_samples, p=[0.52, 0.48])
    loan = rng.choice(YES_NO_OPTIONS, n_samples, p=[0.13, 0.87])
    contact = rng.choice(CONTACT_OPTIONS, n_samples, p=[0.72, 0.07, 0.21])
    day = rng.integers(1, 29, n_samples)
    month = rng.choice(
        MONTH_OPTIONS, n_samples,
        p=[0.04, 0.04, 0.03, 0.06, 0.30, 0.14, 0.15, 0.10, 0.03, 0.03, 0.04, 0.04],
    )

    duration = np.clip(rng.gamma(shape=2.0, scale=185, size=n_samples), 5, 3800).astype(int)
    campaign = np.clip(rng.gamma(shape=1.5, scale=1.7, size=n_samples), 1, 40).astype(int)

    previous = np.zeros(n_samples, dtype=int)
    had_prev = rng.random(n_samples) < 0.30
    previous[had_prev] = rng.integers(1, 8, had_prev.sum())

    pdays = np.full(n_samples, -1, dtype=int)
    pdays[had_prev] = rng.integers(1, 872, had_prev.sum())

    poutcome = np.full(n_samples, "unknown", dtype=object)
    poutcome[~had_prev] = "unknown"
    prev_idx = np.where(had_prev)[0]
    poutcome[prev_idx] = rng.choice(
        ["success", "failure", "other"], len(prev_idx), p=[0.34, 0.50, 0.16]
    )

    # --- Build target probability using directionally realistic weights ---
    logit = -0.75
    logit += 0.0030 * (duration - 260)           # longer calls -> more likely
    logit += np.where(poutcome == "success", 2.1, 0.0)
    logit += np.where(poutcome == "failure", -0.35, 0.0)
    logit += -0.11 * campaign                     # more contacts -> less likely
    logit += np.where(housing == "yes", -0.45, 0.0)
    logit += np.where(loan == "yes", -0.40, 0.0)
    logit += np.where(default == "yes", -0.55, 0.0)
    logit += 0.00009 * balance
    logit += np.where((age < 25) | (age > 60), 0.35, 0.0)
    logit += np.where(education == "tertiary", 0.20, 0.0)
    logit += np.where(contact == "cellular", 0.25, 0.0)
    logit += np.where(contact == "unknown", -0.40, 0.0)
    logit += np.where(month == "mar", 0.5, 0.0) + np.where(month == "dec", 0.5, 0.0) \
        + np.where(month == "sep", 0.5, 0.0) + np.where(month == "oct", 0.4, 0.0)
    logit += 0.05 * np.minimum(previous, 5)
    logit += rng.normal(0, 0.55, n_samples)        # noise

    prob = 1 / (1 + np.exp(-logit))
    deposit = (rng.random(n_samples) < prob).astype(int)
    deposit_label = np.where(deposit == 1, "yes", "no")

    df = pd.DataFrame({
        "age": age, "job": job, "marital": marital, "education": education,
        "default": default, "balance": balance, "housing": housing, "loan": loan,
        "contact": contact, "day": day, "month": month, "duration": duration,
        "campaign": campaign, "pdays": pdays, "previous": previous,
        "poutcome": poutcome, "deposit": deposit_label,
    })
    return df


def load_data() -> pd.DataFrame:
    """Load the real bank.csv if present at USE_REAL_DATA_PATH, else fall
    back to the built-in synthetic generator."""
    import os
    if os.path.exists(USE_REAL_DATA_PATH):
        return pd.read_csv(USE_REAL_DATA_PATH)
    return _generate_synthetic_bank_data()


def preprocess(df: pd.DataFrame):
    """Replicates the notebook's encoding: one-hot (drop_first=True) for the
    categorical columns, binary map for the target."""
    data = df.copy()
    data["deposit"] = data["deposit"].map({"yes": 1, "no": 0})
    data = pd.get_dummies(data, columns=CATEGORICAL_COLS, drop_first=True)
    X = data.drop("deposit", axis=1)
    y = data["deposit"]
    return X, y


def train_model(df: pd.DataFrame = None):
    """Train the Gradient Boosting Classifier (the notebook's final chosen
    model) and return the fitted pipeline pieces + evaluation metrics."""
    if df is None:
        df = load_data()

    X, y = preprocess(df)
    feature_columns = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=3,
        subsample=0.9,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    test_pred = model.predict(X_test)
    test_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, test_pred),
        "precision": precision_score(y_test, test_pred),
        "recall": recall_score(y_test, test_pred),
        "f1": f1_score(y_test, test_pred),
        "roc_auc": roc_auc_score(y_test, test_proba),
        "confusion_matrix": confusion_matrix(y_test, test_pred),
    }

    fpr, tpr, _ = roc_curve(y_test, test_proba)
    metrics["roc_curve"] = (fpr, tpr)

    importances = pd.Series(model.feature_importances_, index=feature_columns)
    metrics["feature_importance"] = importances.sort_values(ascending=False)

    return model, feature_columns, metrics


def build_input_row(raw_input: dict, feature_columns: list) -> pd.DataFrame:
    """Take a dict of raw (human-entered) feature values, one-hot encode it
    the same way as training, then align columns to the trained model's
    feature_columns (filling any missing dummy columns with 0)."""
    row = pd.DataFrame([raw_input])
    row = pd.get_dummies(row, columns=CATEGORICAL_COLS, drop_first=True)
    row = row.reindex(columns=feature_columns, fill_value=0)
    return row
