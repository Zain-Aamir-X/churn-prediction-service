"""Load and validate the raw IBM Telco Customer Churn data.

This module only does *stateless* cleaning (things that are true regardless of
the data split, e.g. fixing a malformed column type). Anything that learns
from the data (imputation values, scaling, category lists) lives in the
sklearn Pipeline in ``preprocessing.py`` so it is fit on training data only.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

TARGET_COL = "Churn"
ID_COL = "customerID"
RANDOM_STATE = 42

# Columns that must exist for the pipeline to make sense.
REQUIRED_COLUMNS = [
    ID_COL, "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges", TARGET_COL,
]


class DataValidationError(ValueError):
    """Raised when the raw data does not match the expected schema."""


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Read the raw CSV and apply stateless cleaning.

    - ``TotalCharges`` is stored as text in the source file and contains blank
      strings (" ") for customers with tenure == 0. We coerce to numeric so the
      blanks become NaN, which the pipeline's imputer then handles.
    - Strips whitespace from column names and string values.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found: {path}")

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    # Strip whitespace in text columns (keeps NaN as NaN).
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].str.strip()

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    validate_raw_data(df)
    return df


def validate_raw_data(df: pd.DataFrame) -> None:
    """Fail fast on schema problems. Raises ``DataValidationError``."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")

    if df.empty:
        raise DataValidationError("Dataset is empty.")

    if df[ID_COL].duplicated().any():
        n = int(df[ID_COL].duplicated().sum())
        raise DataValidationError(f"{n} duplicated {ID_COL} values found.")

    if df[TARGET_COL].isna().any():
        raise DataValidationError(f"{TARGET_COL} contains missing values.")

    bad_target = set(df[TARGET_COL].unique()) - {"Yes", "No"}
    if bad_target:
        raise DataValidationError(
            f"Unexpected values in {TARGET_COL}: {sorted(bad_target)}"
        )


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Drop the identifier, encode the target as 1 (churn) / 0 (stay)."""
    y = (df[TARGET_COL] == "Yes").astype(int)
    X = df.drop(columns=[TARGET_COL, ID_COL])
    return X, y


def train_val_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = RANDOM_STATE,
):
    """Stratified train / validation / test split.

    ``val_size`` is the fraction of the *total* data used for validation, so the
    default gives 60 / 20 / 20. The validation set is used to tune the decision
    threshold; the test set is touched only once, for final evaluation.
    """
    if not 0 < test_size + val_size < 1:
        raise ValueError("test_size + val_size must be between 0 and 1.")

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    relative_val = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=relative_val,
        stratify=y_trainval, random_state=random_state,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
