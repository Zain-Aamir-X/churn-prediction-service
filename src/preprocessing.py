"""Preprocessing + model pipeline (single sklearn ``Pipeline`` object).

    raw DataFrame -> ColumnTransformer -> classifier

Every learned transformation (imputation medians, scaler mean/std, one-hot
categories) lives *inside* the Pipeline, so:

* fitting on the training split only means no leakage into val/test;
* the exact same fitted object is saved and used for inference, so training
  and serving can never drift apart.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42

# Select columns by dtype, not by name, so adding/renaming columns doesn't
# silently break the pipeline.
NUMERIC_DTYPES = np.number
CATEGORICAL_DTYPES = ["object", "category", "string", "bool"]


def build_numeric_pipeline() -> Pipeline:
    """Median-impute (+ missing-indicator columns) then standardize."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )


def build_categorical_pipeline() -> Pipeline:
    """Fill missing with an explicit 'missing' level then one-hot encode.

    ``handle_unknown="ignore"`` encodes categories unseen during training as
    all zeros instead of raising, so the API won't crash on new values.
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )


def build_preprocessor() -> ColumnTransformer:
    """ColumnTransformer that routes columns to the right sub-pipeline by dtype."""
    return ColumnTransformer(
        transformers=[
            (
                "num",
                build_numeric_pipeline(),
                make_column_selector(dtype_include=NUMERIC_DTYPES),
            ),
            (
                "cat",
                build_categorical_pipeline(),
                make_column_selector(dtype_include=CATEGORICAL_DTYPES),
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,  # always dense (HistGradientBoosting needs dense)
    )


def build_classifier(model_type: str = "hgb", **params: Any):
    """Return an (unfitted) classifier. ``class_weight='balanced'`` addresses the
    73.5/26.5 imbalance and favors recall on the churn class."""
    if model_type == "hgb":
        defaults = dict(
            learning_rate=0.05,
            max_iter=300,
            max_leaf_nodes=15,
            class_weight="balanced",
            early_stopping=True,
            random_state=RANDOM_STATE,
        )
        return HistGradientBoostingClassifier(**{**defaults, **params})
    if model_type == "rf":
        defaults = dict(
            n_estimators=300,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        return RandomForestClassifier(**{**defaults, **params})
    raise ValueError(f"Unknown model_type '{model_type}'. Use 'hgb' or 'rf'.")


def build_pipeline(model_type: str = "hgb", **model_params: Any) -> Pipeline:
    """Full ``preprocessor -> classifier`` pipeline. Call ``.fit`` on TRAIN data only."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", build_classifier(model_type, **model_params)),
        ]
    )


def get_feature_names(pipeline: Pipeline) -> list[str]:
    """Output feature names of the fitted preprocessor (for importance/debugging)."""
    return list(pipeline.named_steps["preprocessor"].get_feature_names_out())


def save_pipeline(pipeline: Pipeline, path: str | Path) -> Path:
    """Persist the fitted pipeline together with the sklearn version used."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "sklearn_version": sklearn.__version__}, path)
    return path


def load_pipeline(path: str | Path) -> Pipeline:
    """Load a pipeline saved with ``save_pipeline``; warn on sklearn version drift."""
    artifact = joblib.load(path)
    saved = artifact.get("sklearn_version")
    if saved != sklearn.__version__:
        warnings.warn(
            f"Pipeline was saved with scikit-learn {saved}, "
            f"but {sklearn.__version__} is installed. Results may differ.",
            stacklevel=2,
        )
    return artifact["pipeline"]
