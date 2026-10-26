"""Tests for the preprocessing pipeline using a small synthetic Telco-like dataset."""
import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from src.data_loader import (
    DataValidationError,
    load_raw_data,
    split_features_target,
    train_val_test_split,
    validate_raw_data,
)
from src.preprocessing import (
    build_pipeline,
    get_feature_names,
    load_pipeline,
    save_pipeline,
)


def make_raw_df(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tenure = rng.integers(0, 72, n)
    monthly = rng.normal(65, 25, n).clip(18, 120).round(2)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.2])

    def yn(p=0.5):
        return rng.choice(["Yes", "No"], n, p=[p, 1 - p])

    def three(extra):
        return rng.choice(["Yes", "No", extra], n)

    logit = -1.0 - 0.04 * tenure + 0.02 * monthly + 1.2 * (contract == "Month-to-month")
    churn = rng.random(n) < 1 / (1 + np.exp(-logit))
    return pd.DataFrame(
        {
            "customerID": [f"C{i:05d}" for i in range(n)],
            "gender": rng.choice(["Male", "Female"], n),
            "SeniorCitizen": rng.integers(0, 2, n),
            "Partner": yn(), "Dependents": yn(0.3),
            "tenure": tenure,
            "PhoneService": yn(0.9),
            "MultipleLines": three("No phone service"),
            "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n),
            "OnlineSecurity": three("No internet service"),
            "OnlineBackup": three("No internet service"),
            "DeviceProtection": three("No internet service"),
            "TechSupport": three("No internet service"),
            "StreamingTV": three("No internet service"),
            "StreamingMovies": three("No internet service"),
            "Contract": contract,
            "PaperlessBilling": yn(0.6),
            "PaymentMethod": rng.choice(
                ["Electronic check", "Mailed check", "Bank transfer (automatic)",
                 "Credit card (automatic)"], n),
            "MonthlyCharges": monthly,
            "TotalCharges": (tenure * monthly).astype(float),
            "Churn": np.where(churn, "Yes", "No"),
        }
    )


@pytest.fixture(scope="module")
def raw_df():
    df = make_raw_df()
    # Mimic the real dataset: TotalCharges is blank for tenure == 0 customers.
    df.loc[df["tenure"] == 0, "TotalCharges"] = np.nan
    return df


@pytest.fixture(scope="module")
def splits(raw_df):
    X, y = split_features_target(raw_df)
    return train_val_test_split(X, y)


@pytest.fixture(scope="module")
def fitted(splits):
    X_train, _, _, y_train, _, _ = splits
    return build_pipeline("hgb", max_iter=50).fit(X_train, y_train)


# ---------------------------------------------------------------- data loader
def test_split_is_stratified_and_disjoint(splits, raw_df):
    X_train, X_val, X_test, y_train, y_val, y_test = splits
    assert len(X_train) + len(X_val) + len(X_test) == len(raw_df)
    assert not set(X_train.index) & set(X_test.index)
    assert not set(X_train.index) & set(X_val.index)
    assert not set(X_val.index) & set(X_test.index)
    overall = (raw_df["Churn"] == "Yes").mean()
    for y in (y_train, y_val, y_test):
        assert abs(y.mean() - overall) < 0.02


def test_features_exclude_id_and_target(raw_df):
    X, y = split_features_target(raw_df)
    assert "customerID" not in X.columns and "Churn" not in X.columns
    assert set(y.unique()) <= {0, 1}


def test_validation_catches_bad_schema(raw_df):
    with pytest.raises(DataValidationError):
        validate_raw_data(raw_df.drop(columns=["tenure"]))
    bad = raw_df.copy()
    bad.loc[0, "Churn"] = "Maybe"
    with pytest.raises(DataValidationError):
        validate_raw_data(bad)


def test_load_raw_data_coerces_blank_total_charges(tmp_path, raw_df):
    path = tmp_path / "telco.csv"
    out = raw_df.copy()
    out["TotalCharges"] = out["TotalCharges"].astype(object)
    out.loc[out["tenure"] == 0, "TotalCharges"] = " "
    out.to_csv(path, index=False)
    loaded = load_raw_data(path)
    assert pd.api.types.is_numeric_dtype(loaded["TotalCharges"])
    assert loaded["TotalCharges"].isna().sum() == (raw_df["tenure"] == 0).sum()


# ------------------------------------------------------------------- pipeline
def test_pipeline_structure():
    pipe = build_pipeline()
    assert [name for name, _ in pipe.steps] == ["preprocessor", "classifier"]


def test_unfitted_pipeline_raises():
    with pytest.raises(NotFittedError):
        check_is_fitted(build_pipeline())


def test_predict_on_raw_input(fitted, splits):
    _, _, X_test, _, _, _ = splits
    proba = fitted.predict_proba(X_test)
    assert proba.shape == (len(X_test), 2)
    assert np.isfinite(proba).all()


def test_no_leakage_scaler_uses_train_stats_only(splits, fitted):
    X_train, _, X_test, _, _, _ = splits
    num = fitted.named_steps["preprocessor"].named_transformers_["num"]
    scaler = num.named_steps["scaler"]
    idx = list(X_train.select_dtypes("number").columns).index("tenure")
    # Fitted statistics equal the TRAIN statistics...
    assert scaler.mean_[idx] == pytest.approx(X_train["tenure"].mean())
    # ...and are not influenced by the test split.
    assert scaler.mean_[idx] != pytest.approx(
        pd.concat([X_train, X_test])["tenure"].mean()
    )


def test_imputer_median_comes_from_train(splits, fitted):
    X_train, _, _, _, _, _ = splits
    imputer = fitted.named_steps["preprocessor"].named_transformers_["num"].named_steps["imputer"]
    cols = list(X_train.select_dtypes("number").columns)
    idx = cols.index("TotalCharges")
    assert imputer.statistics_[idx] == pytest.approx(X_train["TotalCharges"].median())


def test_unknown_category_does_not_crash(fitted, splits):
    _, _, X_test, _, _, _ = splits
    row = X_test.iloc[[0]].copy()
    row["Contract"] = "Five year"            # never seen in training
    row["PaymentMethod"] = "Cryptocurrency"  # never seen in training
    proba = fitted.predict_proba(row)
    assert proba.shape == (1, 2) and np.isfinite(proba).all()


def test_missing_values_are_handled(fitted, splits):
    _, _, X_test, _, _, _ = splits
    row = X_test.iloc[[0]].copy()
    row["TotalCharges"] = np.nan
    row["MonthlyCharges"] = np.nan
    row["InternetService"] = None
    assert np.isfinite(fitted.predict_proba(row)).all()


def test_missing_indicator_columns_are_created(fitted):
    names = get_feature_names(fitted)
    assert any("missingindicator" in n for n in names)  # TotalCharges had NaNs in train


def test_column_order_does_not_matter(fitted, splits):
    _, _, X_test, _, _, _ = splits
    shuffled = X_test[X_test.columns[::-1]]
    np.testing.assert_allclose(
        fitted.predict_proba(X_test), fitted.predict_proba(shuffled)
    )


def test_extra_column_is_ignored(fitted, splits):
    _, _, X_test, _, _, _ = splits
    extra = X_test.assign(brand_new_col="x")
    np.testing.assert_allclose(
        fitted.predict_proba(X_test), fitted.predict_proba(extra)
    )


def test_save_and_load_roundtrip(fitted, splits, tmp_path):
    _, _, X_test, _, _, _ = splits
    path = save_pipeline(fitted, tmp_path / "pipeline.joblib")
    reloaded = load_pipeline(path)
    np.testing.assert_allclose(
        fitted.predict_proba(X_test), reloaded.predict_proba(X_test)
    )


@pytest.mark.parametrize("model_type", ["hgb", "rf"])
def test_both_classifiers_fit(splits, model_type):
    X_train, _, X_test, y_train, _, _ = splits
    params = {"max_iter": 20} if model_type == "hgb" else {"n_estimators": 20}
    pipe = build_pipeline(model_type, **params).fit(X_train, y_train)
    assert pipe.predict(X_test).shape == (len(X_test),)


def test_invalid_model_type():
    with pytest.raises(ValueError):
        build_pipeline("svm")
