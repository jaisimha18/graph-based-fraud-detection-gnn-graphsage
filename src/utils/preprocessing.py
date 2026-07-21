# ============================================================
# src/utils/preprocessing.py — Shared preprocessing utilities
# ============================================================
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


def strip_dollar(series: pd.Series) -> pd.Series:
    """Remove '$' and ',' and cast to float."""
    return series.astype(str).str.replace(r"[\$,]", "", regex=True).astype(float)


def encode_fraud(series: pd.Series) -> pd.Series:
    """'Yes' → 1, 'No' → 0."""
    return (series.str.strip().str.lower() == "yes").astype(np.int8)


def encode_chip(series: pd.Series) -> pd.Series:
    """Swipe → 0, Chip → 1, Online → 2."""
    m = {"Swipe Transaction": 0, "Chip Transaction": 1, "Online Transaction": 2}
    return series.map(m).fillna(0).astype(np.int8)


def cyclical(series: pd.Series, period: int):
    """Sine/cosine encoding for periodic features."""
    rad = 2 * np.pi * series / period
    return np.sin(rad).astype(np.float32), np.cos(rad).astype(np.float32)


def encode_errors(series: pd.Series) -> pd.DataFrame:
    """Multi-label error encoding into binary columns + has_any_error."""
    error_types = [
        "Insufficient Balance", "Bad PIN", "Technical Glitch",
        "Bad Card Number", "Bad Expiration", "Bad CVV", "Bad Zipcode"
    ]
    result = {}
    for err in error_types:
        col = "err_" + err.lower().replace(" ", "_")
        result[col] = series.astype(str).str.contains(err, na=False).astype(np.int8)
    result["has_any_error"] = (series.notna() & (series.astype(str) != "nan")).astype(np.int8)
    return pd.DataFrame(result)


def normalize_df(df: pd.DataFrame, cols: list, scaler=None):
    """Fit-transform or transform a set of columns. Returns (df, scaler)."""
    if scaler is None:
        scaler = StandardScaler()
        df[cols] = scaler.fit_transform(df[cols].astype(float))
    else:
        df[cols] = scaler.transform(df[cols].astype(float))
    return df, scaler


def safe_label_encode(series: pd.Series) -> np.ndarray:
    le = LabelEncoder()
    return le.fit_transform(series.astype(str).fillna("unknown"))
