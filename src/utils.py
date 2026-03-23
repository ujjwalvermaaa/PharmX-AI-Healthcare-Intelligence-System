from pathlib import Path
from typing import Dict, Optional, Tuple, List
import joblib
import pandas as pd
import numpy as np

try:
    from tensorflow.keras.models import load_model as keras_load_model
except Exception:
    keras_load_model = None


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def models_dir() -> Path:
    return project_root() / "models"


def data_dir() -> Path:
    return project_root() / "data"


def encoded_dir() -> Path:
    return data_dir() / "encoded"


def load_pickle(filename: str):
    path = models_dir() / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing model file: {path}")
    return joblib.load(path)


def load_keras_model(filename: str):
    if keras_load_model is None:
        raise RuntimeError("TensorFlow/Keras not available to load DL model")
    path = models_dir() / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing DL model file: {path}")
    return keras_load_model(str(path), compile=False)


def get_patient_dataset() -> pd.DataFrame:
    path = encoded_dir() / "patient_encoded.csv"
    return pd.read_csv(path)


def get_hospital_dataset() -> pd.DataFrame:
    path = encoded_dir() / "hospital_encoded.csv"
    return pd.read_csv(path)


def get_pharmacy_dataset() -> pd.DataFrame:
    path = encoded_dir() / "pharmacy_encoded.csv"
    return pd.read_csv(path)


def get_outbreak_dataset() -> pd.DataFrame:
    path = encoded_dir() / "outbreak_encoded.csv"
    return pd.read_csv(path)


def feature_target_split(df: pd.DataFrame, target: str) -> Tuple[List[str], List[str]]:
    cols = list(df.columns)
    if target in cols:
        feature_cols = [c for c in cols if c != target]
        target_cols = [target]
    else:
        feature_cols = cols
        target_cols = []
    return feature_cols, target_cols


def build_feature_vector_from_inputs(
    df: pd.DataFrame,
    user_values: Dict[str, float],
    target: Optional[str] = None,
) -> pd.DataFrame:
    feature_cols, _ = feature_target_split(df, target if target else "")
    X = df[feature_cols]

    defaults = {}
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(X[col]):
            defaults[col] = float(np.nanmedian(X[col]))
        else:
            defaults[col] = 0.0

    row = {col: defaults[col] for col in feature_cols}
    for k, v in user_values.items():
        if k in row:
            row[k] = v
        else:
            one_hot_col = None
            if f"{k}_{v}" in row:
                one_hot_col = f"{k}_{v}"
            elif f"{k}_{str(v)}" in row:
                one_hot_col = f"{k}_{str(v)}"
            if one_hot_col:
                row[one_hot_col] = 1.0

    return pd.DataFrame([row], columns=feature_cols)


def scale_features(X: pd.DataFrame, scaler_filename: str) -> pd.DataFrame:
    scaler = load_pickle(scaler_filename)
    if hasattr(scaler, "feature_names_in_"):
        names = list(scaler.feature_names_in_)
        X = X.reindex(columns=names)
        for c in names:
            if c not in X.columns:
                X[c] = 0.0
    X_scaled = scaler.transform(X.values)
    return pd.DataFrame(X_scaled, columns=getattr(scaler, "feature_names_in_", X.columns))


def inverse_disease_label(y_pred: np.ndarray) -> List[str]:
    encoder = load_pickle("disease_encoder.pkl")
    return list(encoder.inverse_transform(y_pred))


def safe_predict_classifier(model_filename: str, X: pd.DataFrame) -> np.ndarray:
    model = load_pickle(model_filename)
    if hasattr(model, "feature_names_in_"):
        names = list(model.feature_names_in_)
        X = X.reindex(columns=names)
        for c in names:
            if c not in X.columns:
                X[c] = 0.0
    return model.predict(X.values)


def safe_predict_regressor(model_filename: str, X: pd.DataFrame) -> np.ndarray:
    model = load_pickle(model_filename)
    if hasattr(model, "feature_names_in_"):
        names = list(model.feature_names_in_)
        X = X.reindex(columns=names)
        for c in names:
            if c not in X.columns:
                X[c] = 0.0
    return model.predict(X.values)


def nlp_predict(texts: List[str]) -> List[str]:
    tfidf = load_pickle("tfidf.pkl")
    clf = load_pickle("nlp_model.pkl")
    le = load_pickle("label_encoder.pkl")
    X = tfidf.transform(texts)
    y = clf.predict(X)
    return list(le.inverse_transform(y))


def align_row_to_feature_names(
    df: pd.DataFrame,
    input_map: Dict[str, float],
    feature_names: List[str],
) -> pd.DataFrame:
    base = {c: 0 for c in feature_names}
    for k, v in input_map.items():
        if k in base:
            base[k] = v
        else:
            one_hot = None
            if f"{k}_{v}" in base:
                one_hot = f"{k}_{v}"
            elif f"{k}_{str(v)}" in base:
                one_hot = f"{k}_{str(v)}"
            if one_hot:
                base[one_hot] = 1
    return pd.DataFrame([base], columns=feature_names)


def training_feature_names(scaler, model, df: pd.DataFrame, target: Optional[str]) -> List[str]:
    if scaler is not None and hasattr(scaler, "feature_names_in_"):
        return list(scaler.feature_names_in_)
    if scaler is not None and hasattr(scaler, "n_features_in_"):
        cols = [c for c in df.columns if not (target and c == target)]
        return cols[: scaler.n_features_in_]
    if model is not None and hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    cols = list(df.columns)
    if target and target in cols:
        return [c for c in cols if c != target]
    return cols
