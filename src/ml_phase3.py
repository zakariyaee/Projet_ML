
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import inspect
import json
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
except ImportError as exc:
    XGBClassifier = None
    XGBOOST_IMPORT_ERROR = exc
else:
    XGBOOST_IMPORT_ERROR = None
from sklearn.utils.class_weight import compute_sample_weight

TARGET = "is_potentially_hazardous"
RANDOM_STATE = 42
N_SPLITS = 5


def project_root() -> Path:
    cwd = Path.cwd().resolve()
    return cwd.parent if cwd.name == "notebooks" else cwd


def load_dataset(path: Optional[Path] = None) -> pd.DataFrame:
    root = project_root()
    return pd.read_csv(path or (root / "data" / "dataset.csv"))


def split_xy(df: pd.DataFrame, target: str = TARGET) -> Tuple[pd.DataFrame, pd.Series]:
    return df.drop(columns=[target]), df[target].astype(int)


def make_splits(df: pd.DataFrame, target: str = TARGET, random_state: int = RANDOM_STATE) -> Dict[str, Any]:
    X, y = split_xy(df, target)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.20, stratify=y_train_val, random_state=random_state
    )
    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "X_train_val": X_train_val, "y_train_val": y_train_val,
    }


def one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_features),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", one_hot_encoder())]), categorical_features),
        ]
    )


def make_xgboost(random_state: int = RANDOM_STATE) -> Any:
    if XGBClassifier is None:
        raise ImportError(
            "XGBoost n'est pas installe. Installe-le avec: "
            "conda install -c conda-forge xgboost  ou  pip install xgboost"
        ) from XGBOOST_IMPORT_ERROR
    return XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )


def estimator_catalog(random_state: int = RANDOM_STATE) -> Dict[str, Any]:
    return {
        "M1_LogisticRegression": LogisticRegression(max_iter=2000, solver="lbfgs", random_state=random_state),
        "M2_DecisionTree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, random_state=random_state),
        "M3_XGBoost": make_xgboost(random_state),
        "M4_LinearSVM": LinearSVC(C=1.0, class_weight=None, random_state=random_state, max_iter=5000),
        "M4_MLP_optional": MLPClassifier(
            hidden_layer_sizes=(48, 24), alpha=1e-4, max_iter=180, early_stopping=True, random_state=random_state
        ),
    }


def build_pipeline(model_key: str, X_reference: pd.DataFrame, estimator: Optional[Any] = None) -> Pipeline:
    est = estimator if estimator is not None else estimator_catalog()[model_key]
    return Pipeline([("preprocessor", make_preprocessor(X_reference)), ("model", clone(est))])


def model_score(estimator: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    if hasattr(estimator, "decision_function"):
        return np.asarray(estimator.decision_function(X), dtype=float)
    return estimator.predict(X)


def predict_with_threshold(estimator: Any, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
    scores = model_score(estimator, X)
    if hasattr(estimator, "predict_proba"):
        return (scores >= threshold).astype(int)
    return (scores >= 0.0).astype(int)


def metrics_from_scores(y_true: Iterable[int], scores: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    out = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, scores),
    }
    try:
        out["roc_auc"] = roc_auc_score(y_true, scores)
    except ValueError:
        out["roc_auc"] = np.nan
    return out


def resample_training(X: pd.DataFrame, y: pd.Series, strategy: str, random_state: int = RANDOM_STATE) -> Tuple[pd.DataFrame, pd.Series]:
    if strategy in {"none", "baseline"}:
        return X, y
    rng = np.random.default_rng(random_state)
    y_reset = pd.Series(y).reset_index(drop=True)
    X_reset = X.reset_index(drop=True)
    idx_majority = y_reset[y_reset == 0].index.to_numpy()
    idx_minority = y_reset[y_reset == 1].index.to_numpy()
    if strategy == "oversample":
        sampled_minority = rng.choice(idx_minority, size=len(idx_majority), replace=True)
        selected = np.concatenate([idx_majority, sampled_minority])
    elif strategy == "undersample":
        sampled_majority = rng.choice(idx_majority, size=len(idx_minority), replace=False)
        selected = np.concatenate([sampled_majority, idx_minority])
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    rng.shuffle(selected)
    return X_reset.iloc[selected].reset_index(drop=True), y_reset.iloc[selected].reset_index(drop=True)


def supports_sample_weight(estimator: Any) -> bool:
    try:
        return "sample_weight" in inspect.signature(estimator.fit).parameters
    except (TypeError, ValueError):
        return False


def set_class_weight_if_available(pipe: Pipeline) -> Pipeline:
    model = pipe.named_steps["model"]
    if hasattr(model, "class_weight"):
        pipe.set_params(model__class_weight="balanced")
    return pipe


def fit_strategy(pipe: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, strategy: str, random_state: int = RANDOM_STATE) -> Pipeline:
    pipe = clone(pipe)
    if strategy == "class_weight":
        pipe = set_class_weight_if_available(pipe)
        model = pipe.named_steps["model"]
        if supports_sample_weight(model) and not hasattr(model, "class_weight"):
            weights = compute_sample_weight(class_weight="balanced", y=y_train)
            pipe.fit(X_train, y_train, model__sample_weight=weights)
        else:
            pipe.fit(X_train, y_train)
        return pipe
    X_fit, y_fit = resample_training(X_train, y_train, strategy, random_state)
    pipe.fit(X_fit, y_fit)
    return pipe


def modeling_configurations() -> List[Dict[str, str]]:
    return [
        {"model_key": model_key, "strategy": strategy}
        for model_key in ["M1_LogisticRegression", "M2_DecisionTree", "M3_XGBoost", "M4_LinearSVM"]
        for strategy in ["baseline", "class_weight", "oversample"]
    ]


def cross_validate_configurations(X: pd.DataFrame, y: pd.Series, configs: Optional[List[Dict[str, str]]] = None, n_splits: int = N_SPLITS) -> pd.DataFrame:
    configs = configs or modeling_configurations()
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for config in configs:
        model_key, strategy = config["model_key"], config["strategy"]
        fold_rows = []
        start = time.time()
        for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
            pipe = build_pipeline(model_key, X_train)
            fitted = fit_strategy(pipe, X_train, y_train, strategy, RANDOM_STATE + fold)
            scores = model_score(fitted, X_valid)
            y_pred = predict_with_threshold(fitted, X_valid)
            m = metrics_from_scores(y_valid, scores, y_pred)
            m["fold"] = fold
            fold_rows.append(m)
        summary = pd.DataFrame(fold_rows).agg(["mean", "std"])
        rows.append({
            "model": model_key, "strategy": strategy,
            "precision_mean": summary.loc["mean", "precision"], "precision_std": summary.loc["std", "precision"],
            "recall_mean": summary.loc["mean", "recall"], "recall_std": summary.loc["std", "recall"],
            "f1_mean": summary.loc["mean", "f1"], "f1_std": summary.loc["std", "f1"],
            "pr_auc_mean": summary.loc["mean", "pr_auc"], "pr_auc_std": summary.loc["std", "pr_auc"],
            "roc_auc_mean": summary.loc["mean", "roc_auc"], "roc_auc_std": summary.loc["std", "roc_auc"],
            "fit_seconds": time.time() - start,
        })
    return pd.DataFrame(rows).sort_values(["f1_mean", "pr_auc_mean", "recall_mean"], ascending=False).reset_index(drop=True)


def evaluate_on_validation(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, modeling_results: pd.DataFrame, top_n: int = 4) -> pd.DataFrame:
    rows = []
    for _, row in modeling_results.head(top_n).iterrows():
        pipe = build_pipeline(row["model"], X_train)
        fitted = fit_strategy(pipe, X_train, y_train, row["strategy"])
        scores = model_score(fitted, X_val)
        y_pred = predict_with_threshold(fitted, X_val)
        m = metrics_from_scores(y_val, scores, y_pred)
        m.update({"model": row["model"], "strategy": row["strategy"]})
        rows.append(m)
    return pd.DataFrame(rows).sort_values(["f1", "pr_auc", "recall"], ascending=False).reset_index(drop=True)


def tune_estimator(model_key: str, strategy: str, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[Pipeline, pd.DataFrame]:
    base_pipe = build_pipeline(model_key, X_train)
    if strategy == "class_weight":
        base_pipe = set_class_weight_if_available(base_pipe)
    if model_key == "M1_LogisticRegression":
        param_grid = {"model__C": [0.1, 0.5, 1.0, 2.0, 5.0]}
        search = GridSearchCV(base_pipe, param_grid, scoring="f1", cv=5, n_jobs=-1)
    elif model_key == "M2_DecisionTree":
        param_grid = {"model__max_depth": [3, 4, 5, 6, 8, 10], "model__min_samples_leaf": [5, 10, 20, 40], "model__criterion": ["gini", "entropy"]}
        search = GridSearchCV(base_pipe, param_grid, scoring="f1", cv=5, n_jobs=-1)
    elif model_key == "M3_XGBoost":
        param_grid = {
            "model__n_estimators": [100, 200, 300],
            "model__learning_rate": [0.03, 0.05, 0.10],
            "model__max_depth": [3, 4, 5],
            "model__subsample": [0.8, 0.9, 1.0],
            "model__colsample_bytree": [0.8, 0.9, 1.0],
            "model__scale_pos_weight": [1, 5, 9],
        }
        search = RandomizedSearchCV(base_pipe, param_grid, n_iter=15, scoring="f1", cv=5, n_jobs=-1, random_state=RANDOM_STATE)
    else:
        search = GridSearchCV(base_pipe, {"model__C": [0.1, 0.5, 1.0, 2.0]}, scoring="f1", cv=5, n_jobs=-1)

    if strategy in {"baseline", "oversample", "undersample"}:
        X_fit, y_fit = resample_training(X_train, y_train, strategy)
        search.fit(X_fit, y_fit)
    else:
        model = base_pipe.named_steps["model"]
        if supports_sample_weight(model) and not hasattr(model, "class_weight"):
            weights = compute_sample_weight(class_weight="balanced", y=y_train)
            search.fit(X_train, y_train, model__sample_weight=weights)
        else:
            search.fit(X_train, y_train)
    return search.best_estimator_, pd.DataFrame(search.cv_results_).sort_values("rank_test_score")


def threshold_analysis(estimator: Any, X_valid: pd.DataFrame, y_valid: pd.Series, min_precision: float = 0.40, min_recall: float = 0.90) -> pd.DataFrame:
    scores = model_score(estimator, X_valid)
    if hasattr(estimator, "predict_proba"):
        thresholds = np.round(np.arange(0.10, 0.91, 0.01), 2)
    else:
        thresholds = np.quantile(scores, np.linspace(0.01, 0.99, 99))
    rows = []
    for threshold in thresholds:
        y_pred = (scores >= threshold).astype(int)
        m = metrics_from_scores(y_valid, scores, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_valid, y_pred, labels=[0, 1]).ravel()
        rows.append({
            "threshold": float(threshold), **m,
            "fp": int(fp), "fn": int(fn), "tp": int(tp), "tn": int(tn),
            "business_cost": int(20 * fp + 1000 * fn),
            "meets_objectives": bool(m["precision"] >= min_precision and m["recall"] >= min_recall),
        })
    return pd.DataFrame(rows).sort_values(["meets_objectives", "f1", "business_cost"], ascending=[False, False, True]).reset_index(drop=True)


def final_test_report(estimator: Any, X_test: pd.DataFrame, y_test: pd.Series, threshold: float) -> Dict[str, Any]:
    scores = model_score(estimator, X_test)
    y_pred = (scores >= threshold).astype(int)
    m = metrics_from_scores(y_test, scores, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    return {**m, "threshold": float(threshold), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
