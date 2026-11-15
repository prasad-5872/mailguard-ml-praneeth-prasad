"""Model definitions, the reusable ``SpamModel`` artifact, explanations and artifact I/O."""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.special import expit
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, StandardScaler
from sklearn.svm import LinearSVC

from src.features import FEATURE_DESCRIPTIONS, STRUCTURED_FEATURES, TEXT_COLUMN

ARTIFACT_SCHEMA_VERSION = 1


class ModelLoadError(RuntimeError):
    """Raised when the model artifact is missing, corrupt or incompatible."""


# --------------------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PreprocessorConfig:
    word_max_features: int = 50_000
    char_max_features: int = 50_000
    word_min_df: int = 2
    char_min_df: int = 3
    struct_scaling: str = "standard"  # "standard" | "minmax" | "none"


def build_preprocessor(config: PreprocessorConfig) -> ColumnTransformer:
    """Word TF-IDF (1-2 grams) + char TF-IDF (3-5 grams, word-bounded) + scaled structured features."""
    word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=config.word_min_df,
        max_df=0.9,
        max_features=config.word_max_features,
        sublinear_tf=True,
        strip_accents="unicode",
        token_pattern=r"(?u)\b\w\w+\b",
        dtype=np.float32,
    )
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=config.char_min_df,
        max_df=0.95,
        max_features=config.char_max_features,
        sublinear_tf=True,
        dtype=np.float32,
    )
    # Structured counts are heavy-tailed (e.g. hundreds of <img> tags); log1p compresses the
    # tails before scaling so a single extreme value cannot dominate a linear model's score.
    log1p = FunctionTransformer(np.log1p, feature_names_out="one-to-one")
    if config.struct_scaling == "standard":
        struct: Any = Pipeline([("log1p", log1p), ("scale", StandardScaler())])
    elif config.struct_scaling == "minmax":
        # Multinomial NB needs non-negative inputs; clip keeps unseen extremes in [0, 1].
        struct = Pipeline([("log1p", log1p), ("scale", MinMaxScaler(clip=True))])
    elif config.struct_scaling == "none":
        struct = "passthrough"
    else:
        raise ValueError(f"unknown struct_scaling: {config.struct_scaling}")
    return ColumnTransformer(
        [("word", word, TEXT_COLUMN), ("char", char, TEXT_COLUMN), ("struct", struct, list(STRUCTURED_FEATURES))],
        sparse_threshold=1.0,
    )


def compact_preprocessor(preprocessor: ColumnTransformer) -> None:
    """Drop large fit-time attributes that are not needed for inference (shrinks the artifact)."""
    for _, transformer, _ in preprocessor.transformers_:
        if isinstance(transformer, TfidfVectorizer) and hasattr(transformer, "stop_words_"):
            try:
                delattr(transformer, "stop_words_")
            except AttributeError:
                pass


# --------------------------------------------------------------------------------------
# Candidate models
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    display_name: str
    preprocessor: PreprocessorConfig
    param_grid: tuple[dict[str, Any], ...]
    factory: Callable[[dict[str, Any], int], Any]
    calibrate: bool  # Platt-scale scores on the validation split
    explainer: str  # "linear" | "tree"
    description: str = ""


def _make_nb(params: dict[str, Any], seed: int) -> MultinomialNB:
    return MultinomialNB(alpha=params["alpha"])


def _make_lr(params: dict[str, Any], seed: int) -> LogisticRegression:
    return LogisticRegression(C=params["C"], solver="liblinear", class_weight="balanced", max_iter=5000, random_state=seed)


def _make_svm(params: dict[str, Any], seed: int) -> LinearSVC:
    return LinearSVC(C=params["C"], class_weight="balanced", max_iter=20_000, random_state=seed)


def _make_lgbm(params: dict[str, Any], seed: int):
    from lightgbm import LGBMClassifier  # imported lazily: only needed for this candidate

    return LGBMClassifier(
        n_estimators=params.get("n_estimators", 400),
        learning_rate=params.get("learning_rate", 0.05),
        num_leaves=params.get("num_leaves", 31),
        min_child_samples=params.get("min_child_samples", 10),
        colsample_bytree=params.get("colsample_bytree", 0.5),
        subsample=0.8,
        subsample_freq=1,
        reg_lambda=1.0,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )


def default_candidates(small: bool = False) -> list[CandidateSpec]:
    """Candidate models compared during training. ``small=True`` shrinks vocabularies (used by tests)."""
    if small:
        lin = PreprocessorConfig(2_000, 2_000, 1, 1, "standard")
        nb = PreprocessorConfig(2_000, 2_000, 1, 1, "minmax")
        tree = PreprocessorConfig(500, 500, 1, 1, "none")
    else:
        lin = PreprocessorConfig(50_000, 50_000, 2, 3, "standard")
        nb = PreprocessorConfig(50_000, 50_000, 2, 3, "minmax")
        tree = PreprocessorConfig(10_000, 10_000, 3, 5, "none")
    lgbm_grid = (
        ({"n_estimators": 30, "num_leaves": 7, "min_child_samples": 2},)
        if small
        else ({"num_leaves": 15}, {"num_leaves": 31})
    )
    return [
        CandidateSpec(
            "naive_bayes", "Multinomial Naive Bayes", nb,
            ({"alpha": 0.01}, {"alpha": 0.1}, {"alpha": 0.5}), _make_nb, calibrate=True, explainer="linear",
            description="Classic probabilistic text baseline; raw probabilities are over-confident, so scores are Platt-calibrated.",
        ),
        CandidateSpec(
            "logistic_regression", "Logistic Regression", lin,
            ({"C": 1.0}, {"C": 10.0}, {"C": 30.0}), _make_lr, calibrate=False, explainer="linear",
            description="L2-regularized linear model; natively outputs log-odds and probabilities.",
        ),
        CandidateSpec(
            "linear_svm", "Linear SVM", lin,
            ({"C": 0.1}, {"C": 0.5}, {"C": 1.0}), _make_svm, calibrate=True, explainer="linear",
            description="Max-margin linear classifier; margins are not probabilities, so they are Platt-calibrated.",
        ),
        CandidateSpec(
            "lightgbm", "LightGBM", tree, lgbm_grid, _make_lgbm, calibrate=False, explainer="tree",
            description="Gradient-boosted trees on a reduced TF-IDF vocabulary plus raw structured features.",
        ),
    ]


# --------------------------------------------------------------------------------------
# Reusable artifact
# --------------------------------------------------------------------------------------


@dataclass
class Contribution:
    feature: str
    kind: str  # "word" | "char" | "structured"
    contribution: float
    value: float | None = None
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "kind": self.kind,
            "contribution": self.contribution,
            "value": self.value,
            "description": self.description,
            "direction": "spam" if self.contribution > 0 else "ham",
        }


@dataclass
class SpamModel:
    """Self-contained trained model: preprocessing + estimator + optional calibrator + threshold."""

    name: str
    display_name: str
    preprocessor: ColumnTransformer
    estimator: Any
    explainer: str
    threshold: float
    calibrator: LogisticRegression | None = None
    version: str = "0.0.0"
    params: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    feature_names: np.ndarray | None = None
    structured_features: tuple[str, ...] = STRUCTURED_FEATURES
    schema_version: int = ARTIFACT_SCHEMA_VERSION

    # -- core inference ------------------------------------------------------------------

    def transform(self, frame: pd.DataFrame) -> sp.csr_matrix:
        missing = [c for c in (TEXT_COLUMN, *self.structured_features) if c not in frame.columns]
        if missing:
            raise ValueError(f"input frame is missing columns: {missing}")
        return sp.csr_matrix(self.preprocessor.transform(frame), dtype=np.float32)

    def _scores_from_matrix(self, X: sp.csr_matrix) -> np.ndarray:
        est = self.estimator
        if isinstance(est, MultinomialNB):
            jll = est.predict_joint_log_proba(X)
            return jll[:, 1] - jll[:, 0]
        if self.explainer == "tree":
            return np.asarray(est.predict(X, raw_score=True), dtype=float)
        return np.asarray(est.decision_function(X), dtype=float)

    def decision_scores(self, frame: pd.DataFrame) -> np.ndarray:
        """Uncalibrated spam scores (log-odds for LR/LightGBM/NB, margin for SVM)."""
        return self._scores_from_matrix(self.transform(frame))

    def scores_to_proba(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=float)
        if self.calibrator is not None:
            return self.calibrator.predict_proba(scores.reshape(-1, 1))[:, 1]
        return expit(scores)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """Spam probability for each row."""
        return self.scores_to_proba(self.decision_scores(frame))

    def predict(self, frame: pd.DataFrame, threshold: float | None = None) -> np.ndarray:
        t = self.threshold if threshold is None else threshold
        return (self.predict_proba(frame) >= t).astype(int)

    # -- explanations --------------------------------------------------------------------

    def _names(self) -> np.ndarray:
        if self.feature_names is None:
            self.feature_names = np.asarray(self.preprocessor.get_feature_names_out(), dtype=object)
        return self.feature_names

    def linear_weights(self) -> tuple[np.ndarray, float]:
        """Per-feature weights in score space and the intercept (linear explainers only)."""
        est = self.estimator
        if isinstance(est, MultinomialNB):
            weights = est.feature_log_prob_[1] - est.feature_log_prob_[0]
            intercept = float(est.class_log_prior_[1] - est.class_log_prior_[0])
            return np.asarray(weights, dtype=float), intercept
        coef = est.coef_
        coef = coef.toarray() if sp.issparse(coef) else np.asarray(coef)
        return coef.ravel().astype(float), float(np.ravel(est.intercept_)[0])

    def explain(self, frame: pd.DataFrame, top_k: int = 12) -> dict[str, Any]:
        """Explain the first row of ``frame``.

        * Linear models: contribution_i = weight_i * x_i over the features present in
          the email (exact additive decomposition of the score minus the intercept).
          For Naive Bayes, weight_i is the per-feature log-likelihood ratio.
        * LightGBM: exact TreeSHAP values from LightGBM's ``pred_contrib``.
        Contributions are in score (log-odds / margin) units, positive = towards spam.
        """
        X = self.transform(frame.iloc[:1])
        names = self._names()
        raw_row = frame.iloc[0]

        if self.explainer == "tree":
            contrib = self.estimator.predict(X, pred_contrib=True)
            contrib = contrib.toarray() if sp.issparse(contrib) else np.asarray(contrib)
            contrib = contrib.ravel()
            base_value, values = float(contrib[-1]), contrib[:-1]
            method = "TreeSHAP (LightGBM pred_contrib)"
        else:
            weights, base_value = self.linear_weights()
            row = X.getrow(0)
            values = np.zeros(X.shape[1], dtype=float)
            values[row.indices] = weights[row.indices] * row.data
            method = "Naive Bayes log-likelihood ratio x feature value" if isinstance(
                self.estimator, MultinomialNB
            ) else "Linear coefficient x feature value"

        nonzero = np.flatnonzero(values)
        order = nonzero[np.argsort(-np.abs(values[nonzero]))]
        x_row = X.getrow(0).toarray().ravel()
        contributions: list[dict[str, Any]] = []
        for idx in order[:top_k]:
            contributions.append(
                self._contribution(names[idx], float(values[idx]), raw_row, float(x_row[idx])).to_dict()
            )

        by_kind: dict[str, float] = {}
        for idx in nonzero:
            kind = _split_name(names[idx])[0]
            by_kind[kind] = by_kind.get(kind, 0.0) + float(values[idx])

        score = float(self._scores_from_matrix(X)[0])
        return {
            "method": method,
            "score": score,
            "base_value": base_value,
            "contributions": contributions,
            "contribution_by_kind": by_kind,
        }

    def _contribution(self, full_name: str, value: float, raw_row: pd.Series, x_value: float) -> Contribution:
        kind, feat = _split_name(full_name)
        if kind == "structured":
            return Contribution(feat, kind, value, float(raw_row.get(feat, np.nan)), FEATURE_DESCRIPTIONS.get(feat, ""))
        label = f"'{feat}'" if kind == "char" else feat
        what = "character n-gram" if kind == "char" else "word / bigram"
        # Tree models can attribute weight to text that is ABSENT (e.g. no quoted-reply '>' lines).
        status = f"present (tf-idf {x_value:.3f})" if x_value else "absent from this email"
        return Contribution(label, kind, value, x_value, f"{what}, {status}")

    def global_importance(self, top_k: int = 20) -> dict[str, list[dict[str, Any]]]:
        """Most influential features overall (coefficients or LightGBM gain)."""
        names = self._names()
        if self.explainer == "tree":
            gain = np.asarray(self.estimator.booster_.feature_importance(importance_type="gain"), dtype=float)
            order = np.argsort(-gain)[:top_k]
            return {
                "importance": [
                    {"feature": _display_name(names[i]), "kind": _split_name(names[i])[0], "gain": float(gain[i])}
                    for i in order if gain[i] > 0
                ]
            }
        weights, _ = self.linear_weights()
        kinds = np.array([_split_name(n)[0] for n in names])
        result: dict[str, list[dict[str, Any]]] = {}
        mask = kinds != "char"  # words + structured are the most readable
        idx = np.flatnonzero(mask)
        order_pos = idx[np.argsort(-weights[idx])][:top_k]
        order_neg = idx[np.argsort(weights[idx])][:top_k]
        result["spam_indicators"] = [
            {"feature": _display_name(names[i]), "kind": kinds[i], "weight": float(weights[i])} for i in order_pos
        ]
        result["ham_indicators"] = [
            {"feature": _display_name(names[i]), "kind": kinds[i], "weight": float(weights[i])} for i in order_neg
        ]
        return result


def _split_name(full_name: str) -> tuple[str, str]:
    prefix, _, feat = str(full_name).partition("__")
    kind = {"word": "word", "char": "char", "struct": "structured"}.get(prefix, prefix)
    return kind, feat


def _display_name(full_name: str) -> str:
    """Human-readable feature label; char n-grams are quoted so leading/trailing spaces stay visible."""
    kind, feat = _split_name(full_name)
    return f"'{feat}'" if kind == "char" else feat


# --------------------------------------------------------------------------------------
# Fitting helpers
# --------------------------------------------------------------------------------------


def fit_platt_calibrator(scores: np.ndarray, y: np.ndarray) -> LogisticRegression:
    """Platt scaling: logistic regression on the 1-D decision score."""
    calibrator = LogisticRegression(C=1e4, solver="lbfgs", max_iter=1000)
    calibrator.fit(np.asarray(scores, dtype=float).reshape(-1, 1), np.asarray(y, dtype=int))
    return calibrator


# --------------------------------------------------------------------------------------
# Artifact I/O
# --------------------------------------------------------------------------------------


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_model(model: SpamModel, path: Path) -> dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    compact_preprocessor(model.preprocessor)
    model._names()  # cache feature names inside the artifact
    joblib.dump(model, path, compress=3)
    return {"path": path.name, "sha256": file_sha256(path), "size_bytes": path.stat().st_size}


def load_model(path: Path, expected_sha256: str | None = None) -> SpamModel:
    """Load and sanity-check a model artifact. Raises ``ModelLoadError`` with a readable message.

    Security: joblib/pickle files can execute code when loaded. Only load artifacts
    you produced yourself (the sha256 recorded in the model card is verified when given).
    """
    path = Path(path)
    if not path.is_file():
        raise ModelLoadError(
            f"Model artifact not found at '{path}'. Generate it with: python -m training.train"
        )
    if expected_sha256:
        actual = file_sha256(path)
        if actual != expected_sha256:
            raise ModelLoadError(
                f"Model artifact checksum mismatch (expected {expected_sha256[:12]}..., got {actual[:12]}...). "
                "The file may be corrupt or out of sync with model_card.json; retrain or restore both files."
            )
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            obj = joblib.load(path)
    except Exception as exc:
        raise ModelLoadError(f"Could not load model artifact '{path.name}': {type(exc).__name__}: {exc}") from exc
    if not isinstance(obj, SpamModel):
        raise ModelLoadError(f"Artifact '{path.name}' does not contain a SpamModel (found {type(obj).__name__}).")
    if obj.schema_version != ARTIFACT_SCHEMA_VERSION:
        raise ModelLoadError(
            f"Artifact schema version {obj.schema_version} is not supported (expected {ARTIFACT_SCHEMA_VERSION})."
        )
    if tuple(obj.structured_features) != STRUCTURED_FEATURES:
        raise ModelLoadError("Artifact was trained with a different structured feature set; retrain the model.")
    obj.metadata = dict(obj.metadata)
    obj.metadata["load_warnings"] = sorted({str(w.message)[:300] for w in caught})
    return obj
