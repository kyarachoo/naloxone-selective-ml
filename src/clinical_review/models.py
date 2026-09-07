from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class ModelBundle:
    name: str
    pipeline: Pipeline


def make_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipe, numeric),
            ("cat", categorical_pipe, categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_models(numeric: list[str], categorical: list[str], seed: int = 42) -> list[ModelBundle]:
    logistic = Pipeline(
        [
            ("preprocess", make_preprocessor(numeric, categorical)),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    random_state=seed,
                ),
            ),
        ]
    )

    nonlinear = Pipeline(
        [
            ("preprocess", make_preprocessor(numeric, categorical)),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.06,
                    max_iter=180,
                    max_leaf_nodes=15,
                    l2_regularization=0.2,
                    random_state=seed,
                ),
            ),
        ]
    )

    return [
        ModelBundle("logistic_regression", logistic),
        ModelBundle("hist_gradient_boosting", nonlinear),
    ]


def transformed_feature_names(pipeline: Pipeline) -> list[str]:
    pre = pipeline.named_steps["preprocess"]
    return pre.get_feature_names_out().tolist()


def logistic_coefficients(pipeline: Pipeline) -> pd.DataFrame:
    model = pipeline.named_steps["model"]
    if not isinstance(model, LogisticRegression):
        raise TypeError("Coefficient extraction is only defined for logistic regression")
    names = transformed_feature_names(pipeline)
    coefs = model.coef_[0]
    return pd.DataFrame({"feature": names, "coefficient": coefs}).sort_values(
        "coefficient", key=lambda s: s.abs(), ascending=False
    )
