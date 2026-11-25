"""Usage: python3 w102.py data.csv --height-unit cm --out results
No input file: runs a clearly synthetic demonstration, not clinical evidence.
Requires numpy pandas scikit-learn matplotlib seaborn.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from sklearn.inspection import permutation_importance


def fixture(n=600, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "age": rng.uniform(25, 85, n),
            "weight": rng.normal(80, 14, n),
            "height": rng.normal(170, 10, n),
            "smoker": rng.choice(["yes", "no"], n),
        }
    )
    z = (df.age - 55) / 14 + (df.weight - 80) / 30 + (df.smoker == "yes") * 0.6
    df["target"] = rng.binomial(1, 1 / (1 + np.exp(-z)))
    df.loc[::17, "weight"] = np.nan
    return pd.concat([df, df.iloc[:3]], ignore_index=True)


def run(df, height_unit, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if "target" not in df or df.target.isna().any() or not df.target.isin([0, 1]).all():
        raise ValueError("target must contain only nonmissing 0/1 labels")
    if not {"age", "weight", "height"} <= set(df.columns):
        raise ValueError("age, weight (kg), height are required; map names first")
    print("missing counts before cleaning:\n", df.isna().sum().to_string())
    print("exact duplicate rows:", df.duplicated().sum())
    # Actual patient IDs, timestamps and post-outcome columns must be removed or used
    # to define an appropriate grouped/temporal split BEFORE calling this function.
    df = df.drop_duplicates().copy()
    for col in ["age", "weight", "height"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    hm = df.height / (100 if height_unit == "cm" else 1)
    # These are declared broad plausibility rules, not automatic statistical clipping.
    bad = ~df.age.between(0, 120) | ~df.weight.between(2, 500) | ~hm.between(0.4, 2.6)
    print("rows flagged by broad plausibility checks:", int(bad.sum()))
    df.loc[~df.age.between(0, 120), "age"] = np.nan
    df.loc[~df.weight.between(2, 500), "weight"] = np.nan
    hm = hm.where(hm.between(0.4, 2.6))
    df["height"] = hm  # Explicit meters from this point onward.
    df["BMI"] = df.weight / hm**2
    y = df.pop("target").astype(int)
    Xtr, Xte, ytr, yte = train_test_split(
        df, y, test_size=0.25, random_state=42, stratify=y
    )
    print(
        "training label ratios:\n",
        ytr.value_counts(normalize=True).sort_index().to_string(),
    )
    eda = Xtr.assign(target=ytr)
    sns.countplot(data=eda, x="target")
    plt.savefig(out / "labels.png")
    plt.close()
    for col in ["age", "weight", "height", "BMI"]:
        sns.histplot(
            data=eda,
            x=col,
            hue="target",
            stat="density",
            common_norm=False,
            element="step",
        )
        plt.savefig(out / f"{col}.png")
        plt.close()
    corr = eda.select_dtypes(include="number").corr()
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.tight_layout()
    plt.savefig(out / "correlations.png")
    plt.close()
    print(
        "training target correlations (association, not causality):\n",
        corr.target.drop("target").sort_values(key=abs, ascending=False).to_string(),
    )
    num = Xtr.select_dtypes(include="number").columns.tolist()
    cat = [c for c in Xtr if c not in num]
    prep = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        (
                            "impute",
                            SimpleImputer(strategy="median", add_indicator=True),
                        ),
                        ("scale", StandardScaler()),
                    ]
                ),
                num,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat,
            ),
        ]
    )
    model = Pipeline([("prep", prep), ("model", LogisticRegression(max_iter=2000))])
    model.fit(Xtr, ytr)
    dummy = DummyClassifier(strategy="prior").fit(Xtr, ytr)
    for name, m in [("prior baseline", dummy), ("logistic", model)]:
        prob = m.predict_proba(Xte)[:, 1]
        pred = prob >= 0.5
        print(
            name,
            dict(
                accuracy=accuracy_score(yte, pred),
                precision=precision_score(yte, pred, zero_division=0),
                recall=recall_score(yte, pred, zero_division=0),
                roc_auc=roc_auc_score(yte, prob),
                pr_auc=average_precision_score(yte, prob),
            ),
        )
        print(
            "confusion matrix [true rows, predicted columns]:\n",
            confusion_matrix(yte, pred),
        )
    imp = permutation_importance(
        model, Xte, yte, scoring="roc_auc", n_repeats=10, random_state=42
    )
    importance = pd.DataFrame(
        {
            "feature": Xte.columns,
            "auc_drop": imp.importances_mean,
            "std": imp.importances_std,
        }
    ).sort_values("auc_drop", ascending=False)
    print(
        "top 3 features by held-out permutation:\n",
        importance.head(3).to_string(index=False),
    )
    importance.head(10).set_index("feature").auc_drop.sort_values().plot.barh(
        xerr=importance.head(10).sort_values("auc_drop")["std"]
    )
    plt.xlabel("ROC-AUC decrease after permutation")
    plt.tight_layout()
    plt.savefig(out / "importance.png")
    plt.close()
    assert np.isfinite(model.predict_proba(Xte)).all()
    assert len(Xtr) + len(Xte) == len(df)
    return model, importance


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("csv", nargs="?")
    p.add_argument("--height-unit", choices=["cm", "m"], required=True)
    p.add_argument("--out", default="w102_results")
    a = p.parse_args()
    print(
        "ACTUAL CSV"
        if a.csv
        else "SYNTHETIC DEMONSTRATION: no patient dataset supplied"
    )
    run(pd.read_csv(a.csv) if a.csv else fixture(), a.height_unit, a.out)
