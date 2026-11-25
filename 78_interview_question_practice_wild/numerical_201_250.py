"""Numerical reference code, requiring NumPy/PyTorch/sklearn as noted."""

# BEGIN W220
import numpy as np


def triangular_solve(R, b):
    result = np.empty_like(b, dtype=float)
    for i in range(len(b) - 1, -1, -1):
        result[i] = (b[i] - R[i, i + 1 :] @ result[i + 1 :]) / R[i, i]
    return result


def add_best_regressor(X1, X2, y, qr=None):
    """X1 [N,k], X2 [N,c], y [N]; returns index, new coefficients, RSS drop."""
    Q, R = qr if qr is not None else np.linalg.qr(X1, mode="reduced")
    beta = triangular_solve(R, Q.T @ y)
    e = y - X1 @ beta
    C = Q.T @ X2
    Z = X2 - Q @ C
    denom = np.einsum("ij,ij->j", Z, Z)
    scale = np.einsum("ij,ij->j", X2, X2)
    valid = denom > np.finfo(float).eps * 100 * scale
    if not valid.any():
        return None, beta, 0.0
    numer = Z.T @ e
    drop = np.full(X2.shape[1], -np.inf)
    drop[valid] = numer[valid] ** 2 / denom[valid]
    j = int(np.argmax(drop))
    alpha = numer[j] / denom[j]
    updated = beta - triangular_solve(R, C[:, j]) * alpha
    return j, np.r_[updated, alpha], float(drop[j])


# END W220

# BEGIN W222
import torch
from torch import nn


class TwoHeadRegressor(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(features, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU()
        )
        self.head_a, self.head_b = nn.Linear(16, 1), nn.Linear(16, 1)

    def forward(self, x):
        h = self.shared(x)
        return torch.cat((self.head_a(h), self.head_b(h)), dim=1)  # [batch,2]


def fit_two_targets(x_train, y_train, epochs=300, seed=0):
    """Inputs CPU float tensors [n,d], [n,2]; original target A negative."""
    if y_train.ndim != 2 or y_train.shape[1] != 2:
        raise ValueError("expected targets [n,2]")
    if not bool((y_train[:, 0] < 0).all()) or not bool((y_train[:, 1] > 0).all()):
        raise ValueError("task A negative and task B positive required")
    torch.manual_seed(seed)
    xm, xs = x_train.mean(0), x_train.std(0, unbiased=False).clamp_min(1e-6)
    signed = y_train * y_train.new_tensor([-1.0, 1.0])
    ym, ys = signed.mean(0), signed.std(0, unbiased=False).clamp_min(1e-6)
    x, target = (x_train - xm) / xs, (signed - ym) / ys
    model = TwoHeadRegressor(x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(epochs):
        pred = model(x)
        loss = ((pred - target) ** 2).mean(0).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    return model, (xm, xs, ym, ys)


@torch.no_grad()
def predict_two_targets(model, stats, x):
    xm, xs, ym, ys = stats
    signed = model((x - xm) / xs) * ys + ym
    # Standardized linear heads are unbounded. Domain projection enforces signs.
    signed = signed.clamp_min(torch.finfo(signed.dtype).eps)
    return signed * signed.new_tensor([-1.0, 1.0])


# END W222

# BEGIN W224
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def double_descent(output_dir, seeds=30, sample_wise=False):
    """Nested Gaussian fixture; exact minimum-norm fits, ridge comparison."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    fixed, maximum, m, sigma = 80, 200, 2000, 0.5
    axis = [10, 20, 40, 60, 70, 75, 78, 79, 80, 81, 82, 85, 90, 100, 120, 160, 200]
    # Fixed truth on the first 10 coordinates: adding pure-noise features does
    # not silently change the target or renormalize the true signal.
    records = []
    for seed in range(seeds):
        g = torch.Generator().manual_seed(seed)
        d = fixed if sample_wise else maximum
        X = torch.randn((maximum, d), generator=g, dtype=torch.float64)
        Xt = torch.randn((m, d), generator=g, dtype=torch.float64)
        w = torch.zeros(d, dtype=torch.float64)
        w[:10] = 1 / np.sqrt(10)
        y = X @ w + sigma * torch.randn(maximum, generator=g, dtype=torch.float64)
        yt = Xt @ w + sigma * torch.randn(m, generator=g, dtype=torch.float64)
        for a in axis:
            n, p = (a, fixed) if sample_wise else (fixed, a)
            design, test = X[:n, :p], Xt[:, :p]
            # SVD solves both regimes and picks the minimum-Euclidean-norm fit.
            U, s, Vh = torch.linalg.svd(design, full_matrices=False)
            coeff = U.T @ y[:n]
            inv = torch.where(
                s > torch.finfo(s.dtype).eps * max(n, p) * s[0], 1 / s, 0.0
            )
            beta = Vh.T @ (inv * coeff)
            # Objective mean squared error + lam * ||w||^2.
            lam = 0.02
            ridge = Vh.T @ ((s / (s * s + n * lam)) * coeff)
            records.append(
                dict(
                    seed=seed,
                    axis=a,
                    train=float(((design @ beta - y[:n]) ** 2).mean()),
                    test=float(((test @ beta - yt) ** 2).mean()),
                    ridge=float(((test @ ridge - yt) ** 2).mean()),
                )
            )
    summary = []
    for a in axis:
        row = [r for r in records if r["axis"] == a]
        summary.append(
            {
                "axis": a,
                **{
                    k: float(np.median([r[k] for r in row]))
                    for k in ("train", "test", "ridge")
                },
                "test_q25": float(np.quantile([r["test"] for r in row], 0.25)),
                "test_q75": float(np.quantile([r["test"] for r in row], 0.75)),
            }
        )
    name = "sample_wise" if sample_wise else "parameter_wise"
    (output_dir / (name + ".json")).write_text(
        json.dumps({"records": records, "summary": summary}, indent=2)
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for k in ("train", "test", "ridge"):
        ax.plot(axis, [r[k] for r in summary], marker=".", label=k)
    ax.fill_between(
        axis,
        [r["test_q25"] for r in summary],
        [r["test_q75"] for r in summary],
        alpha=0.15,
    )
    ax.axvline(fixed, color="gray", linestyle="--", label="n=p")
    ax.set(
        xlabel="training samples n (p=80)" if sample_wise else "features p (n=80)",
        ylabel="MSE (median across seeds)",
        title=(
            "Sample-wise double descent"
            if sample_wise
            else "Parameter-wise double descent"
        ),
    )
    ax.set_yscale("symlog", linthresh=0.1)
    ax.set_ylim(bottom=0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / (name + ".png"), dpi=160)
    plt.close(fig)
    return summary


# END W224

# BEGIN W227


def _q_values(env, V, gamma):
    return np.array(
        [
            [
                sum(p * (r + gamma * V[sp]) for p, sp, r, done in env.P[s][a])
                for a in range(env.nA)
            ]
            for s in range(env.nS)
        ]
    )


def value_iteration(env, gamma=0.95, tol=1e-8, max_iterations=10000):
    V = np.zeros(env.nS, dtype=float)
    for _ in range(max_iterations):
        new = _q_values(env, V, gamma).max(axis=1)
        change = np.max(np.abs(new - V))
        V = new
        if change < tol:
            break
    else:
        raise RuntimeError("value iteration did not converge within cap")
    return V, _q_values(env, V, gamma).argmax(axis=1).astype(int)


def policy_iteration(env, gamma=0.95, tol=1e-8, max_iterations=1000):
    V, policy = np.zeros(env.nS), np.zeros(env.nS, dtype=int)
    for _ in range(max_iterations):
        # Discounted Bellman evaluation is a contraction; no sampling required.
        while True:
            q = _q_values(env, V, gamma)
            new = q[np.arange(env.nS), policy]
            delta = np.max(np.abs(new - V))
            V = new
            if delta < tol:
                break
        greedy = _q_values(env, V, gamma).argmax(axis=1).astype(int)
        if np.array_equal(greedy, policy):
            return V, greedy
        policy = greedy
    raise RuntimeError("policy iteration did not converge within cap")


# END W227

# BEGIN W229
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, f1_score, log_loss

FEATURES = ["hours_spent_reading_a", "hours_spent_reading_b", "hours_spent_reading_c"]


def compare_ctr(df):
    """Return held-out metrics; choose model and threshold ONLY on validation."""
    df = df.copy()
    if not df["click"].isin([0, 1]).all():
        raise ValueError("invalid/missing binary label: quarantine before fitting")
    print(df.dtypes.to_string())
    print(df[FEATURES].describe().to_string())
    print("missing:", df.isna().sum().to_dict(), "positive rate:", df.click.mean())
    print(df[FEATURES + ["click"]].corr().to_string())
    X, y = df[FEATURES + ["current_post_category"]], df.click.astype(int)
    Xdev, Xtest, ydev, ytest = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    Xtrain, Xval, ytrain, yval = train_test_split(
        Xdev, ydev, test_size=0.25, stratify=ydev, random_state=43
    )
    models = {
        "dummy": DummyClassifier(strategy="prior"),
        "logistic": LogisticRegression(
            max_iter=1000, class_weight="balanced", solver="liblinear"
        ),
        "forest": RandomForestClassifier(
            n_estimators=100,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
        ),
    }
    fitted, validation = {}, {}
    for name, estimator in models.items():
        prep = ColumnTransformer(
            [
                (
                    "numeric",
                    Pipeline(
                        [
                            ("impute", SimpleImputer(strategy="median")),
                            ("scale", StandardScaler()),
                        ]
                    ),
                    FEATURES,
                ),
                (
                    "category",
                    Pipeline(
                        [
                            ("impute", SimpleImputer(strategy="most_frequent")),
                            ("encode", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    ["current_post_category"],
                ),
            ]
        )
        pipe = Pipeline([("preprocess", prep), ("model", estimator)]).fit(
            Xtrain, ytrain
        )
        pv = pipe.predict_proba(Xval)[:, 1]
        thresholds = np.linspace(0.05, 0.95, 91)
        threshold = max(
            thresholds, key=lambda t: f1_score(yval, pv >= t, zero_division=0)
        )
        validation[name] = {
            "ap": average_precision_score(yval, pv),
            "threshold": float(threshold),
        }
        fitted[name] = pipe
    chosen = max(validation, key=lambda k: validation[k]["ap"])
    p = fitted[chosen].predict_proba(Xtest)[:, 1]
    return {
        "validation": validation,
        "chosen": chosen,
        "test": {
            "ap": average_precision_score(ytest, p),
            "f1": f1_score(ytest, p >= validation[chosen]["threshold"]),
            "log_loss": log_loss(ytest, p),
        },
    }


# END W229

# BEGIN W230


def logistic_step(X, y, w, b, learning_rate):
    """X [n,d], y [n], w [d], scalar b; mean BCE gradient step."""
    z = X @ w + b
    # Stable sigmoid without evaluating an overflowing unused branch.
    p = np.empty_like(z, dtype=float)
    positive = z >= 0
    p[positive] = 1 / (1 + np.exp(-z[positive]))
    ez = np.exp(z[~positive])
    p[~positive] = ez / (1 + ez)
    error = p - y
    return w - learning_rate * X.T @ error / len(y), b - learning_rate * error.mean()


# END W230
