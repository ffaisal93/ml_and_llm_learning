import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


def prepare(prices, predictors=None):
    # Chosen schema: UTC-convertible timestamp and positive close for ONE asset.
    p = prices.copy()
    p["timestamp"] = pd.to_datetime(p.timestamp, utc=True, errors="raise")
    p = p.sort_values("timestamp")
    if p.timestamp.duplicated().any():
        raise ValueError("conflicting bar timestamps require resolution")
    p = p.set_index("timestamp")
    if (p.close <= 0).any():
        raise ValueError("positive prices required")
    # Daily UTC is only a demonstration calendar. Production uses market sessions.
    p = p[["close"]].resample("1D").last().dropna()
    r = np.log(p.close).diff()
    # Decision is just AFTER this close, execution assumptions discussed in answer.
    p["return_now"] = r
    p["momentum_5"] = r.rolling(5, min_periods=5).sum()
    p["volatility_5"] = r.rolling(5, min_periods=5).std()
    p["target"] = np.log(p.close.shift(-1) / p.close)
    p["label_end"] = p.index.to_series().shift(-1)
    if predictors is not None:
        q = predictors.copy()
        q["available_at"] = pd.to_datetime(q.available_at, utc=True, errors="raise")
        q = q.sort_values("available_at")
        p = pd.merge_asof(
            p.reset_index().sort_values("timestamp"),
            q,
            on=None,
            left_on="timestamp",
            right_on="available_at",
            direction="backward",
        ).set_index("timestamp")
    return p


def evaluate(table, min_train=30, test_size=10):
    features = ["return_now", "momentum_5", "volatility_5"]
    if "signal" in table:
        features += ["signal"]
    t = table.dropna(subset=features + ["target", "label_end"])
    results = []
    for start in range(min_train, len(t), test_size):
        test = t.iloc[start : start + test_size]
        cutoff = test.index[0]
        # A label ending at cutoff is observable after that close; exclude equality
        # conservatively to avoid overlapping the new decision boundary.
        train = t.iloc[:start]
        train = train[train.label_end < cutoff]
        model = make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0)
        )
        model.fit(train[features], train.target)
        pred = model.predict(test[features])
        y = test.target.to_numpy()
        results.append(
            {
                "test_start": str(cutoff),
                "n": len(y),
                "model_mse": mean_squared_error(y, pred),
                "zero_mse": float(np.mean(y * y)),
                "train_mean_mse": float(np.mean((y - train.target.mean()) ** 2)),
            }
        )
    return pd.DataFrame(results)


if __name__ == "__main__":
    rng = np.random.default_rng(6)
    dates = pd.date_range("2026-01-01", periods=100, tz="UTC")
    prices = pd.DataFrame(
        {"timestamp": dates, "close": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 100)))}
    )
    t = prepare(prices)
    stats = evaluate(t)
    assert np.isclose(
        t.iloc[0].target, np.log(prices.close.iloc[1] / prices.close.iloc[0])
    )
    changed = prices.copy()
    changed.loc[80:, "close"] *= 2
    assert np.allclose(
        prepare(changed).iloc[:79][["return_now", "momentum_5", "volatility_5"]],
        t.iloc[:79][["return_now", "momentum_5", "volatility_5"]],
        equal_nan=True,
    )
    q = pd.DataFrame(
        {"available_at": [dates[20] + pd.Timedelta(hours=1)], "signal": [99.0]}
    )
    joined = prepare(prices, q)
    assert (
        pd.isna(joined.loc[dates[20], "signal"])
        and joined.loc[dates[21], "signal"] == 99
    )
    print(
        "SYNTHETIC no-signal fixture; not claimed investment performance:\n",
        stats.to_string(index=False),
    )
