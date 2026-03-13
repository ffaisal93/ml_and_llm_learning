"""
Common data manipulation patterns for ML interviews.
"""
import numpy as np
import pandas as pd


def fill_missing_with_train_mean(train: pd.Series, test: pd.Series) -> tuple[pd.Series, pd.Series, float]:
    """
    Fit the imputation value on training data only, then apply to both.
    """
    mean_value = float(train.mean())
    return train.fillna(mean_value), test.fillna(mean_value), mean_value


def zscore_with_train_stats(train: pd.Series, test: pd.Series) -> tuple[pd.Series, pd.Series, float, float]:
    """
    Normalize with training mean/std only.
    """
    mean_value = float(train.mean())
    std_value = float(train.std(ddof=0))
    std_value = std_value if std_value > 0 else 1.0
    return (
        (train - mean_value) / std_value,
        (test - mean_value) / std_value,
        mean_value,
        std_value,
    )


def user_level_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Example groupby aggregation by user.
    """
    return (
        df.groupby("user_id", as_index=False)
        .agg(
            total_spend=("spend", "sum"),
            avg_spend=("spend", "mean"),
            num_events=("spend", "size"),
        )
    )


def one_hot_encode_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    One-hot encode a single categorical column.
    """
    return pd.get_dummies(df, columns=[column], dummy_na=False)


def left_join_features(labels: pd.DataFrame, features: pd.DataFrame, on: str) -> pd.DataFrame:
    """
    Left join preserves the label table as the reference set.
    """
    return labels.merge(features, on=on, how="left")


if __name__ == "__main__":
    print("Data Manipulation for ML")
    print("=" * 60)

    train_df = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 3],
            "spend": [10.0, np.nan, 5.0, 20.0],
            "country": ["US", "US", "CA", "US"],
        }
    )
    test_df = pd.DataFrame(
        {
            "user_id": [4, 5],
            "spend": [np.nan, 15.0],
            "country": ["CA", "US"],
        }
    )

    filled_train, filled_test, mean_value = fill_missing_with_train_mean(
        train_df["spend"], test_df["spend"]
    )
    print(f"Training mean used for imputation: {mean_value:.4f}")
    print(f"Filled train spend: {filled_train.tolist()}")
    print(f"Filled test spend: {filled_test.tolist()}")

    z_train, z_test, mean_value, std_value = zscore_with_train_stats(filled_train, filled_test)
    print(f"\nTrain mean/std for z-score: {mean_value:.4f}, {std_value:.4f}")
    print(f"Normalized train spend: {z_train.round(4).tolist()}")
    print(f"Normalized test spend: {z_test.round(4).tolist()}")

    print("\nUser-level aggregates:")
    print(user_level_aggregates(train_df.fillna({"spend": mean_value})))

    print("\nOne-hot encoding:")
    print(one_hot_encode_column(train_df.fillna({"spend": mean_value}), "country"))

    labels = pd.DataFrame({"user_id": [1, 2, 4], "label": [1, 0, 1]})
    features = user_level_aggregates(train_df.fillna({"spend": mean_value}))
    print("\nLeft join labels with features:")
    print(left_join_features(labels, features, on="user_id"))
