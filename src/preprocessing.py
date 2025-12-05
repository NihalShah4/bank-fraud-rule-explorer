import numpy as np
import pandas as pd


def load_and_preprocess(csv_path: str) -> pd.DataFrame:
    """
    Load the synthetic transactions and create features for modeling.

    Steps:
    - Parse timestamp
    - Create time-based features
    - Create customer-level daily aggregates
    - One-hot encode categorical variables
    """
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])

    # Sort for any time-based calculations
    df = df.sort_values(["customer_id", "timestamp"]).reset_index(drop=True)

    # ----- Time features -----
    df["tx_date"] = df["timestamp"].dt.date
    df["tx_hour"] = df["timestamp"].dt.hour
    df["tx_dayofweek"] = df["timestamp"].dt.dayofweek  # 0=Mon
    df["tx_is_weekend"] = df["tx_dayofweek"].isin([5, 6]).astype(int)

    # Log of amount to reduce skew
    df["amount_log"] = np.log1p(df["amount"])

    # ----- Customer daily aggregates -----
    # Number of transactions per customer per day
    grp = df.groupby(["customer_id", "tx_date"])
    df["cust_day_txn_count"] = grp["transaction_id"].transform("count")
    df["cust_day_total_amount"] = grp["amount"].transform("sum")

    # Overall transaction count per customer (lifetime in this dataset)
    df["cust_total_txn_count"] = df.groupby("customer_id")["transaction_id"].transform("count")
    df["cust_avg_amount"] = df.groupby("customer_id")["amount"].transform("mean")

    # ----- One-hot encode categoricals -----
    cat_cols = ["country", "merchant_category", "channel"]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True, prefix_sep="__")

    return df


def get_feature_target_matrices(df: pd.DataFrame):
    """
    Split dataframe into X (features) and y (label) for modeling.

    We drop ID / raw columns that are not useful for the model.
    """
    drop_cols = [
        "transaction_id",
        "customer_id",
        "timestamp",
        "currency",
        "tx_date",
        "device_id",  # string ID, not a numeric feature
    ]
    feature_cols = [c for c in df.columns if c not in drop_cols + ["is_fraud"]]

    X = df[feature_cols].values
    y = df["is_fraud"].values

    return X, y, feature_cols
