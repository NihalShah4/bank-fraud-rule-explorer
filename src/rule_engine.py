import numpy as np
import pandas as pd

from metrics import compute_basic_metrics


# ---------- Helpers to rebuild readable categories from one-hot ----------


def add_category_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruct human-readable label columns from one-hot encoded fields:

        - country__US, country__IN -> country_label
        - merchant_category__GROCERIES -> merchant_category_label
        - channel__ONLINE -> channel_label
    """

    def rebuild_label(df_inner: pd.DataFrame, prefix: str, new_col: str) -> pd.DataFrame:
        one_hot_cols = [c for c in df_inner.columns if c.startswith(prefix + "__")]
        if not one_hot_cols:
            df_inner[new_col] = "UNKNOWN"
            return df_inner

        def infer_from_row(row):
            for col in one_hot_cols:
                if row[col] == 1:
                    return col.split("__", 1)[1]
            # base category that was dropped when we used drop_first=True
            return "OTHER"

        df_inner[new_col] = df_inner[one_hot_cols].apply(infer_from_row, axis=1)
        return df_inner

    df = rebuild_label(df, "country", "country_label")
    df = rebuild_label(df, "merchant_category", "merchant_category_label")
    df = rebuild_label(df, "channel", "channel_label")
    return df


# ---------- Rule definitions ----------


def rule_high_amount(df: pd.DataFrame) -> pd.Series:
    """Rule 1: Very high single transaction amount."""
    return df["amount"] >= 5000  # USD


def rule_foreign_high_risk_online(df: pd.DataFrame) -> pd.Series:
    """Rule 2: Foreign, high-risk merchant, online channel."""
    high_risk_mcc = {"GAMBLING", "CRYPTO_EXCHANGE"}
    return (
        (df["country_label"] != "US")
        & (df["merchant_category_label"].isin(high_risk_mcc))
        & (df["channel_label"] == "ONLINE")
    )


def rule_velocity_per_day(df: pd.DataFrame) -> pd.Series:
    """
    Rule 3: Unusual velocity per customer per day.
    Uses engineered fields from preprocessing: cust_day_txn_count, cust_day_total_amount.
    """
    return (df["cust_day_txn_count"] >= 6) & (df["cust_day_total_amount"] >= 3000)


def rule_nighttime_high_amount(df: pd.DataFrame) -> pd.Series:
    """Rule 4: High-value transactions at night (00:00–04:59)."""
    return (df["tx_hour"].between(0, 4)) & (df["amount"] >= 1500)


RULE_SET = [
    {
        "id": "R1",
        "name": "High amount > 5,000",
        "description": "Flags single transactions with amount >= 5,000 USD.",
        "func": rule_high_amount,
    },
    {
        "id": "R2",
        "name": "Foreign + high-risk MCC + online",
        "description": "Flags foreign online transactions at gambling/crypto merchants.",
        "func": rule_foreign_high_risk_online,
    },
    {
        "id": "R3",
        "name": "Daily velocity spike",
        "description": "Flags customers with >= 6 txns and >= 3,000 USD in a single day.",
        "func": rule_velocity_per_day,
    },
    {
        "id": "R4",
        "name": "Night-time high-amount",
        "description": "Flags high-value transactions between midnight and 5 AM.",
        "func": rule_nighttime_high_amount,
    },
]


# ---------- Core rule engine ----------


def apply_rules(df: pd.DataFrame):
    """
    Apply all rules in RULE_SET to the dataframe.

    Returns:
        df_with_flags: original df with extra columns: rule_<id>_flag
        metrics_df:    one row per rule with metrics
    """
    df = df.copy()

    # Ensure readable label columns exist
    df = add_category_labels(df)

    metrics_rows = []

    for rule in RULE_SET:
        rule_id = rule["id"]
        rule_name = rule["name"]
        rule_desc = rule["description"]
        rule_func = rule["func"]

        flag_col = f"rule_{rule_id}_flag"

        # Apply rule
        flags = rule_func(df).astype(int)
        df[flag_col] = flags

        # Evaluate against true fraud label
        m = compute_basic_metrics(df, label_col="is_fraud", flag_col=flag_col)

        metrics_rows.append(
            {
                "rule_id": rule_id,
                "rule_name": rule_name,
                "description": rule_desc,
                "flags": m["flags"],
                "frauds": m["frauds"],
                "tp": m["tp"],
                "fp": m["fp"],
                "fn": m["fn"],
                "precision": m["precision"],
                "recall": m["recall"],
                "flag_rate": m["flag_rate"],
                "fraud_rate": m["fraud_rate"],
                "f1": m["f1"],
            }
        )

    metrics_df = pd.DataFrame(metrics_rows)
    return df, metrics_df


if __name__ == "__main__":
    # CLI usage: evaluate rules on the scored transactions
    input_path = "data/transactions_with_scores.csv"
    output_tx_path = "data/transactions_with_rules.csv"
    output_metrics_path = "data/rule_metrics.csv"

    df_scored = pd.read_csv(input_path, parse_dates=["timestamp"])
    print(f"Loaded {len(df_scored):,} scored transactions from {input_path}")

    df_with_flags, rule_metrics = apply_rules(df_scored)

    df_with_flags.to_csv(output_tx_path, index=False)
    rule_metrics.to_csv(output_metrics_path, index=False)

    print(f"\nSaved transactions with rule flags to: {output_tx_path}")
    print(f"Saved rule-level metrics to:         {output_metrics_path}\n")

    print("=== Rule performance summary ===")
    # pretty print metrics
    display_cols = [
        "rule_id",
        "rule_name",
        "flags",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "flag_rate",
    ]
    print(rule_metrics[display_cols].to_string(index=False, float_format=lambda x: f"{x:0.2%}" if isinstance(x, float) else f"{x}"))
