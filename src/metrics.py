import numpy as np
import pandas as pd


def compute_basic_metrics(
    df: pd.DataFrame,
    label_col: str = "is_fraud",
    flag_col: str = "model_anomaly_flag",
) -> dict:
    """
    Compute simple classification metrics from a scored dataframe.

    Assumes:
        - df[label_col] is 0/1 ground-truth fraud label
        - df[flag_col] is 0/1 model flag (e.g. anomaly flag)
    """
    y_true = df[label_col].values.astype(int)
    y_pred = df[flag_col].values.astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    total = tp + fp + fn + tn
    frauds = tp + fn
    flags = tp + fp

    recall = tp / frauds if frauds else 0.0
    precision = tp / flags if flags else 0.0
    accuracy = (tp + tn) / total if total else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    fraud_rate = frauds / total if total else 0.0
    flag_rate = flags / total if total else 0.0

    return {
        "total_rows": total,
        "frauds": frauds,
        "flags": flags,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "fraud_rate": fraud_rate,
        "flag_rate": flag_rate,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "accuracy": accuracy,
    }


def metrics_for_thresholds(
    df: pd.DataFrame,
    score_col: str = "model_anomaly_score",
    label_col: str = "is_fraud",
    quantiles=(0.90, 0.95, 0.97, 0.99),
) -> pd.DataFrame:
    """
    Evaluate model performance at different anomaly score thresholds.

    For each quantile of the anomaly score, we:
        - create a temporary flag column
        - compute precision / recall / flag rate
    """
    scores = df[score_col].values
    thresholds = [np.quantile(scores, q) for q in quantiles]

    rows = []
    for q, thr in zip(quantiles, thresholds):
        temp_df = df.copy()
        temp_df["temp_flag"] = (temp_df[score_col] >= thr).astype(int)

        m = compute_basic_metrics(temp_df, label_col=label_col, flag_col="temp_flag")

        rows.append(
            {
                "quantile": q,
                "threshold": thr,
                "flag_rate": m["flag_rate"],
                "recall": m["recall"],
                "precision": m["precision"],
                "f1": m["f1"],
            }
        )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Quick CLI tool: run this file directly to inspect metrics.
    path = "data/transactions_with_scores.csv"
    df = pd.read_csv(path)

    print(f"Loaded {len(df):,} rows from {path}\n")

    basic = compute_basic_metrics(df)
    print("=== Overall metrics at current flag column ===")
    for k, v in basic.items():
        if k in {"fraud_rate", "flag_rate", "recall", "precision", "f1", "accuracy"}:
            print(f"{k:12s}: {v:6.2%}")
        else:
            print(f"{k:12s}: {v}")

    print("\n=== Metrics by anomaly score quantile ===")
    table = metrics_for_thresholds(df)
    # pretty print without index
    print(table.to_string(index=False, float_format=lambda x: f"{x:0.4f}"))
