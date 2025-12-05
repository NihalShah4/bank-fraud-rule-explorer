import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from metrics import compute_basic_metrics


from preprocessing import load_and_preprocess, get_feature_target_matrices


def train_and_score(csv_path: str, output_path: str = "data/transactions_with_scores.csv"):
    # 1. Load + feature engineer
    df = load_and_preprocess(csv_path)

    # 2. Build matrices
    X, y, feature_cols = get_feature_target_matrices(df)

    # 3. Train Isolation Forest on "normal" data (non-fraud) where possible
    normal_mask = y == 0
    X_train = X[normal_mask]

    model = IsolationForest(
        n_estimators=200,
        contamination=0.015,  # expected fraud share (~1.5%)
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train)

    # 4. Score all transactions
    # IsolationForest: higher negative_score -> more normal
    decision_scores = model.decision_function(X)
    anomaly_scores = -decision_scores  # flip so higher = more anomalous

    df["model_anomaly_score"] = anomaly_scores

    # 5. Choose threshold: top 1.5% most anomalous
    threshold = np.quantile(anomaly_scores, 0.985)
    df["model_anomaly_flag"] = (df["model_anomaly_score"] >= threshold).astype(int)

    # 6. Compute evaluation metrics using reusable function
    metrics = compute_basic_metrics(df)

    print("\nModel summary ---")
    print(f"Rows: {metrics['total_rows']:,}")
    print(f"True frauds: {metrics['frauds']:,} ({metrics['fraud_rate']:.2%})")
    print(f"Flagged by model: {metrics['flags']:,} ({metrics['flag_rate']:.2%})")
    print(f"Detected frauds (TP): {metrics['tp']:,}")
    print(f"Recall: {metrics['recall']:.2%}")
    print(f"Precision: {metrics['precision']:.2%}")
    print(f"F1-score: {metrics['f1']:.2%}")

    # 7. Save scored dataset
    df.to_csv(output_path, index=False)
    print(f"\nScored transactions saved to: {output_path}")

    return df, feature_cols, model


if __name__ == "__main__":
    train_and_score("data/sample_transactions.csv")
