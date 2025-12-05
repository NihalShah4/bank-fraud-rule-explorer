import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

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

    # 6. Simple evaluation vs true fraud labels (remember: this is synthetic data)
    fraud_mask = df["is_fraud"] == 1
    true_frauds = fraud_mask.sum()
    detected_frauds = (fraud_mask & (df["model_anomaly_flag"] == 1)).sum()
    total_flagged = df["model_anomaly_flag"].sum()

    recall = detected_frauds / true_frauds if true_frauds > 0 else 0.0
    precision = detected_frauds / total_flagged if total_flagged > 0 else 0.0
    fraud_rate = df["is_fraud"].mean()

    print("=== Model summary ===")
    print(f"Rows: {len(df):,}")
    print(f"True frauds in data: {true_frauds} ({fraud_rate:.2%})")
    print(f"Flagged by model: {total_flagged} ({total_flagged / len(df):.2%})")
    print(f"Detected frauds: {detected_frauds}")
    print(f"Recall (of frauds): {recall:.2%}")
    print(f"Precision (of flags): {precision:.2%}")

    # 7. Save scored dataset
    df.to_csv(output_path, index=False)
    print(f"\nScored transactions saved to: {output_path}")

    return df, feature_cols, model


if __name__ == "__main__":
    train_and_score("data/sample_transactions.csv")
