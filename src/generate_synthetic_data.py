import numpy as np
import pandas as pd
import datetime


def generate_synthetic_transactions(
    n_customers: int = 500,
    n_transactions: int = 15000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic banking transactions with some embedded fraud patterns.

    Columns:
        - transaction_id
        - customer_id
        - timestamp
        - amount
        - currency
        - country
        - merchant_category
        - channel
        - device_id
        - is_fraud (0/1)
    """
    rng = np.random.default_rng(seed)

    # Create a pool of customers
    customer_ids = [f"C{str(i).zfill(5)}" for i in range(1, n_customers + 1)]

    # Time window: last 90 days
    end = datetime.datetime.now()
    start = end - datetime.timedelta(days=90)
    time_range_seconds = int((end - start).total_seconds())

    merchant_categories = [
        "GROCERIES",
        "RESTAURANT",
        "ELECTRONICS",
        "CLOTHING",
        "ONLINE_SUBSCRIPTION",
        "GAMBLING",
        "CRYPTO_EXCHANGE",
        "TRAVEL",
        "ATM_WITHDRAWAL",
        "UTILITY_BILL",
    ]

    channels = ["POS", "ONLINE", "ATM", "MOBILE"]

    # Bias towards US to look realistic
    countries = ["US", "US", "US", "US", "US", "CA", "GB", "IN", "MX", "CN", "DE"]

    rows = []
    for tx_id in range(1, n_transactions + 1):
        customer_id = rng.choice(customer_ids)

        # Random timestamp in the last 90 days
        random_seconds = int(rng.integers(0, time_range_seconds))
        timestamp = start + datetime.timedelta(seconds=random_seconds)

        # Transaction amount: log-normal to mimic skewed real-life amounts
        base_amount = float(rng.lognormal(mean=3, sigma=1))
        amount = round(base_amount, 2)

        merchant_category = rng.choice(
            merchant_categories,
            p=[0.18, 0.18, 0.10, 0.12, 0.08, 0.04, 0.02, 0.08, 0.10, 0.10],
        )
        channel = rng.choice(channels, p=[0.40, 0.35, 0.15, 0.10])
        country = rng.choice(countries)

        device_id = f"D{int(rng.integers(1, 400)):04d}"

        rows.append(
            [
                f"T{tx_id:07d}",
                customer_id,
                timestamp,
                amount,
                "USD",
                country,
                merchant_category,
                channel,
                device_id,
            ]
        )

    df = pd.DataFrame(
        rows,
        columns=[
            "transaction_id",
            "customer_id",
            "timestamp",
            "amount",
            "currency",
            "country",
            "merchant_category",
            "channel",
            "device_id",
        ],
    )

    # Fraud label (start as non-fraud)
    df["is_fraud"] = 0

    # Pattern 1: Very high amounts are suspicious (top 0.5%)
    high_amount_mask = df["amount"] > df["amount"].quantile(0.995)
    df.loc[high_amount_mask, "is_fraud"] = 1

    # Pattern 2: Foreign + high-risk merchant + online
    high_risk_mask = (
        (df["country"] != "US")
        & (df["merchant_category"].isin(["GAMBLING", "CRYPTO_EXCHANGE"]))
        & (df["channel"] == "ONLINE")
    )
    df.loc[high_risk_mask, "is_fraud"] = 1

    # Pattern 3: Burst of transactions in 1 hour for same customer (velocity)
    df_sorted = df.sort_values(["customer_id", "timestamp"]).copy()
    fraud_indices = set(df.index[df["is_fraud"] == 1])

    for customer_id in df_sorted["customer_id"].unique():
        cust_df = df_sorted[df_sorted["customer_id"] == customer_id]
        times = cust_df["timestamp"].values
        idxs = cust_df.index.values

        left = 0
        for right in range(len(times)):
            # Move left until the window is <= 1 hour
            while times[right] - times[left] > np.timedelta64(3600, "s"):
                left += 1

            window_size = right - left + 1
            if window_size >= 6:
                # Flag all transactions in this 1-hour burst
                for k in range(left, right + 1):
                    fraud_indices.add(int(idxs[k]))

    df.loc[list(fraud_indices), "is_fraud"] = 1

    return df


if __name__ == "__main__":
    df = generate_synthetic_transactions()
    output_path = "data/sample_transactions.csv"
    df.to_csv(output_path, index=False)
    fraud_rate = df["is_fraud"].mean()
    print(f"Saved {len(df)} transactions to {output_path}")
    print(f"Fraud rate: {fraud_rate:.2%}")
