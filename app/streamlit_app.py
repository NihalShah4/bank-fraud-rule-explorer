import pathlib

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# ---------- Helpers to rebuild readable categories from one-hot ----------

def add_category_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Our scored dataset only has one-hot columns like country__US, merchant_category__GROCERIES, etc.
    This function reconstructs human-readable label columns:
        - country_label
        - merchant_category_label
        - channel_label
    """

    def rebuild_label(df_inner: pd.DataFrame, prefix: str, new_col: str) -> pd.DataFrame:
        one_hot_cols = [c for c in df_inner.columns if c.startswith(prefix + "__")]
        if not one_hot_cols:
            # nothing to do
            df_inner[new_col] = "UNKNOWN"
            return df_inner

        def infer_from_row(row):
            for col in one_hot_cols:
                if row[col] == 1:
                    return col.split("__", 1)[1]
            # this is the "base" category that got dropped with drop_first=True
            return "OTHER"

        df_inner[new_col] = df_inner[one_hot_cols].apply(infer_from_row, axis=1)
        return df_inner

    df = rebuild_label(df, "country", "country_label")
    df = rebuild_label(df, "merchant_category", "merchant_category_label")
    df = rebuild_label(df, "channel", "channel_label")
    return df


# ---------- Data loading ----------

@st.cache_data
def load_scored_transactions(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])

    # Ensure columns exist
    if "is_fraud" not in df.columns:
        df["is_fraud"] = 0

    if "model_anomaly_flag" not in df.columns:
        df["model_anomaly_flag"] = 0

    if "model_anomaly_score" not in df.columns:
        df["model_anomaly_score"] = 0.0

    # Add readable label columns from one-hot encoded fields
    df = add_category_labels(df)

    return df


# ---------- Layout helpers ----------

def kpi_row(df: pd.DataFrame):
    total = len(df)
    frauds = int(df["is_fraud"].sum())
    flagged = int(df["model_anomaly_flag"].sum())
    fraud_rate = frauds / total if total else 0
    flag_rate = flagged / total if total else 0

    detected = int(((df["is_fraud"] == 1) & (df["model_anomaly_flag"] == 1)).sum())
    recall = detected / frauds if frauds else 0
    precision = detected / flagged if flagged else 0

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total transactions", f"{total:,}")
    c2.metric("True frauds", f"{frauds:,}", f"{fraud_rate:.2%}")
    c3.metric("Flagged by model", f"{flagged:,}", f"{flag_rate:.2%}")
    c4.metric("Detected frauds", f"{detected:,}")
    c5.metric("Recall (fraud caught)", f"{recall:.1%}")
    c6.metric("Precision (of flags)", f"{precision:.1%}")


def plot_flag_rate_by_category(df: pd.DataFrame, column: str, title: str):
    if column not in df.columns:
        st.info(f"Column `{column}` not found in data.")
        return

    grouped = (
        df.groupby(column)["model_anomaly_flag"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )

    fig, ax = plt.subplots()
    grouped[::-1].plot(kind="barh", ax=ax)
    ax.set_xlabel("Share of transactions flagged")
    ax.set_ylabel(column)
    ax.set_title(title)
    st.pyplot(fig)


# ---------- App ----------

def main():
    st.set_page_config(
        page_title="Bank Fraud Rule Explorer",
        layout="wide",
    )

    st.title("🕵️ Bank Fraud Rule Explorer")
    st.write(
        "Explore model-generated anomaly scores, understand which transactions "
        "look suspicious, and see how simple rules might perform."
    )

    # Sidebar: data source
    st.sidebar.header("Data")
    default_path = pathlib.Path("data/transactions_with_scores.csv")

    data_option = st.sidebar.radio(
        "Choose data source:",
        ["Use sample scored dataset", "Upload your own CSV"],
    )

    if data_option == "Use sample scored dataset":
        if not default_path.exists():
            st.error(
                f"Sample scored dataset not found at `{default_path}`. "
                "Run `python src/anomaly_model.py` from the project root first."
            )
            return
        df = load_scored_transactions(str(default_path))
    else:
        uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
        if uploaded is None:
            st.warning("Upload a CSV file to continue.")
            return
        df = load_scored_transactions(uploaded)

    # Sidebar: filters
    st.sidebar.header("Filters")

    min_score = float(df["model_anomaly_score"].min())
    max_score = float(df["model_anomaly_score"].max())

    score_threshold = st.sidebar.slider(
        "Minimum anomaly score",
        min_value=min_score,
        max_value=max_score,
        value=float(np.quantile(df["model_anomaly_score"], 0.9)),
    )

    # Country filter (using reconstructed label)
    country_options = ["All"] + sorted(df["country_label"].dropna().unique().tolist())
    selected_country = st.sidebar.selectbox("Country", country_options)

    # Merchant category filter
    merchant_options = (
        ["All"] + sorted(df["merchant_category_label"].dropna().unique().tolist())
    )
    selected_merchant = st.sidebar.selectbox("Merchant category", merchant_options)

    # Channel filter
    channel_options = ["All"] + sorted(df["channel_label"].dropna().unique().tolist())
    selected_channel = st.sidebar.selectbox("Channel", channel_options)

    # Apply filters
    filtered = df.copy()
    filtered = filtered[filtered["model_anomaly_score"] >= score_threshold]

    if selected_country != "All":
        filtered = filtered[filtered["country_label"] == selected_country]

    if selected_merchant != "All":
        filtered = filtered[
            filtered["merchant_category_label"] == selected_merchant
        ]

    if selected_channel != "All":
        filtered = filtered[filtered["channel_label"] == selected_channel]

    # Overall KPIs (on full dataset)
    st.subheader("Overall model performance (full dataset)")
    kpi_row(df)

    st.markdown("---")
    st.subheader("High-risk transactions (after filters)")

    st.write(
        f"Showing **{len(filtered):,}** transactions with anomaly score ≥ "
        f"`{score_threshold:.3f}`"
    )

    # Display columns for the table
    display_cols = [
        "timestamp",
        "transaction_id",
        "customer_id",
        "amount",
        "country_label",
        "merchant_category_label",
        "channel_label",
        "is_fraud",
        "model_anomaly_score",
        "model_anomaly_flag",
    ]
    display_cols = [c for c in display_cols if c in filtered.columns]

    st.dataframe(
        filtered.sort_values("model_anomaly_score", ascending=False)[display_cols].head(
            200
        )
    )

    st.markdown("---")
    st.subheader("Where is the model most suspicious?")

    c1, c2 = st.columns(2)
    with c1:
        plot_flag_rate_by_category(
            df, "merchant_category_label", "Flag rate by merchant type"
        )
    with c2:
        plot_flag_rate_by_category(df, "country_label", "Flag rate by country")


if __name__ == "__main__":
    main()
