import pathlib

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# ---------- Helpers to rebuild readable categories from one-hot ----------


def add_category_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruct human-readable label columns from one-hot encoded fields:
        - country__US -> country_label
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
            # base category that was dropped with drop_first=True
            return "OTHER"

        df_inner[new_col] = df_inner[one_hot_cols].apply(infer_from_row, axis=1)
        return df_inner

    df = rebuild_label(df, "country", "country_label")
    df = rebuild_label(df, "merchant_category", "merchant_category_label")
    df = rebuild_label(df, "channel", "channel_label")
    return df


# ---------- Data loading ----------


@st.cache_data
def load_dataset_with_optional_rules(
    base_scored_path: str,
    rules_scored_path: str,
    rules_metrics_path: str,
):
    """
    Load the main dataset.

    Priority:
    1. If rules_scored_path exists -> load it and any rule metrics
    2. Else -> load base_scored_path only
    """
    base_path = pathlib.Path(base_scored_path)
    rules_path = pathlib.Path(rules_scored_path)
    metrics_path = pathlib.Path(rules_metrics_path)

    rule_metrics_df = None
    has_rules = False

    if rules_path.exists():
        df = pd.read_csv(rules_path, parse_dates=["timestamp"])
        has_rules = True

        if metrics_path.exists():
            rule_metrics_df = pd.read_csv(metrics_path)
    else:
        df = pd.read_csv(base_path, parse_dates=["timestamp"])

    # Ensure essential columns
    if "is_fraud" not in df.columns:
        df["is_fraud"] = 0

    if "model_anomaly_flag" not in df.columns:
        df["model_anomaly_flag"] = 0

    if "model_anomaly_score" not in df.columns:
        df["model_anomaly_score"] = 0.0

    # Add readable label columns
    df = add_category_labels(df)

    return df, rule_metrics_df, has_rules


@st.cache_data
def load_uploaded_csv(file):
    df = pd.read_csv(file, parse_dates=["timestamp"], low_memory=False)

    if "is_fraud" not in df.columns:
        df["is_fraud"] = 0

    if "model_anomaly_flag" not in df.columns:
        df["model_anomaly_flag"] = 0

    if "model_anomaly_score" not in df.columns:
        df["model_anomaly_score"] = 0.0

    df = add_category_labels(df)

    rule_cols = [c for c in df.columns if c.startswith("rule_") and c.endswith("_flag")]
    has_rules = len(rule_cols) > 0

    return df, None, has_rules


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


def plot_model_rule_overlap(df: pd.DataFrame, rule_flag_cols):
    """
    Simple bar chart showing overlap between model alerts and any rule alerts.
    """
    if not rule_flag_cols:
        st.info("No rule flags available to compare overlap.")
        return

    model_flag = df["model_anomaly_flag"].astype(int)
    any_rule = df[rule_flag_cols].max(axis=1).astype(int)

    both = int(((model_flag == 1) & (any_rule == 1)).sum())
    model_only = int(((model_flag == 1) & (any_rule == 0)).sum())
    rule_only = int(((model_flag == 0) & (any_rule == 1)).sum())
    neither = int(((model_flag == 0) & (any_rule == 0)).sum())

    labels = ["Model only", "Rules only", "Both", "Neither"]
    counts = [model_only, rule_only, both, neither]

    fig, ax = plt.subplots()
    ax.bar(labels, counts)
    ax.set_ylabel("Number of transactions")
    ax.set_title("Overlap between model and rule alerts")

    for i, v in enumerate(counts):
        ax.text(i, v, f"{v}", ha="center", va="bottom", fontsize=8)

    st.pyplot(fig)


def format_rule_metrics(rule_metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean up and format rule metrics for display.
    """
    df = rule_metrics_df.copy()

    for col in ["precision", "recall", "f1", "flag_rate", "fraud_rate"]:
        if col in df.columns:
            df[col] = (df[col] * 100).round(2)

    rename_map = {
        "rule_id": "Rule ID",
        "rule_name": "Rule name",
        "flags": "Flags",
        "tp": "TP",
        "fp": "FP",
        "fn": "FN",
        "precision": "Precision (%)",
        "recall": "Recall (%)",
        "f1": "F1 (%)",
        "flag_rate": "Flag rate (%)",
    }

    df = df[[c for c in rename_map.keys() if c in df.columns]].rename(columns=rename_map)
    return df


def build_customer_risk_table(df: pd.DataFrame, rule_flag_cols):
    """
    Aggregate to customer level and compute a simple risk score.
    """
    temp = df.copy()

    if rule_flag_cols:
        temp["rule_any_flag"] = temp[rule_flag_cols].max(axis=1).astype(int)
    else:
        temp["rule_any_flag"] = 0

    grp = temp.groupby("customer_id").agg(
        n_tx=("transaction_id", "count"),
        total_amount=("amount", "sum"),
        avg_amount=("amount", "mean"),
        max_anomaly=("model_anomaly_score", "max"),
        model_flags=("model_anomaly_flag", "sum"),
        rule_flags=("rule_any_flag", "sum"),
        frauds=("is_fraud", "sum"),
    )

    # Simple heuristic risk score
    grp["risk_score"] = (
        grp["max_anomaly"]
        + 0.01 * grp["model_flags"]
        + 0.02 * grp["rule_flags"]
        + 0.5 * (grp["frauds"] > 0).astype(int)
    )

    grp = grp.sort_values("risk_score", ascending=False).reset_index()
    return grp


# ---------- App ----------


def main():
    st.set_page_config(
        page_title="Bank Fraud Rule Explorer",
        layout="wide",
    )

    st.title("🕵️ Bank Fraud Rule Explorer")
    st.write(
        "Explore model-generated anomaly scores, understand which transactions look suspicious, "
        "and compare ML-based alerts with rule-based alerts."
    )

    base_scored_path = "data/transactions_with_scores.csv"
    rules_scored_path = "data/transactions_with_rules.csv"
    rules_metrics_path = "data/rule_metrics.csv"

    # Sidebar: data source
    st.sidebar.header("Data")
    data_option = st.sidebar.radio(
        "Choose data source:",
        ["Use sample scored dataset", "Upload your own CSV"],
    )

    if data_option == "Use sample scored dataset":
        base_path = pathlib.Path(base_scored_path)
        if not base_path.exists():
            st.error(
                f"Base scored dataset not found at `{base_scored_path}`. "
                "Run `python src/anomaly_model.py` from the project root first."
            )
            return

        df, rule_metrics_df, has_rules = load_dataset_with_optional_rules(
            base_scored_path, rules_scored_path, rules_metrics_path
        )

        if not has_rules:
            st.info(
                "Rule engine outputs were not found. "
                "To enable rule metrics, run:\n\n"
                "`python src/rule_engine.py`"
            )
    else:
        uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
        if uploaded is None:
            st.warning("Upload a CSV file to continue.")
            return
        df, rule_metrics_df, has_rules = load_uploaded_csv(uploaded)

    # Detect rule flag columns
    rule_flag_cols = []
    if has_rules:
        rule_flag_cols = [
            c for c in df.columns if c.startswith("rule_") and c.endswith("_flag")
        ]
        has_rules = has_rules and len(rule_flag_cols) > 0

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

    country_options = ["All"] + sorted(df["country_label"].dropna().unique().tolist())
    selected_country = st.sidebar.selectbox("Country", country_options)

    merchant_options = ["All"] + sorted(
        df["merchant_category_label"].dropna().unique().tolist()
    )
    selected_merchant = st.sidebar.selectbox("Merchant category", merchant_options)

    channel_options = ["All"] + sorted(df["channel_label"].dropna().unique().tolist())
    selected_channel = st.sidebar.selectbox("Channel", channel_options)

    # Rule filter (if rules available)
    rule_filter_col = None
    if has_rules and rule_flag_cols:
        display_to_col = {"None (no rule filter)": None}

        if rule_metrics_df is not None and "rule_id" in rule_metrics_df.columns:
            rule_metrics_df["flag_col"] = (
                "rule_" + rule_metrics_df["rule_id"].astype(str) + "_flag"
            )
            for _, row in rule_metrics_df.iterrows():
                col = row["flag_col"]
                if col in rule_flag_cols:
                    label = f"{row['rule_id']} – {row['rule_name']}"
                    display_to_col[label] = col
        else:
            for col in rule_flag_cols:
                display_to_col[col] = col

        selected_rule_display = st.sidebar.selectbox(
            "Filter by rule flag", list(display_to_col.keys())
        )
        rule_filter_col = display_to_col[selected_rule_display]

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

    if rule_filter_col is not None:
        filtered = filtered[filtered[rule_filter_col] == 1]

    # Overall KPIs (on full dataset)
    st.subheader("Overall model performance (full dataset)")
    kpi_row(df)

    st.markdown("---")
    st.subheader("High-risk transactions (after filters)")

    st.write(
        f"Showing **{len(filtered):,}** transactions with anomaly score ≥ "
        f"`{score_threshold:.3f}`"
    )

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

    if has_rules and rule_flag_cols:
        display_cols.extend(rule_flag_cols)

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

    # Model vs rule overlap
    if has_rules and rule_flag_cols:
        st.markdown("---")
        st.subheader("Model vs rule overlap")
        st.write(
            "How often do ML-based alerts and rule-based alerts agree on the same transactions?"
        )
        plot_model_rule_overlap(df, rule_flag_cols)

    # Rule engine metrics table
    if has_rules and rule_metrics_df is not None and not rule_metrics_df.empty:
        st.markdown("---")
        st.subheader("Rule engine performance")

        st.write(
            "Comparison of individual rule performance against the ground-truth fraud labels. "
            "This helps explain which expert rules are most effective and how they compare "
            "to the ML-based anomaly detector."
        )

        formatted_rules = format_rule_metrics(rule_metrics_df)
        st.dataframe(formatted_rules)

    # Customer risk ranking
    st.markdown("---")
    st.subheader("Customer risk ranking")

    st.write(
        "Aggregated view of customers combining anomaly scores, alerts, and fraud tags."
    )
    top_n = st.slider(
        "Number of top customers to display", min_value=10, max_value=100, value=25, step=5
    )
    cust_risk = build_customer_risk_table(df, rule_flag_cols)
    st.dataframe(cust_risk.head(top_n))


if __name__ == "__main__":
    main()
