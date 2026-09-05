"""IKEA customer segmentation analysis.

The customer-level model uses a public Customer Personality Analysis dataset.
IKEA is the business case context; the public dataset is not presented as IKEA's
confidential customer database.
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "marketing_campaign.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Run `python src/download_data.py` first."
        )
    df = pd.read_csv(DATA_PATH, sep=";")
    df.columns = [c.strip().replace("\ufeff", "") for c in df.columns]
    return df


def clean_and_engineer(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    raw_rows = len(df)

    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], errors="coerce")
    duplicate_rows = int(df.duplicated(subset=["ID"]).sum())
    df = df.drop_duplicates(subset=["ID"]).copy()

    missing_income = int(df["Income"].isna().sum())
    df["Income"] = df["Income"].fillna(df["Income"].median())

    reference_year = int(df["Dt_Customer"].dt.year.max())
    df["Age"] = reference_year - df["Year_Birth"]

    # Remove clearly implausible ages while keeping the cleaning rule explicit.
    implausible_age = int(((df["Age"] < 18) | (df["Age"] > 90)).sum())
    df = df.loc[df["Age"].between(18, 90)].copy()

    spending_cols = [
        "MntWines", "MntFruits", "MntMeatProducts",
        "MntFishProducts", "MntSweetProducts", "MntGoldProds"
    ]
    channel_cols = ["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]
    campaign_cols = ["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5", "Response"]

    df["Total_Spending"] = df[spending_cols].sum(axis=1)
    df["Total_Purchases"] = df[channel_cols].sum(axis=1)
    df["Children_Total"] = df["Kidhome"] + df["Teenhome"]
    df["Campaign_Acceptances"] = df[campaign_cols].sum(axis=1)
    df["Web_Purchase_Share"] = np.where(df["Total_Purchases"] > 0, df["NumWebPurchases"] / df["Total_Purchases"], 0)
    df["Store_Purchase_Share"] = np.where(df["Total_Purchases"] > 0, df["NumStorePurchases"] / df["Total_Purchases"], 0)
    df["Catalog_Purchase_Share"] = np.where(df["Total_Purchases"] > 0, df["NumCatalogPurchases"] / df["Total_Purchases"], 0)
    df["Customer_Tenure_Years"] = (df["Dt_Customer"].max() - df["Dt_Customer"]).dt.days / 365.25

    # Cleaned customer table is useful for the notebook and dashboard.
    df.to_csv(OUT / "cleaned_customers.csv", index=False)

    metadata = {
        "raw_rows": raw_rows,
        "duplicate_customer_rows_removed": duplicate_rows,
        "missing_income_values_imputed": missing_income,
        "implausible_age_rows_removed": implausible_age,
        "reference_year_for_age": reference_year,
        "final_customer_count": int(len(df)),
    }
    return df, metadata


def choose_clusters(X_scaled: np.ndarray) -> tuple[int, pd.DataFrame]:
    rows = []
    for k in range(2, 9):
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
        labels = model.fit_predict(X_scaled)
        rows.append({
            "k": k,
            "inertia": float(model.inertia_),
            "silhouette_score": float(silhouette_score(X_scaled, labels)),
            "calinski_harabasz": float(calinski_harabasz_score(X_scaled, labels)),
            "davies_bouldin": float(davies_bouldin_score(X_scaled, labels)),
        })
    scores = pd.DataFrame(rows)
    best_k = int(scores.loc[scores["silhouette_score"].idxmax(), "k"])
    scores.to_csv(OUT / "cluster_evaluation.csv", index=False)
    return best_k, scores


def add_business_labels(df: pd.DataFrame, profiles: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    profiles = profiles.copy()
    # Relative scores are used only to translate model output into readable business labels.
    profiles["value_score"] = (
        profiles["Total_Spending"].rank(pct=True)
        + profiles["Total_Purchases"].rank(pct=True)
        + profiles["Income"].rank(pct=True)
        + (1 - profiles["Recency"].rank(pct=True))
    )
    profiles["engagement_score"] = (
        profiles["Total_Purchases"].rank(pct=True)
        + (1 - profiles["Recency"].rank(pct=True))
        + profiles["Campaign_Acceptances"].rank(pct=True)
    )
    profiles["risk_score"] = profiles["Recency"].rank(pct=True) + (1 - profiles["Total_Purchases"].rank(pct=True))

    # Highest-value cluster first.
    high_value = int(profiles["value_score"].idxmax())
    remaining = [i for i in profiles.index if i != high_value]
    labels = {high_value: "High-Value Customers"}

    if remaining:
        at_risk = int(profiles.loc[remaining, "risk_score"].idxmax())
        labels[at_risk] = "Lower-Value Customers"
        remaining = [i for i in remaining if i != at_risk]

    if remaining:
        loyal = int(profiles.loc[remaining, "engagement_score"].idxmax())
        labels[loyal] = "Engaged Customers"
        remaining = [i for i in remaining if i != loyal]

    for i in remaining:
        labels[i] = "Developing Customers"

    df["Segment"] = df["Cluster"].map(labels)
    profiles["Segment"] = profiles.index.map(labels)
    return df, labels


def make_charts(df: pd.DataFrame, scores: pd.DataFrame, pca_df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 5))
    plt.plot(scores["k"], scores["inertia"], marker="o")
    plt.xlabel("Number of clusters (K)")
    plt.ylabel("Within-cluster sum of squares")
    plt.title("Elbow Method")
    plt.tight_layout()
    plt.savefig(OUT / "01_elbow_method.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(scores["k"], scores["silhouette_score"], marker="o")
    plt.xlabel("Number of clusters (K)")
    plt.ylabel("Silhouette score")
    plt.title("Silhouette Score by K")
    plt.tight_layout()
    plt.savefig(OUT / "02_silhouette_scores.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 6))
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="Segment", palette="Set2", s=55, alpha=0.8)
    plt.title("Customer Segments in PCA Space")
    plt.tight_layout()
    plt.savefig(OUT / "03_pca_segments.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    order = df.groupby("Segment")["Total_Spending"].mean().sort_values(ascending=False).index
    sns.barplot(data=df, x="Segment", y="Total_Spending", order=order, estimator="mean", errorbar=None)
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Average spending")
    plt.title("Average Spending by Customer Segment")
    plt.tight_layout()
    plt.savefig(OUT / "04_spending_by_segment.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    counts = df["Segment"].value_counts().sort_values(ascending=False)
    sns.barplot(x=counts.index, y=counts.values)
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Customers")
    plt.title("Customer Count by Segment")
    plt.tight_layout()
    plt.savefig(OUT / "05_customers_by_segment.png", dpi=180)
    plt.close()


def main():
    df = load_data()
    df, metadata = clean_and_engineer(df)

    clustering_features = [
        "Age", "Income", "Recency", "Total_Spending",
        "Total_Purchases", "NumWebVisitsMonth", "Children_Total",
        "Campaign_Acceptances"
    ]

    # Log transforms reduce the influence of highly skewed monetary and income variables.
    X = df[clustering_features].copy()
    for col in ["Income", "Total_Spending", "Total_Purchases"]:
        X[col] = np.log1p(X[col].clip(lower=0))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    best_k, scores = choose_clusters(X_scaled)

    model = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=20)
    df["Cluster"] = model.fit_predict(X_scaled)

    profile_cols = [
        "Income", "Age", "Recency", "Total_Spending", "Total_Purchases",
        "NumWebVisitsMonth", "Children_Total", "Campaign_Acceptances",
        "Web_Purchase_Share", "Store_Purchase_Share", "Catalog_Purchase_Share"
    ]
    profiles = df.groupby("Cluster")[profile_cols].mean().round(2)
    profiles["Customers"] = df.groupby("Cluster").size()
    profiles["Revenue_Contribution_%"] = (profiles["Total_Spending"] / profiles["Total_Spending"].sum() * 100).round(2)

    df, labels = add_business_labels(df, profiles)
    profiles = df.groupby(["Cluster", "Segment"])[profile_cols].mean().round(2)
    profiles["Customers"] = df.groupby(["Cluster", "Segment"]).size()
    profiles["Revenue_Contribution_%"] = (
        profiles["Total_Spending"] / profiles["Total_Spending"].sum() * 100
    ).round(2)
    profiles = profiles.reset_index()
    profiles.to_csv(OUT / "segment_profiles.csv", index=False)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pcs = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame({"PC1": pcs[:, 0], "PC2": pcs[:, 1], "Segment": df["Segment"].values})
    pca_df.to_csv(OUT / "pca_segments.csv", index=False)

    make_charts(df, scores, pca_df)

    # Save model summary for the dashboard and report.
    best_row = scores.loc[scores["k"] == best_k].iloc[0]
    summary = {
        **metadata,
        "selected_k": best_k,
        "silhouette_score": round(float(best_row["silhouette_score"]), 4),
        "calinski_harabasz": round(float(best_row["calinski_harabasz"]), 2),
        "davies_bouldin": round(float(best_row["davies_bouldin"]), 4),
        "pca_explained_variance": [round(float(x), 4) for x in pca.explained_variance_ratio_],
        "segment_labels": labels,
        "ikea_context": {
            "fy25_retail_sales_eur_billion": 44.6,
            "fy25_store_visitors_million": 915,
            "source": "IKEA FY25 Year in Review"
        }
    }
    (OUT / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    df.to_csv(OUT / "segmented_customers.csv", index=False)

    print("=" * 70)
    print("IKEA CUSTOMER SEGMENTATION ANALYSIS")
    print("=" * 70)
    print(f"Customers analyzed: {len(df):,}")
    print(f"Selected K: {best_k}")
    print(f"Silhouette score: {best_row['silhouette_score']:.4f}")
    print("\nSegments:")
    print(df["Segment"].value_counts().to_string())
    print("\nOutputs saved to:", OUT)


if __name__ == "__main__":
    main()
