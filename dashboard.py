"""Generate an interactive Plotly HTML dashboard from analysis outputs."""

from pathlib import Path
import json
import subprocess
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
SEGMENTED = OUT / "segmented_customers.csv"
SUMMARY = OUT / "analysis_summary.json"


def ensure_analysis():
    if not SEGMENTED.exists() or not SUMMARY.exists():
        subprocess.run([sys.executable, str(ROOT / "src" / "analysis.py")], check=True)


def fmt_money(value):
    return f"{value:,.0f}"


def main():
    ensure_analysis()
    df = pd.read_csv(SEGMENTED)
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))

    segment_order = (
        df.groupby("Segment")["Total_Spending"].sum()
        .sort_values(ascending=False).index.tolist()
    )

    counts = df["Segment"].value_counts().reindex(segment_order).fillna(0)
    revenue = df.groupby("Segment")["Total_Spending"].sum().reindex(segment_order)
    avg_spend = df.groupby("Segment")["Total_Spending"].mean().reindex(segment_order)

    fig = make_subplots(
        rows=3, cols=2,
        specs=[
            [{"type": "indicator"}, {"type": "indicator"}],
            [{"type": "pie"}, {"type": "bar"}],
            [{"type": "xy"}, {"type": "bar"}],
        ],
        subplot_titles=(
            "", "", "Customer mix", "Revenue by segment",
            "Income vs spending", "Average recency by segment"
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    fig.add_trace(go.Indicator(
        mode="number", value=len(df), title={"text": "Customers analyzed"},
        number={"valueformat": ","}
    ), row=1, col=1)

    fig.add_trace(go.Indicator(
        mode="number", value=df["Total_Spending"].sum(), title={"text": "Total customer spending"},
        number={"prefix": "€", "valueformat": ",.0f"}
    ), row=1, col=2)

    fig.add_trace(go.Pie(
        labels=counts.index, values=counts.values, hole=0.55,
        textinfo="label+percent"
    ), row=2, col=1)

    fig.add_trace(go.Bar(
        x=revenue.index, y=revenue.values,
        text=[fmt_money(x) for x in revenue.values], textposition="outside",
        name="Revenue"
    ), row=2, col=2)

    fig.add_trace(go.Scatter(
        x=df["Income"], y=df["Total_Spending"], mode="markers",
        marker={"size": 7, "opacity": 0.65},
        text=df["Segment"], hovertemplate="Income: %{x:,.0f}<br>Spending: %{y:,.0f}<br>%{text}<extra></extra>",
        name="Customers"
    ), row=3, col=1)

    recency = df.groupby("Segment")["Recency"].mean().reindex(segment_order)
    fig.add_trace(go.Bar(
        x=recency.index, y=recency.values,
        text=[f"{x:.1f}" for x in recency.values], textposition="outside",
        name="Recency"
    ), row=3, col=2)

    fig.update_xaxes(title_text="Income", row=3, col=1)
    fig.update_yaxes(title_text="Total spending", row=3, col=1)
    fig.update_yaxes(title_text="Reported customer spending", row=2, col=2)
    fig.update_yaxes(title_text="Average days since purchase", row=3, col=2)

    fig.update_layout(
        height=1100,
        title={
            "text": "IKEA Customer Segmentation & Behavioral Analytics",
            "x": 0.5,
            "xanchor": "center",
        },
        template="plotly_white",
        margin={"l": 60, "r": 60, "t": 110, "b": 60},
        showlegend=False,
    )

    # Segment profile table for the second page section.
    profiles = df.groupby("Segment").agg(
        Customers=("ID", "count"),
        Avg_Age=("Age", "mean"),
        Avg_Income=("Income", "mean"),
        Avg_Spending=("Total_Spending", "mean"),
        Avg_Purchases=("Total_Purchases", "mean"),
        Avg_Recency=("Recency", "mean"),
        Avg_Campaign_Acceptances=("Campaign_Acceptances", "mean"),
    ).reindex(segment_order).round(2).reset_index()

    table = go.Figure(data=[go.Table(
        header=dict(values=["Segment", "Customers", "Avg Age", "Avg Income", "Avg Spending", "Avg Purchases", "Avg Recency", "Campaign Acceptances"]),
        cells=dict(values=[
            profiles["Segment"], profiles["Customers"], profiles["Avg_Age"],
            profiles["Avg_Income"], profiles["Avg_Spending"], profiles["Avg_Purchases"],
            profiles["Avg_Recency"], profiles["Avg_Campaign_Acceptances"]
        ])
    )])
    table.update_layout(height=420, title="Segment Profile")

    strategies = {
        "High-Value Customers": "Protect retention with loyalty benefits, early access and personalized cross-sell recommendations.",
        "Engaged Customers": "Encourage repeat purchases and category expansion through relevant bundles and membership benefits.",
        "Developing Customers": "Use low-friction offers and personalized recommendations to increase purchase frequency and basket value.",
        "At-Risk Customers": "Prioritize re-engagement with targeted reminders, service recovery and carefully selected incentives."
    }
    strategy_rows = []
    for seg in segment_order:
        strategy_rows.append(f"<tr><td><b>{seg}</b></td><td>{strategies.get(seg, 'Use segment-specific messaging based on measured behavior.')}</td></tr>")

    plot_html = fig.to_html(full_html=False, include_plotlyjs="cdn")
    table_html = table.to_html(full_html=False, include_plotlyjs=False)

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>IKEA Customer Segmentation & Behavioral Analytics</title>
<style>
body{{font-family:Inter,Arial,sans-serif;background:#f5f7fa;margin:0;color:#182230}}
.container{{max-width:1250px;margin:0 auto;padding:28px}}
.hero{{background:#0b6f5f;color:white;border-radius:18px;padding:30px;margin-bottom:20px}}
.hero h1{{margin:0 0 8px;font-size:32px}}
.hero p{{margin:4px 0;opacity:.92}}
.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:20px 0}}
.card{{background:white;border-radius:14px;padding:18px;box-shadow:0 3px 12px rgba(0,0,0,.06)}}
.card .label{{font-size:13px;color:#64748b;text-transform:uppercase;letter-spacing:.06em}}
.card .value{{font-size:25px;font-weight:700;margin-top:8px}}
.panel{{background:white;border-radius:16px;padding:18px;margin:20px 0;box-shadow:0 3px 12px rgba(0,0,0,.06)}}
.strategy table{{width:100%;border-collapse:collapse}}.strategy td{{padding:12px;border-bottom:1px solid #e5e7eb;vertical-align:top}}
.note{{font-size:13px;color:#64748b;line-height:1.6}}
@media(max-width:800px){{.cards{{grid-template-columns:1fr}}.container{{padding:14px}}}}
</style>
</head>
<body>
<div class="container">
<div class="hero">
<h1>IKEA Customer Segmentation & Behavioral Analytics</h1>
<p>Python • scikit-learn • K-Means • PCA • Plotly</p>
<p>Customer segmentation based on demographic and purchasing behavior features.</p>
</div>
<div class="cards">
<div class="card"><div class="label">Selected clusters</div><div class="value">{summary['selected_k']}</div></div>
<div class="card"><div class="label">Silhouette score</div><div class="value">{summary['silhouette_score']:.3f}</div></div>
<div class="card"><div class="label">IKEA FY25 retail sales</div><div class="value">€44.6B</div></div>
</div>
<div class="panel"><p class="note">Model data note: the customer-level segmentation uses a public Customer Personality Analysis dataset. IKEA is used as the business case context; the public dataset is not presented as IKEA's private customer database. IKEA FY25 context is from IKEA's official Year in Review.</p></div>
<div class="panel">{plot_html}</div>
<div class="panel">{table_html}</div>
<div class="panel strategy"><h2>Segment Strategies</h2><table>{''.join(strategy_rows)}</table></div>
<div class="panel"><h2>Company Context</h2><p>IKEA reported €44.6 billion in FY25 retail sales and 915 million store visitors. The company reported that 69% of retail sales were products sold through stores, 28% products sold online and 3% services. These figures are included as business context, not as customer-level modeling inputs.</p><p class="note">Source: IKEA FY25 Year in Review — https://www.ikea.com/global/en/our-business/how-we-work/year-in-review-fy25/</p></div>
</div>
</body>
</html>"""

    path = OUT / "ikea_customer_segmentation_dashboard.html"
    path.write_text(html, encoding="utf-8")
    print(f"Dashboard created: {path}")


if __name__ == "__main__":
    main()
