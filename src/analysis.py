"""
All figures are saved as PNG files to  ./analysis_output/.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # save to file – no display window needed
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
from itertools import combinations

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR  = os.path.join(BASE_DIR, "analysis_output")
os.makedirs(OUT_DIR, exist_ok=True)

# Constants
MODELS = ["textblob", "vader", "bert", "gpt-2"]
MODEL_LABELS = {
    "textblob": "TextBlob",
    "vader":    "VADER",
    "bert":     "BERT",
    "gpt-2":    "GPT-2",
}
# The runtime CSV uses title-cased names
RT_TO_KEY = {"TextBlob": "textblob", "VADER": "vader", "BERT": "bert", "GPT-2": "gpt-2"}

CLASSES = ["negative", "neutral", "positive"]

CLASS_COLOR = {
    "negative": "#d62728",
    "neutral":  "#aec7e8",
    "positive": "#2ca02c",
}
MODEL_COLOR = {
    "TextBlob": "#1f77b4",
    "VADER":    "#ff7f0e",
    "BERT":     "#9467bd",
    "GPT-2":    "#8c564b",
}

sns.set_theme(style="whitegrid", font_scale=1.05)

# Helper
_fig_counter = [0]

def save(fig, stem):
    _fig_counter[0] += 1
    name = f"{_fig_counter[0]:02d}_{stem}.png"
    fig.savefig(os.path.join(OUT_DIR, name), bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {name}")


def styled_table(ax, df, col_labels=None):
    """Render a DataFrame as a nicely styled table"""
    col_labels = col_labels or list(df.columns)
    t = ax.table(
        cellText=df.values,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    t.auto_set_font_size(False)
    t.set_fontsize(8.5)
    t.scale(1.1, 1.55)
    for (r, c), cell in t.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f4f4f4")


# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("Loading data …")
rt_df  = pd.read_csv(os.path.join(BASE_DIR, "model_run_time.csv"))
cls_df = pd.read_csv(os.path.join(BASE_DIR, "classified_tweets.csv"),
                     on_bad_lines="skip", engine="python")
lab_df = pd.read_csv(os.path.join(BASE_DIR, "labelled_tweets.csv"),
                     on_bad_lines="skip", engine="python")

lab_df["human_label"]   = lab_df["human_label"].str.strip().str.lower()
lab_df["sampled_model"] = lab_df["sampled_model"].str.strip().str.lower()

N_CLS = len(cls_df)
N_LAB = len(lab_df)
print(f"  Classified tweets : {N_CLS:,}")
print(f"  Labelled tweets   : {N_LAB:,}")
print(f"  Runtime records   : {len(rt_df)}")
print(f"\n  Human label breakdown:")
print(lab_df["human_label"].value_counts().to_string())
print(f"\n  Labelled sample source:")
print(lab_df["sampled_model"].value_counts().to_string())

y_true = lab_df["human_label"]

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — RUNTIME / THROUGHPUT
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1/9] Runtime analysis …")

rt_stats = (
    rt_df.groupby("model")
         .agg(mean_s=("total_time_seconds", "mean"),
              std_s =("total_time_seconds", "std"),
              n     =("tweets_processed",   "first"))
         .reset_index()
)
rt_stats["tps"]     = rt_stats["n"]   / rt_stats["mean_s"]
rt_stats["tps_err"] = rt_stats["tps"] * (rt_stats["std_s"] / rt_stats["mean_s"])
rt_stats["display"] = rt_stats["model"]          # already title-cased in CSV
rt_stats = rt_stats.sort_values("mean_s")        # fastest → slowest

# --- Figure 1: Runtime bar chart (time + throughput) ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Model Runtime Comparison  (3 runs each, 49,447 tweets)",
             fontsize=13, fontweight="bold")

colors = [MODEL_COLOR[m] for m in rt_stats["display"]]

ax = axes[0]
bars = ax.bar(rt_stats["display"], rt_stats["mean_s"],
              yerr=rt_stats["std_s"], capsize=6,
              color=colors, edgecolor="black", linewidth=0.7)
ax.set_yscale("log")
ax.set_ylabel("Total Classification Time (s, log scale)")
ax.set_title("Mean Classification Time")
for bar, val in zip(bars, rt_stats["mean_s"]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.5,
            f"{val:.1f}s", ha="center", fontsize=9, fontweight="bold")

ax = axes[1]
bars2 = ax.bar(rt_stats["display"], rt_stats["tps"],
               yerr=rt_stats["tps_err"], capsize=6,
               color=colors, edgecolor="black", linewidth=0.7)
ax.set_ylabel("Tweets per Second")
ax.set_title("Throughput")
for bar, val in zip(bars2, rt_stats["tps"]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
            f"{val:,.0f}", ha="center", fontsize=9, fontweight="bold")

plt.tight_layout()
save(fig, "runtime_comparison")

# --- Figure 2: Runtime summary table ---
rt_tbl = rt_stats[["display", "mean_s", "std_s", "tps", "n"]].copy()
rt_tbl.columns = ["Model", "Mean Time (s)", "Std Dev (s)", "Tweets / sec", "Tweets Processed"]
rt_tbl["Mean Time (s)"]      = rt_tbl["Mean Time (s)"].round(2)
rt_tbl["Std Dev (s)"]        = rt_tbl["Std Dev (s)"].round(2)
rt_tbl["Tweets / sec"]       = rt_tbl["Tweets / sec"].round(1)
rt_tbl["Tweets Processed"]   = rt_tbl["Tweets Processed"].apply(lambda v: f"{int(v):,}")

fig, ax = plt.subplots(figsize=(10, 2.4))
ax.axis("off")
styled_table(ax, rt_tbl)
ax.set_title("Runtime Statistics  (3 runs)", fontweight="bold", pad=10)
save(fig, "runtime_table")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — CLASSIFICATION DISTRIBUTION (full corpus)
# ══════════════════════════════════════════════════════════════════════════════
print("[2/9] Classification distribution …")

dist = pd.DataFrame(
    {MODEL_LABELS[m]: cls_df[f"{m}_class_1"]
                          .value_counts(normalize=True)
                          .reindex(CLASSES, fill_value=0) * 100
     for m in MODELS}
).T

print("\n  Sentiment distribution (%) across full corpus:")
print(dist.round(1).to_string())

fig, ax = plt.subplots(figsize=(9, 5))
x      = np.arange(len(dist))
width  = 0.25
for i, cls in enumerate(CLASSES):
    offset = (i - 1) * width
    bars = ax.bar(x + offset, dist[cls], width,
                  label=cls.capitalize(), color=CLASS_COLOR[cls],
                  edgecolor="black", linewidth=0.6)
    for bar, val in zip(bars, dist[cls]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=7.5)
ax.set_xticks(x)
ax.set_xticklabels(dist.index)
ax.set_ylabel("% of All Tweets")
ax.set_title(f"Sentiment Classification Distribution per Model  (n = {N_CLS:,})",
             fontweight="bold")
ax.legend(title="Sentiment", loc="upper right")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
plt.tight_layout()
save(fig, "classification_distribution")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — SCORE DISTRIBUTIONS (full corpus)
# ══════════════════════════════════════════════════════════════════════════════
print("[3/9] Score distributions …")

# Histograms
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("Sentiment Score Distributions  (full corpus, 49,447 tweets)",
             fontsize=13, fontweight="bold")
for ax, m in zip(axes.flat, MODELS):
    scores = cls_df[f"{m}_score_1"].dropna()
    lbl    = MODEL_LABELS[m]
    ax.hist(scores, bins=60, color=MODEL_COLOR[lbl], alpha=0.85, edgecolor="none")
    ax.axvline(scores.mean(),   color="red",   linestyle="--", linewidth=1.4,
               label=f"Mean: {scores.mean():.3f}")
    ax.axvline(scores.median(), color="black", linestyle=":",  linewidth=1.4,
               label=f"Median: {scores.median():.3f}")
    ax.set_title(lbl, fontweight="bold")
    ax.set_xlabel("Sentiment Score")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)
plt.tight_layout()
save(fig, "score_histograms")

# Violin plot
score_long = pd.concat(
    [cls_df[[f"{m}_score_1"]]
       .rename(columns={f"{m}_score_1": "score"})
       .assign(Model=MODEL_LABELS[m])
     for m in MODELS],
    ignore_index=True,
)
fig, ax = plt.subplots(figsize=(10, 5))
order = [MODEL_LABELS[m] for m in MODELS]
sns.violinplot(data=score_long, x="Model", y="score",
               hue="Model", palette=MODEL_COLOR, order=order,
               inner="quartile", ax=ax, legend=False)
ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
ax.set_title("Sentiment Score Distribution per Model  (Violin Plot)", fontweight="bold")
ax.set_xlabel("")
ax.set_ylabel("Sentiment Score")
plt.tight_layout()
save(fig, "score_violin")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — MODEL AGREEMENT (full corpus)
# ══════════════════════════════════════════════════════════════════════════════
print("[4/9] Model agreement …")

mlabels  = [MODEL_LABELS[m] for m in MODELS]
agree_df = pd.DataFrame({lbl: cls_df[f"{m}_class_1"]
                         for m, lbl in zip(MODELS, mlabels)})

# Pairwise agreement matrix
pw = pd.DataFrame(100.0, index=mlabels, columns=mlabels, dtype=float)
for m1, m2 in combinations(mlabels, 2):
    v = (agree_df[m1] == agree_df[m2]).mean() * 100
    pw.loc[m1, m2] = v
    pw.loc[m2, m1] = v

print("\n  Pairwise agreement (%):")
print(pw.round(1).to_string())

# Heatmap
fig, ax = plt.subplots(figsize=(7, 6))
annot_pw = pw.round(1).apply(lambda col: col.map(lambda x: f"{x:.1f}%"))
sns.heatmap(pw.astype(float), annot=annot_pw, fmt="",
            cmap="YlOrRd", vmin=0, vmax=100,
            linewidths=0.5, linecolor="white", ax=ax,
            cbar_kws={"label": "Agreement (%)"},
            annot_kws={"fontsize": 11})
ax.set_title("Pairwise Model Agreement\n(% tweets classified identically)",
             fontweight="bold")
plt.tight_layout()
save(fig, "pairwise_agreement_heatmap")

# Pairwise table figure
pw_disp = pw.round(1).apply(lambda col: col.map(lambda x: f"{x:.1f}%"))
pw_tbl  = pw_disp.reset_index().rename(columns={"index": ""})
fig, ax = plt.subplots(figsize=(7, 2.2))
ax.axis("off")
styled_table(ax, pw_tbl)
ax.set_title("Pairwise Model Agreement (%)", fontweight="bold", pad=10)
save(fig, "pairwise_agreement_table")

# Consensus count per tweet
agree_df["_n_agree"] = agree_df[mlabels].apply(
    lambda r: r.value_counts().iloc[0], axis=1
)
consensus = agree_df["_n_agree"].value_counts().sort_index()
tick_map = {1: "1\n(all differ)", 2: "2 agree", 3: "3 agree", 4: "All 4\nagree"}

fig, ax = plt.subplots(figsize=(8, 4.5))
bar_clrs = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]
xs = consensus.index.tolist()
bars = ax.bar([tick_map.get(v, str(v)) for v in xs],
              consensus.values,
              color=[bar_clrs[v - 1] for v in xs],
              edgecolor="black", linewidth=0.7)
for bar, val in zip(bars, consensus.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
            f"{val:,}\n({val / N_CLS * 100:.1f}%)",
            ha="center", va="bottom", fontsize=9)
ax.set_xlabel("Number of Models Agreeing on Sentiment Label")
ax.set_ylabel("Number of Tweets")
ax.set_title(f"Model Consensus Distribution  (n = {N_CLS:,})", fontweight="bold")
plt.tight_layout()
save(fig, "model_consensus")

full_agree_pct = consensus.get(4, 0) / N_CLS * 100
print(f"\n  Full consensus (all 4 agree):  "
      f"{consensus.get(4, 0):,}  ({full_agree_pct:.1f}%)")

# Agreement per-label breakdown for tweets where all 4 agree
all4 = agree_df[agree_df["_n_agree"] == 4].copy()
all4["agreed_label"] = all4[mlabels[0]]
agreed_dist = all4["agreed_label"].value_counts(normalize=True) * 100
print("  Label distribution where all 4 models agree:")
print(agreed_dist.round(1).to_string())

fig, ax = plt.subplots(figsize=(7, 4))
agreed_dist_full = agreed_dist.reindex(CLASSES, fill_value=0)
ax.bar([c.capitalize() for c in CLASSES], agreed_dist_full.values,
       color=[CLASS_COLOR[c] for c in CLASSES], edgecolor="black", linewidth=0.7)
for i, (cls, val) in enumerate(zip(CLASSES, agreed_dist_full.values)):
    ax.text(i, val + 0.5, f"{val:.1f}%", ha="center", fontsize=10, fontweight="bold")
ax.set_ylabel("% of Full-Consensus Tweets")
ax.set_title("Sentiment Distribution When All 4 Models Agree", fontweight="bold")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
plt.tight_layout()
save(fig, "consensus_label_breakdown")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — CLASSIFICATION METRICS vs HUMAN LABELS
# ══════════════════════════════════════════════════════════════════════════════
print("[5/9] Classification metrics vs human labels …")

rows = []
for m in MODELS:
    y_pred = lab_df[f"{m}_class_1"]
    rows.append({
        "Model":                MODEL_LABELS[m],
        "Accuracy":             accuracy_score(y_true, y_pred),
        "Precision (Macro)":    precision_score(y_true, y_pred, average="macro",    zero_division=0),
        "Recall (Macro)":       recall_score   (y_true, y_pred, average="macro",    zero_division=0),
        "F1 (Macro)":           f1_score       (y_true, y_pred, average="macro",    zero_division=0),
        "Precision (Weighted)": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall (Weighted)":    recall_score   (y_true, y_pred, average="weighted", zero_division=0),
        "F1 (Weighted)":        f1_score       (y_true, y_pred, average="weighted", zero_division=0),
    })
metrics_df = pd.DataFrame(rows)

print("\n  Metrics summary:")
print(metrics_df[["Model","Accuracy","Precision (Macro)","Recall (Macro)","F1 (Macro)"]].to_string(index=False))

# Grouped bar chart (macro + weighted side-by-side)
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle(f"Model Performance vs Human Labels  (n = {N_LAB:,})",
             fontsize=13, fontweight="bold")

metric_meta = [
    ("Accuracy",             "Accuracy",   "#4e79a7"),
    ("Precision ({avg})",    "Precision",  "#f28e2b"),
    ("Recall ({avg})",       "Recall",     "#e15759"),
    ("F1 ({avg})",           "F1",         "#76b7b2"),
]
for ax, avg, subtitle in zip(axes, ["Macro", "Weighted"],
                              ["Macro Average", "Weighted Average"]):
    x     = np.arange(len(MODELS))
    width = 0.19
    for i, (col_tmpl, short, col_c) in enumerate(metric_meta):
        col  = col_tmpl.format(avg=avg) if "{avg}" in col_tmpl else col_tmpl
        vals = metrics_df[col].values
        bars = ax.bar(x + (i - 1.5) * width, vals, width,
                      label=short, color=col_c,
                      edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.2f}", ha="center", va="bottom",
                    fontsize=6.5, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df["Model"])
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Score")
    ax.set_title(subtitle)
    ax.legend(fontsize=8, loc="upper right")
plt.tight_layout()
save(fig, "performance_metrics")

# Metrics table
mt = metrics_df.copy()
for col in mt.columns[1:]:
    mt[col] = mt[col].map(lambda v: f"{v:.4f}")
fig, ax = plt.subplots(figsize=(14, 2.5))
ax.axis("off")
styled_table(ax, mt, col_labels=[
    "Model", "Accuracy",
    "Prec\n(Macro)", "Rec\n(Macro)", "F1\n(Macro)",
    "Prec\n(Weighted)", "Rec\n(Weighted)", "F1\n(Weighted)",
])
ax.set_title(f"Classification Metrics vs Human Labels  (n = {N_LAB:,})",
             fontweight="bold", pad=10)
save(fig, "metrics_table")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — CONFUSION MATRICES
# ══════════════════════════════════════════════════════════════════════════════
print("[6/9] Confusion matrices …")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Confusion Matrices vs Human Labels", fontsize=14, fontweight="bold")
for ax, m in zip(axes.flat, MODELS):
    y_pred = lab_df[f"{m}_class_1"]
    cm     = confusion_matrix(y_true, y_pred, labels=CLASSES)
    cm_n   = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    annot  = np.array([[f"{cm[i,j]}\n({cm_n[i,j]*100:.1f}%)"
                        for j in range(3)] for i in range(3)])
    sns.heatmap(cm_n, annot=annot, fmt="", cmap="Blues",
                vmin=0, vmax=1,
                xticklabels=[c.capitalize() for c in CLASSES],
                yticklabels=[c.capitalize() for c in CLASSES],
                ax=ax, linewidths=0.5, linecolor="gray",
                cbar_kws={"label": "Row Proportion"})
    acc = accuracy_score(y_true, y_pred)
    ax.set_title(f"{MODEL_LABELS[m]}  (acc = {acc:.3f})", fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True (Human Label)")
plt.tight_layout()
save(fig, "confusion_matrices")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — PER-CLASS METRICS
# ══════════════════════════════════════════════════════════════════════════════
print("[7/9] Per-class metrics …")

per_class = {}
for m in MODELS:
    y_pred = lab_df[f"{m}_class_1"]
    rpt    = classification_report(
        y_true, y_pred, labels=CLASSES, output_dict=True, zero_division=0
    )
    per_class[MODEL_LABELS[m]] = {c: rpt[c] for c in CLASSES}

# Per-class F1 bar charts
fig, axes = plt.subplots(1, 3, figsize=(13, 5))
fig.suptitle("Per-Class F1 Score by Model", fontsize=13, fontweight="bold")
for ax, cls in zip(axes, CLASSES):
    vals  = {MODEL_LABELS[m]: per_class[MODEL_LABELS[m]][cls]["f1-score"] for m in MODELS}
    bars  = ax.bar(list(vals.keys()), list(vals.values()),
                   color=[MODEL_COLOR[k] for k in vals],
                   edgecolor="black", linewidth=0.6)
    ax.set_title(f"{cls.capitalize()} Class", fontweight="bold")
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0, 1.1)
    for bar, val in zip(bars, vals.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")
plt.tight_layout()
save(fig, "per_class_f1")

# Per-class heatmap (Precision / Recall / F1 for each class × model)
col_order = [f"{cls} {stat}" for cls in CLASSES for stat in ["P", "R", "F1"]]
pc_rows = {}
for m in MODELS:
    lbl = MODEL_LABELS[m]
    row = {}
    for cls in CLASSES:
        row[f"{cls} P"]  = per_class[lbl][cls]["precision"]
        row[f"{cls} R"]  = per_class[lbl][cls]["recall"]
        row[f"{cls} F1"] = per_class[lbl][cls]["f1-score"]
    pc_rows[lbl] = row
pc_hm = pd.DataFrame(pc_rows).T[col_order]

fig, ax = plt.subplots(figsize=(13, 3.8))
sns.heatmap(pc_hm.astype(float), annot=True, fmt=".3f",
            cmap="RdYlGn", vmin=0, vmax=1,
            linewidths=0.5, linecolor="white", ax=ax,
            annot_kws={"fontsize": 8.5})
ax.set_title("Per-Class Precision / Recall / F1  (rows = models, cols = class × metric)",
             fontweight="bold")
ax.set_xticklabels([c.replace(" ", "\n") for c in col_order],
                   rotation=0, fontsize=8.5)
plt.tight_layout()
save(fig, "per_class_metrics_heatmap")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — SAMPLING BIAS CHECK
# ══════════════════════════════════════════════════════════════════════════════
print("[8/9] Sampling bias check …")

# Human label distribution per sampled_model
pivot = (lab_df.groupby(["sampled_model", "human_label"])
               .size().unstack(fill_value=0))
pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
pivot_pct = pivot_pct.reindex(columns=CLASSES, fill_value=0)
pivot_pct.index = [MODEL_LABELS.get(i, i) for i in pivot_pct.index]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Sampling Bias Analysis", fontsize=13, fontweight="bold")

for ax, data, ylabel, title in zip(
    axes,
    [pivot_pct, pivot.reindex(columns=CLASSES, fill_value=0)
                     .rename(index=lambda i: MODEL_LABELS.get(i, i))],
    ["% of Sampled Tweets", "Count of Tweets"],
    ["Human Label Distribution (%)", "Human Label Count"],
):
    data.plot(kind="bar", ax=ax,
              color=[CLASS_COLOR[c] for c in CLASSES],
              edgecolor="black", linewidth=0.6)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Model Used for Sampling")
    ax.set_ylabel(ylabel)
    ax.legend(title="Human Label",
              labels=[c.capitalize() for c in CLASSES],
              fontsize=8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    if ylabel.startswith("%"):
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{v:.0f}%")
        )
plt.tight_layout()
save(fig, "sampling_bias_distribution")

# Classifier accuracy / F1 broken down by sampling source
bias_rows = []
for sm in sorted(lab_df["sampled_model"].unique()):
    sub = lab_df[lab_df["sampled_model"] == sm]
    y_t = sub["human_label"]
    for m in MODELS:
        y_p = sub[f"{m}_class_1"]
        bias_rows.append({
            "Sampled From": MODEL_LABELS.get(sm, sm),
            "Classifier":   MODEL_LABELS[m],
            "Accuracy":     accuracy_score(y_t, y_p),
            "F1 (Macro)":   f1_score(y_t, y_p, average="macro", zero_division=0),
            "N":            len(sub),
        })
bias_df = pd.DataFrame(bias_rows)

print("\n  Accuracy by sampling source (rows = source, cols = classifier):")
print(bias_df.pivot(index="Sampled From", columns="Classifier",
                    values="Accuracy").round(3).to_string())

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Classifier Performance by Sampling Source", fontsize=13, fontweight="bold")
for ax, metric in zip(axes, ["Accuracy", "F1 (Macro)"]):
    piv   = bias_df.pivot(index="Sampled From", columns="Classifier", values=metric)
    x     = np.arange(len(piv))
    width = 0.19
    for i, col in enumerate(piv.columns):
        bars = ax.bar(x + (i - 1.5) * width, piv[col], width,
                      label=col, color=MODEL_COLOR[col],
                      edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(piv.index, rotation=0)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} by Sampling Source")
    ax.legend(title="Classifier", fontsize=8)
plt.tight_layout()
save(fig, "performance_by_sampling_source")

# Sampling bias summary table
bias_acc = bias_df.pivot(index="Sampled From", columns="Classifier", values="Accuracy")
bias_acc = bias_acc.round(3)
bias_tbl = bias_acc.reset_index()
bias_tbl.columns.name = None
fig, ax = plt.subplots(figsize=(8, 2.4))
ax.axis("off")
styled_table(ax, bias_tbl)
ax.set_title("Accuracy per Classifier × Sampling Source", fontweight="bold", pad=10)
save(fig, "sampling_bias_accuracy_table")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — RADAR CHART (multi-dimensional summary)
# ══════════════════════════════════════════════════════════════════════════════
print("[9/9] Radar chart …")

tps_max    = rt_stats["tps"].max()
tps_lookup = dict(zip(rt_stats["display"], rt_stats["tps"] / tps_max))

categories = ["Accuracy", "Precision\n(Macro)", "Recall\n(Macro)", "F1\n(Macro)",
              "Throughput\n(Normalised)"]
N      = len(categories)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

radar_vals = {}
for m in MODELS:
    lbl    = MODEL_LABELS[m]
    y_pred = lab_df[f"{m}_class_1"]
    vals   = [
        accuracy_score (y_true, y_pred),
        precision_score(y_true, y_pred, average="macro", zero_division=0),
        recall_score   (y_true, y_pred, average="macro", zero_division=0),
        f1_score       (y_true, y_pred, average="macro", zero_division=0),
        float(tps_lookup.get(lbl, 0.0)),
    ]
    radar_vals[lbl] = vals

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
for lbl, vals in radar_vals.items():
    v = vals + [vals[0]]
    ax.plot(angles, v, linewidth=2.2, label=lbl, color=MODEL_COLOR[lbl])
    ax.fill(angles, v, alpha=0.10, color=MODEL_COLOR[lbl])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=9.5)
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7)
ax.set_title(
    "Model Comparison Radar Chart\n"
    "(accuracy · precision · recall · F1 — macro avg   |   normalised throughput)",
    fontweight="bold", pad=22, fontsize=10,
)
ax.legend(loc="upper right", bbox_to_anchor=(1.38, 1.12), fontsize=9)
plt.tight_layout()
save(fig, "radar_chart")


# ══════════════════════════════════════════════════════════════════════════════
# CONSOLE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  ANALYSIS COMPLETE")
print("=" * 65)
print(f"  {_fig_counter[0]} figures saved to:  {OUT_DIR}")
print()
print("  ── Key Findings ──────────────────────────────────────────")
best_acc = metrics_df.loc[metrics_df["Accuracy"].idxmax()]
best_f1  = metrics_df.loc[metrics_df["F1 (Macro)"].idxmax()]
fastest  = rt_stats.iloc[0]   # already sorted fastest first
slowest  = rt_stats.iloc[-1]
print(f"  Highest accuracy : {best_acc['Model']}  ({best_acc['Accuracy']:.4f})")
print(f"  Highest F1 macro : {best_f1['Model']}  ({best_f1['F1 (Macro)']:.4f})")
print(f"  Fastest model    : {fastest['display']}  ({fastest['tps']:,.0f} tweets/s)")
print(f"  Slowest model    : {slowest['display']}  ({slowest['tps']:.1f} tweets/s)")
print(f"  Full consensus   : {consensus.get(4,0):,} / {N_CLS:,} tweets  ({consensus.get(4,0)/N_CLS*100:.1f}%)")
print("=" * 65)
