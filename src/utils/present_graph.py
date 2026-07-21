# ============================================================
# src/utils/present_graph.py
#
# Guide-presentation quality graph visualization
# Clean bipartite layout with readable labels
#
# Run: python -m src.utils.present_graph
# ============================================================

import os, sys
import numpy as np
import torch
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils.config import OUTPUT_DIR

GRAPH_DIR = os.path.join(OUTPUT_DIR, "graph")
PLOT_DIR  = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# SLIDE 1 — Clean Bipartite Subgraph
# ─────────────────────────────────────────────────────────────

def plot_clean_bipartite(save_path):
    """
    Handcrafted small bipartite graph for guide presentation.
    Shows 6 users, 8 merchants, with fraud/legit edges labelled.
    """
    fig, ax = plt.subplots(figsize=(16, 10))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    # ── Define nodes ─────────────────────────────────────────
    users = [f"User {i}" for i in range(6)]
    merchants = [
        "Grocery Store", "Gas Station", "Online Shop",
        "Restaurant", "ATM", "Hotel",
        "Pharmacy", "Electronics"
    ]

    # ── Positions (bipartite layout) ─────────────────────────
    u_x, m_x = 0.15, 0.85
    u_positions = {u: (u_x, 1 - i / (len(users) - 1)) for i, u in enumerate(users)}
    m_positions = {m: (m_x, 1 - i / (len(merchants) - 1)) for i, m in enumerate(merchants)}

    # ── Define edges (src, dst, is_fraud) ─────────────────────
    edges = [
        # User 0
        ("User 0", "Grocery Store",  False),
        ("User 0", "Gas Station",    False),
        ("User 0", "Online Shop",    True),   # FRAUD

        # User 1
        ("User 1", "Restaurant",     False),
        ("User 1", "Grocery Store",  False),
        ("User 1", "ATM",            True),   # FRAUD

        # User 2
        ("User 2", "Online Shop",    False),
        ("User 2", "Electronics",    True),   # FRAUD
        ("User 2", "Pharmacy",       False),

        # User 3
        ("User 3", "Hotel",          False),
        ("User 3", "Restaurant",     False),
        ("User 3", "Gas Station",    False),

        # User 4
        ("User 4", "ATM",            False),
        ("User 4", "Online Shop",    True),   # FRAUD
        ("User 4", "Grocery Store",  False),

        # User 5
        ("User 5", "Pharmacy",       False),
        ("User 5", "Electronics",    False),
        ("User 5", "Hotel",          False),
    ]

    # ── Draw edges ────────────────────────────────────────────
    for src, dst, is_fraud in edges:
        x0, y0 = u_positions[src]
        x1, y1 = m_positions[dst]
        color  = "#ff4757" if is_fraud else "#54a0ff"
        alpha  = 0.90 if is_fraud else 0.45
        lw     = 2.5  if is_fraud else 1.2
        ls     = "--" if is_fraud else "-"
        ax.plot([x0, x1], [y0, y1],
                color=color, alpha=alpha, linewidth=lw,
                linestyle=ls, zorder=1)

    # ── Draw USER nodes ───────────────────────────────────────
    for name, (x, y) in u_positions.items():
        circle = plt.Circle((x, y), 0.038, color="#1e90ff",
                             ec="#ffffff", linewidth=1.5, zorder=3)
        ax.add_patch(circle)
        ax.text(x - 0.07, y, name, ha="right", va="center",
                color="white", fontsize=11, fontweight="bold",
                fontfamily="monospace")

    # ── Draw MERCHANT nodes ───────────────────────────────────
    for name, (x, y) in m_positions.items():
        rect = plt.Rectangle((x - 0.045, y - 0.028), 0.09, 0.056,
                              color="#ff9f43", ec="#ffffff",
                              linewidth=1.5, zorder=3)
        ax.add_patch(rect)
        ax.text(x + 0.065, y, name, ha="left", va="center",
                color="white", fontsize=10, fontfamily="monospace")

    # ── Column headers ────────────────────────────────────────
    ax.text(u_x, 1.07, "USER NODES\n(Account Holders)",
            ha="center", va="center", color="#1e90ff",
            fontsize=13, fontweight="bold")
    ax.text(m_x, 1.07, "MERCHANT NODES\n(Transaction Endpoints)",
            ha="center", va="center", color="#ff9f43",
            fontsize=13, fontweight="bold")

    # ── Divider line ──────────────────────────────────────────
    ax.axvline(x=0.5, color="#333355", linewidth=1, linestyle=":", alpha=0.6)

    # ── Legend ───────────────────────────────────────────────
    legend_items = [
        Line2D([0], [0], color="#1e90ff",  linewidth=1.5, label="Legit Transaction (Edge)"),
        Line2D([0], [0], color="#ff4757",  linewidth=2.5,
               linestyle="--", label="Fraud Transaction (Edge)"),
        mpatches.Patch(color="#1e90ff", label="User Node"),
        mpatches.Patch(color="#ff9f43", label="Merchant Node"),
    ]
    ax.legend(handles=legend_items, loc="lower center",
              ncol=4, facecolor="#1a1d2e", edgecolor="#4a9eff",
              labelcolor="white", fontsize=10,
              bbox_to_anchor=(0.5, -0.06))

    # ── Edge stats annotation ─────────────────────────────────
    n_fraud = sum(1 for _, _, f in edges if f)
    n_legit = len(edges) - n_fraud
    info = (f"Shown: {len(users)} users  ·  {len(merchants)} merchants  ·  "
            f"{n_legit} legit edges  ·  {n_fraud} fraud edges\n"
            f"Full graph: 2,000 users  ·  90,133 merchants  ·  24.3M edges")
    ax.text(0.5, -0.04, info, ha="center", va="top",
            transform=ax.transAxes,
            color="#aaaaaa", fontsize=9, style="italic")

    ax.set_xlim(-0.05, 1.10)
    ax.set_ylim(-0.1, 1.15)
    ax.axis("off")
    ax.set_title(
        "Bipartite Graph for Fraud Detection\n"
        "Nodes: Users & Merchants  |  Edges: Transactions  |  "
        "Edge Label: Fraud / Legit",
        color="white", fontsize=15, pad=20, fontweight="bold"
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✅ Saved → {save_path}")


# ─────────────────────────────────────────────────────────────
# SLIDE 2 — Pipeline Overview (Data → Graph → GNN → Output)
# ─────────────────────────────────────────────────────────────

def plot_pipeline(save_path):
    fig, ax = plt.subplots(figsize=(18, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    stages = [
        ("Raw Data\n(CSV Files)",
         "transactions.csv\nuser_details.csv\ncards.csv",
         "#3d5a80"),
        ("Preprocessing\n& SMOTE",
         "Clean features\nEncode labels\nSMOTE fraud edges",
         "#e07a5f"),
        ("Bipartite Graph\n(PyG HeteroData)",
         "2K user nodes\n90K merchant nodes\n24M edges",
         "#81b29a"),
        ("GraphSAGE\nGNN Model",
         "Neighbor sampling\nMessage passing\nEdge classifier",
         "#f2cc8f"),
        ("Fraud\nDetection",
         "AUPRC / AUROC\nF1 Score\nThreshold sweep",
         "#c77dff"),
    ]

    box_w, box_h = 0.16, 0.55
    gap = 0.21
    start_x = 0.02

    for i, (title, subtitle, color) in enumerate(stages):
        x = start_x + i * gap
        y = 0.22

        # Box
        rect = mpatches.FancyBboxPatch(
            (x, y), box_w, box_h,
            boxstyle="round,pad=0.02",
            facecolor=color, edgecolor="white",
            linewidth=1.5, alpha=0.9, zorder=2
        )
        ax.add_patch(rect)

        # Title
        ax.text(x + box_w / 2, y + box_h - 0.07, title,
                ha="center", va="top", color="white",
                fontsize=11, fontweight="bold", zorder=3)

        # Subtitle
        ax.text(x + box_w / 2, y + 0.08, subtitle,
                ha="center", va="bottom", color="#eeeeee",
                fontsize=8.5, zorder=3)

        # Arrow
        if i < len(stages) - 1:
            ax.annotate("",
                xy=(x + box_w + 0.01, y + box_h / 2),
                xytext=(x + box_w + gap - box_w - 0.01, y + box_h / 2),
                arrowprops=dict(
                    arrowstyle="<-", color="white",
                    lw=2.0, mutation_scale=18
                ), zorder=4
            )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(
        "Graph-Based Fraud Detection — Full Pipeline",
        color="white", fontsize=15, fontweight="bold", pad=10
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✅ Saved → {save_path}")


# ─────────────────────────────────────────────────────────────
# SLIDE 3 — Graph Stats Summary Card
# ─────────────────────────────────────────────────────────────

def plot_stats_card(save_path):
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.patch.set_facecolor("#0d1117")

    stats = [
        ("👤  User Nodes",      "2,000",       "Account holders\nFeat dim = 10",   "#1e90ff"),
        ("🏪  Merchant Nodes",  "90,133",       "Transaction endpoints\nFeat dim = 6", "#ff9f43"),
        ("🔗  Total Edges",     "24.3 Million", "Train / Val / Test\nsplit by Year", "#2ed573"),
        ("🚨  Fraud Edges",     "29,688",       "0.12% of all edges\nExtremely rare", "#ff4757"),
    ]

    for ax, (label, value, sub, color) in zip(axes, stats):
        ax.set_facecolor("#1a1d2e")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # Colored top border
        ax.add_patch(plt.Rectangle((0, 0.88), 1, 0.12,
                                   color=color, transform=ax.transAxes,
                                   clip_on=False))

        ax.text(0.5, 0.72, label, ha="center", va="center",
                color="#cccccc", fontsize=11, fontweight="bold",
                transform=ax.transAxes)
        ax.text(0.5, 0.45, value, ha="center", va="center",
                color=color, fontsize=22, fontweight="bold",
                transform=ax.transAxes)
        ax.text(0.5, 0.18, sub, ha="center", va="center",
                color="#aaaaaa", fontsize=9,
                transform=ax.transAxes)

        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)

    fig.suptitle("Bipartite Fraud Detection Graph — Key Statistics",
                 color="white", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✅ Saved → {save_path}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def run_all():
    print("\n" + "="*55)
    print("  Generating Guide-Presentation Plots")
    print("="*55)

    p1 = os.path.join(PLOT_DIR, "guide_bipartite_graph.png")
    p2 = os.path.join(PLOT_DIR, "guide_pipeline.png")
    p3 = os.path.join(PLOT_DIR, "guide_stats_card.png")

    print("\n[1/3] Bipartite graph diagram...")
    plot_clean_bipartite(p1)

    print("[2/3] Pipeline overview...")
    plot_pipeline(p2)

    print("[3/3] Stats summary card...")
    plot_stats_card(p3)

    print("\n" + "="*55)
    print("  ✅ All 3 slides saved in outputs/plots/")
    print("="*55)
    print(f"\n  guide_bipartite_graph.png  ← Show this first")
    print(f"  guide_pipeline.png         ← Explain the flow")
    print(f"  guide_stats_card.png       ← Summary numbers")


if __name__ == "__main__":
    run_all()
