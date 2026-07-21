# ============================================================
# src/utils/visualize_graph.py
#
# Bipartite graph visualizer
# Samples a subgraph (N users + their merchants) and plots it
#
# Run: python -m src.utils.visualize_graph
# ============================================================

import os, sys, json
import numpy as np
import torch
import networkx as nx
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — saves to file
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils.config import OUTPUT_DIR

GRAPH_DIR = os.path.join(OUTPUT_DIR, "graph")
PLOT_DIR  = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


def load_graph():
    path = os.path.join(GRAPH_DIR, "hetero_graph.pt")
    print(f"Loading graph from {path} ...")
    data = torch.load(path, weights_only=False)
    print("  ✅ Loaded")
    return data


def sample_subgraph(data, num_users=10, seed=42):
    """
    Sample a subgraph around `num_users` randomly selected users.
    Returns lists of (user_idx, merchant_idx, is_fraud) for edges.
    """
    np.random.seed(seed)
    ei    = data["user", "train_txn", "merchant"].edge_index  # [2, E]
    el    = data["user", "train_txn", "merchant"].edge_label  # [E]

    # Pick random users
    all_users = ei[0].unique().numpy()
    chosen    = np.random.choice(all_users, size=min(num_users, len(all_users)), replace=False)
    chosen_set = set(chosen.tolist())

    # Filter edges belonging to those users
    mask = torch.tensor([u.item() in chosen_set for u in ei[0]])
    sub_ei    = ei[:, mask]
    sub_label = el[mask]

    edges = list(zip(
        sub_ei[0].tolist(),
        sub_ei[1].tolist(),
        sub_label.tolist()
    ))

    print(f"  Subgraph: {len(chosen)} users | {len(set(sub_ei[1].tolist()))} merchants | {len(edges)} edges")
    print(f"  Fraud edges in subgraph: {sub_label.sum().item()}")
    return chosen.tolist(), list(set(sub_ei[1].tolist())), edges


def build_nx_graph(user_nodes, merchant_nodes, edges):
    """Build a NetworkX bipartite graph from sampled data."""
    G = nx.Graph()

    # Add user nodes
    for u in user_nodes:
        G.add_node(f"U{u}", bipartite=0, node_type="user")

    # Add merchant nodes
    for m in merchant_nodes:
        G.add_node(f"M{m}", bipartite=1, node_type="merchant")

    # Add edges
    for u_idx, m_idx, is_fraud in edges:
        G.add_edge(f"U{u_idx}", f"M{m_idx}", fraud=is_fraud)

    return G


def plot_bipartite(G, user_nodes, merchant_nodes, edges, save_path):
    """
    Plot the bipartite graph with:
      - Blue circles  = Users
      - Orange squares = Merchants
      - Grey lines    = Legit transactions
      - Red lines     = Fraud transactions
    """
    fig, ax = plt.subplots(figsize=(18, 10))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    # ── Layout: users on left, merchants on right ──────────────
    pos = {}
    u_list = [f"U{u}" for u in user_nodes]
    m_list = [f"M{m}" for m in merchant_nodes]

    for i, u in enumerate(u_list):
        pos[u] = (-2, i - len(u_list) / 2)
    for i, m in enumerate(m_list):
        pos[m] = (2,  i - len(m_list) / 2)

    # ── Draw edges ─────────────────────────────────────────────
    legit_edges = [(f"U{u}", f"M{m}") for u, m, fraud in edges if fraud == 0]
    fraud_edges = [(f"U{u}", f"M{m}") for u, m, fraud in edges if fraud == 1]

    nx.draw_networkx_edges(G, pos, edgelist=legit_edges,
                           edge_color="#4a9eff", alpha=0.3, width=0.8, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=fraud_edges,
                           edge_color="#ff4444", alpha=0.9, width=2.5, ax=ax,
                           style="solid")

    # ── Draw nodes ─────────────────────────────────────────────
    nx.draw_networkx_nodes(G, pos, nodelist=u_list,
                           node_color="#4a9eff", node_size=600,
                           node_shape="o", ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=m_list,
                           node_color="#ff9f43", node_size=400,
                           node_shape="s", ax=ax)

    # ── Labels ─────────────────────────────────────────────────
    labels = {}
    for u in u_list:
        labels[u] = u
    for m in m_list:
        labels[m] = m[:6]   # truncate long merchant IDs

    nx.draw_networkx_labels(G, pos, labels=labels,
                            font_color="white", font_size=7, ax=ax)

    # ── Legend ─────────────────────────────────────────────────
    legend_elements = [
        mpatches.Patch(color="#4a9eff", label=f"User nodes ({len(user_nodes)})"),
        mpatches.Patch(color="#ff9f43", label=f"Merchant nodes ({len(merchant_nodes)})"),
        mpatches.Patch(color="#4a9eff", alpha=0.4, label=f"Legit transactions ({len(legit_edges)})"),
        mpatches.Patch(color="#ff4444", label=f"Fraud transactions ({len(fraud_edges)})"),
    ]
    ax.legend(handles=legend_elements, loc="upper right",
              facecolor="#1a1d2e", edgecolor="#4a9eff",
              labelcolor="white", fontsize=10)

    # ── Stats box ──────────────────────────────────────────────
    stats_text = (
        f"Total edges: {len(edges)}\n"
        f"Fraud edges: {len(fraud_edges)}\n"
        f"Fraud rate : {len(fraud_edges)/max(len(edges),1)*100:.2f}%"
    )
    ax.text(-2.8, len(u_list)/2 + 0.5, stats_text,
            color="white", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="#1a1d2e", edgecolor="#4a9eff"))

    ax.set_title(
        "Bipartite Graph — User ↔ Merchant Transactions\n"
        "(Sampled subgraph from full 24M-edge graph)",
        color="white", fontsize=14, pad=20
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✅ Plot saved → {save_path}")


def print_graph_stats(data):
    """Print full graph statistics to terminal."""
    print("\n" + "="*55)
    print("  BIPARTITE GRAPH — FULL STATISTICS")
    print("="*55)

    print(f"\n  Node Types:")
    print(f"    {'user':<16} : {data['user'].x.shape[0]:>8,} nodes  | feat_dim={data['user'].x.shape[1]}")
    print(f"    {'merchant':<16} : {data['merchant'].x.shape[0]:>8,} nodes  | feat_dim={data['merchant'].x.shape[1]}")

    print(f"\n  Edge Types (splits):")
    total_edges = 0
    total_fraud = 0
    for rel in data.edge_types:
        ei    = data[rel].edge_index
        el    = data[rel].edge_label
        n     = ei.shape[1]
        fraud = int(el.sum().item())
        total_edges += n
        total_fraud += fraud
        print(f"    {str(rel[1]):<14} : {n:>12,} edges  | "
              f"fraud={fraud:>6,}  ({fraud/n*100:.4f}%)")

    print(f"\n  Totals:")
    print(f"    All edges  : {total_edges:>12,}")
    print(f"    All fraud  : {total_fraud:>12,}  ({total_fraud/total_edges*100:.4f}%)")

    # Degree stats
    ei_train = data["user", "train_txn", "merchant"].edge_index
    user_degree = torch.bincount(ei_train[0], minlength=data["user"].num_nodes)
    merch_degree= torch.bincount(ei_train[1], minlength=data["merchant"].num_nodes)

    print(f"\n  Degree Stats (train split):")
    print(f"    User  avg degree : {user_degree.float().mean():.1f}  "
          f"max={user_degree.max().item():,}")
    print(f"    Merch avg degree : {merch_degree.float().mean():.1f}  "
          f"max={merch_degree.max().item():,}")
    print("="*55)


def run_all(num_users=15, seed=42):
    # 1. Load graph
    data = load_graph()

    # 2. Print full stats
    print_graph_stats(data)

    # 3. Sample + plot subgraph (users=15 for clarity)
    print(f"\nSampling subgraph ({num_users} users)...")
    user_nodes, merchant_nodes, edges = sample_subgraph(data, num_users=num_users, seed=seed)

    G = build_nx_graph(user_nodes, merchant_nodes, edges)

    plot_path = os.path.join(PLOT_DIR, "bipartite_subgraph.png")
    plot_bipartite(G, user_nodes, merchant_nodes, edges, plot_path)

    # 4. Also plot degree distribution
    plot_degrees(data)


def plot_degrees(data):
    """Plot user and merchant degree distributions."""
    ei = data["user", "train_txn", "merchant"].edge_index
    el = data["user", "train_txn", "merchant"].edge_label

    user_deg  = torch.bincount(ei[0], minlength=data["user"].num_nodes).numpy()
    merch_deg = torch.bincount(ei[1], minlength=data["merchant"].num_nodes).numpy()

    # Fraud degree per user
    fraud_mask = (el == 1)
    fraud_user_deg = torch.bincount(
        ei[0][fraud_mask], minlength=data["user"].num_nodes
    ).numpy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor("#0f1117")

    plots = [
        (user_deg,       "User Transaction Count",     "#4a9eff"),
        (merch_deg,      "Merchant Transaction Count",  "#ff9f43"),
        (fraud_user_deg, "Fraud Transactions per User", "#ff4444"),
    ]

    for ax, (data_arr, title, color) in zip(axes, plots):
        ax.set_facecolor("#1a1d2e")
        # Log scale for better visibility
        non_zero = data_arr[data_arr > 0]
        ax.hist(non_zero, bins=50, color=color, alpha=0.8, edgecolor="none", log=True)
        ax.set_title(title, color="white", fontsize=11, pad=10)
        ax.set_xlabel("Count", color="#aaaaaa", fontsize=9)
        ax.set_ylabel("Frequency (log)", color="#aaaaaa", fontsize=9)
        ax.tick_params(colors="#aaaaaa")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")
        stats = f"mean={non_zero.mean():.1f}\nmax={non_zero.max():,}"
        ax.text(0.97, 0.97, stats, transform=ax.transAxes,
                color="white", fontsize=8, va="top", ha="right",
                bbox=dict(boxstyle="round", facecolor="#0f1117", alpha=0.7))

    fig.suptitle("Graph Degree Distributions", color="white", fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "degree_distributions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✅ Degree plot saved → {path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_users", type=int, default=15,
                        help="Number of users to sample in subgraph visualization")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_all(num_users=args.num_users, seed=args.seed)
