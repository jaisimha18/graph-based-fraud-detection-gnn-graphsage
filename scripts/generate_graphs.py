import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import random

def generate_sample_graph():
    print("Loading data for sample graph...")
    df = pd.read_csv("AML_DATASET/HI-Small_Trans.csv")
    
    # Get some laundering edges
    fraud_df = df[df['Is Laundering'] == 1].head(3)
    
    # Get some normal edges that connect to these fraud nodes if possible, or just random
    fraud_nodes = set(fraud_df['Account']).union(set(fraud_df['Account.1']))
    normal_df = df[(df['Is Laundering'] == 0) & (df['Account'].isin(fraud_nodes) | df['Account.1'].isin(fraud_nodes))].head(12)
    
    if len(normal_df) < 5:
        normal_df = df[df['Is Laundering'] == 0].sample(12, random_state=42)
        
    sample_df = pd.concat([fraud_df, normal_df])
    
    G = nx.from_pandas_edgelist(sample_df, source='Account', target='Account.1', edge_attr='Is Laundering', create_using=nx.DiGraph())
    
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=800, edgecolors='black')
    
    # Draw edges
    normal_edges = [(u, v) for (u, v, d) in G.edges(data=True) if d['Is Laundering'] == 0]
    fraud_edges = [(u, v) for (u, v, d) in G.edges(data=True) if d['Is Laundering'] == 1]
    
    nx.draw_networkx_edges(G, pos, edgelist=normal_edges, edge_color='gray', arrows=True, arrowsize=20)
    nx.draw_networkx_edges(G, pos, edgelist=fraud_edges, edge_color='red', arrows=True, arrowsize=25, width=2)
    
    # Draw labels
    labels = {node: str(node)[-4:] for node in G.nodes()} # Just show last 4 digits for cleanliness
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_weight='bold')
    
    plt.title("Sample Transaction Network (15 Transactions)\nRed = Suspicious", fontsize=14, pad=15)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("sample_graph.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Created sample_graph.png")

def generate_cycle_graph():
    print("Generating cycle graph...")
    # Create a synthetic cycle using realistic IDs
    nodes = ['80001234', '80005678', '80009012', '80003456']
    
    G = nx.DiGraph()
    G.add_edges_from([
        (nodes[0], nodes[1]),
        (nodes[1], nodes[2]),
        (nodes[2], nodes[3]),
        (nodes[3], nodes[0]),
    ])
    
    # Add a couple of branching nodes to make it look like a real subgraph
    G.add_edges_from([
        ('80009999', nodes[0]),
        (nodes[1], '80008888'),
        (nodes[2], '80007777')
    ])
    
    plt.figure(figsize=(8, 6))
    pos = nx.circular_layout(G.subgraph(nodes))
    # Place the extra nodes outside the circle
    pos['80009999'] = [pos[nodes[0]][0] - 0.5, pos[nodes[0]][1] + 0.5]
    pos['80008888'] = [pos[nodes[1]][0] + 0.5, pos[nodes[1]][1] + 0.5]
    pos['80007777'] = [pos[nodes[2]][0] + 0.5, pos[nodes[2]][1] - 0.5]
    
    nx.draw_networkx_nodes(G, pos, node_color='#FFB6C1', node_size=1200, edgecolors='black')
    
    # Highlight the cycle edges
    cycle_edges = [(nodes[0], nodes[1]), (nodes[1], nodes[2]), (nodes[2], nodes[3]), (nodes[3], nodes[0])]
    other_edges = [e for e in G.edges() if e not in cycle_edges]
    
    nx.draw_networkx_edges(G, pos, edgelist=cycle_edges, edge_color='red', arrows=True, arrowsize=25, width=2.5, connectionstyle='arc3,rad=0.1')
    nx.draw_networkx_edges(G, pos, edgelist=other_edges, edge_color='gray', arrows=True, arrowsize=15, style='dashed')
    
    labels = {node: str(node)[-4:] for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_weight='bold')
    
    plt.title("Detected Cycle (Circular Money Flow)", fontsize=14, pad=15)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("cycle_awareness.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Created cycle_awareness.png")

if __name__ == "__main__":
    generate_sample_graph()
    generate_cycle_graph()
