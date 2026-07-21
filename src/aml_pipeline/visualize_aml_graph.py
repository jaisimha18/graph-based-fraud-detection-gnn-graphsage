import os
import sys
import torch
import numpy as np
import networkx as nx
import matplotlib
# matplotlib.use("Agg") # Removed to allow interactive popup window
import matplotlib.pyplot as plt
from torch_geometric.utils import k_hop_subgraph, to_networkx

# Ensure the root directory is in the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.aml_pipeline.aml_config import AML_OUTPUT_DIR

def visualize_laundering_subgraph(graph_path, save_path, num_hops=2, max_nodes_to_plot=100):
    """
    Loads the full AML graph, extracts a small subgraph around a known
    laundering edge, and visualizes it using NetworkX.
    """
    print(f"Loading graph from {graph_path}...")
    data = torch.load(graph_path, weights_only=False)
    print("Graph loaded successfully.")

    # 1. Find a laundering edge to center the visualization on
    # y == 1 indicates a laundering edge
    laundering_edge_indices = torch.where(data.y == 1)[0]
    
    if len(laundering_edge_indices) == 0:
        print("No laundering edges found in the graph!")
        return

    # Pick the first laundering edge (you could also pick randomly)
    target_edge_idx = laundering_edge_indices[0].item()
    
    # Get the source and destination node IDs for this edge
    src_node = data.edge_index[0, target_edge_idx].item()
    dst_node = data.edge_index[1, target_edge_idx].item()
    
    print(f"Centering subgraph around laundering edge {target_edge_idx}: Node {src_node} -> Node {dst_node}")

    # 2. Extract a k-hop subgraph around these core nodes
    # We use both source and destination as the seed nodes
    subset, edge_index, mapping, edge_mask = k_hop_subgraph(
        node_idx=[src_node, dst_node],
        num_hops=num_hops,
        edge_index=data.edge_index,
        relabel_nodes=True # Relabel nodes to 0..N-1 for the subgraph
    )
    
    # Get the labels for the edges in our subgraph
    subgraph_edge_labels = data.y[edge_mask]

    print(f"Extracted {num_hops}-hop subgraph: {len(subset)} nodes, {edge_index.size(1)} edges.")

    # If the subgraph is too large, it will be a mess to plot. Let's warn the user.
    if len(subset) > max_nodes_to_plot:
        print(f"Warning: Subgraph has {len(subset)} nodes, which might look cluttered.")
        print(f"Consider reducing num_hops or picking a different starting edge.")

    # 3. Create a PyG Data object for the subgraph just to use to_networkx
    from torch_geometric.data import Data
    sub_data = Data(edge_index=edge_index, y=subgraph_edge_labels, num_nodes=len(subset))

    # Convert to NetworkX directed graph
    # We pass edge attributes so we can access the 'y' label when drawing
    G = to_networkx(sub_data, to_undirected=False, edge_attrs=['y'])

    # 4. Set up the plot
    plt.figure(figsize=(14, 10))
    plt.title(f"AML Subgraph Visualization ({num_hops}-hop neighborhood)", fontsize=16)

    # Use a force-directed layout. Spring layout usually works well.
    # Kamada-Kawai is another good option for small graphs.
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

    # 5. Draw the graph components
    
    # Identify which nodes are our "core" nodes in the relabeled subgraph
    core_nodes_subgraph_idx = mapping.tolist()

    # Draw nodes
    # Draw core nodes in a distinct color (e.g., orange) and slightly larger
    node_colors = ['orange' if i in core_nodes_subgraph_idx else 'lightblue' for i in range(len(subset))]
    node_sizes = [600 if i in core_nodes_subgraph_idx else 300 for i in range(len(subset))]
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.9, edgecolors='gray')
    nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif')

    # Draw edges
    # Separate legit and laundering edges for styling
    legit_edges = [(u, v) for u, v, d in G.edges(data=True) if d['y'] == 0]
    laundering_edges = [(u, v) for u, v, d in G.edges(data=True) if d['y'] == 1]

    # Draw legit edges (thin, gray, slightly transparent)
    nx.draw_networkx_edges(
        G, pos, 
        edgelist=legit_edges, 
        edge_color='gray', 
        width=1.0, 
        alpha=0.5, 
        arrowsize=15, 
        connectionstyle='arc3,rad=0.1' # Slight curve for parallel edges
    )

    # Draw laundering edges (thicker, red, solid)
    nx.draw_networkx_edges(
        G, pos, 
        edgelist=laundering_edges, 
        edge_color='red', 
        width=2.5, 
        alpha=0.9, 
        arrowsize=20,
        connectionstyle='arc3,rad=0.1'
    )

    # Add a legend
    import matplotlib.lines as mlines
    legit_line = mlines.Line2D([], [], color='gray', linewidth=1, label='Legit Transaction')
    laundering_line = mlines.Line2D([], [], color='red', linewidth=2.5, label='Laundering Transaction')
    core_node_marker = mlines.Line2D([], [], color='orange', marker='o', linestyle='None', markersize=10, label='Seed Node')
    neighbor_marker = mlines.Line2D([], [], color='lightblue', marker='o', linestyle='None', markersize=10, label='Neighbor Node')
    
    plt.legend(handles=[legit_line, laundering_line, core_node_marker, neighbor_marker], loc='upper right')
    
    plt.axis('off') # Hide axes
    
    # 6. Save the plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Visualization saved successfully to: {save_path}")
    plt.show()
    plt.close()

if __name__ == "__main__":
    graph_file = os.path.join(AML_OUTPUT_DIR, "graph", "aml_graph.pt")
    output_image = os.path.join(AML_OUTPUT_DIR, "graph", "aml_subgraph_viz.png")
    
    # Generate the visualization
    # We use 1 hop by default as 2 hops can sometimes explode to thousands of nodes
    # depending on the graph density.
    visualize_laundering_subgraph(graph_file, output_image, num_hops=1)
