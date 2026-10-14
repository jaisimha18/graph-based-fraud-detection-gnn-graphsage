"""
Unit tests for GNN model architectures:
- Baseline AMLGraphSAGEModel
- Cycle-Aware GraphSAGE (CycleAwareAMLModel with both Ablation modes)
"""
import torch
import pytest
from src.models.graphsage_model import AMLGraphSAGEModel
from src.models.cycle_graphsage_model import CycleAwareAMLModel, CycleAwareSAGEConv

def test_baseline_graphsage_forward_pass():
    """Verify standard GraphSAGE model instantiates and runs forward pass."""
    num_nodes = 5
    num_edges = 6
    node_dim = 8
    edge_dim = 14

    model = AMLGraphSAGEModel(
        node_feat_dim=node_dim,
        edge_feat_dim=edge_dim,
        hidden_dim=32,
        out_dim=16,
        num_layers=2,
        dropout=0.1
    )
    model.eval()

    x = torch.randn(num_nodes, node_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, edge_dim)
    edge_label_index = edge_index

    with torch.no_grad():
        logits = model(x, edge_index, edge_attr, edge_label_index)

    assert logits.shape == (num_edges,)
    assert not torch.isnan(logits).any()

def test_cycle_aware_sage_conv_layer():
    """Verify custom CycleAwareSAGEConv layer propagates messages with edge cycle attention."""
    num_nodes = 4
    num_edges = 4
    in_dim = 16
    out_dim = 32
    edge_dim = 22

    conv = CycleAwareSAGEConv(in_channels=in_dim, out_channels=out_dim, edge_channels=edge_dim)
    conv.eval()

    x = torch.randn(num_nodes, in_dim)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    edge_attr = torch.randn(num_edges, edge_dim)

    with torch.no_grad():
        out = conv(x, edge_index, edge_attr)

    assert out.shape == (num_nodes, out_dim)
    assert not torch.isnan(out).any()

def test_cycle_aware_model_ablation_modes():
    """Verify CycleAwareAMLModel in both use_cycle_attention=True and False modes."""
    num_nodes = 6
    num_edges = 8
    node_dim = 8
    edge_dim = 22

    for use_attn in [True, False]:
        model = CycleAwareAMLModel(
            node_feat_dim=node_dim,
            edge_feat_dim=edge_dim,
            hidden_dim=32,
            out_dim=16,
            use_cycle_attention=use_attn,
            num_layers=2,
            dropout=0.1
        )
        model.eval()

        x = torch.randn(num_nodes, node_dim)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        batch_edge_attr = torch.randn(num_edges, edge_dim)
        edge_label_index = edge_index
        subgraph_edge_attr = batch_edge_attr

        with torch.no_grad():
            logits = model(
                x,
                edge_index,
                batch_edge_attr,
                edge_label_index,
                subgraph_edge_attr=subgraph_edge_attr
            )

        assert logits.shape == (num_edges,)
        assert not torch.isnan(logits).any()
