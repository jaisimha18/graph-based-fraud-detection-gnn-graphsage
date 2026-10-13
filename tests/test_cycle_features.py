"""
Unit tests for the Random Walk Cycle Feature Extractor (src/aml_pipeline/cycle_features.py).
"""
import numpy as np
import pytest
from src.aml_pipeline.cycle_features import (
    build_transition_matrix,
    compute_cycle_features_for_edges,
    CYCLE_FEAT_COLS,
)

def test_build_transition_matrix_row_stochastic():
    """Verify that rows with outgoing edges sum to 1.0."""
    num_nodes = 4
    # 0 -> 1, 0 -> 2, 1 -> 2, 2 -> 3
    src = np.array([0, 0, 1, 2], dtype=np.int64)
    dst = np.array([1, 2, 2, 3], dtype=np.int64)

    P = build_transition_matrix(num_nodes, src, dst)
    assert P.shape == (num_nodes, num_nodes)

    # Node 0 has degree 2, each outgoing prob = 0.5
    row_0_sum = float(P[0].sum())
    assert np.isclose(row_0_sum, 1.0)
    assert np.isclose(P[0, 1], 0.5)
    assert np.isclose(P[0, 2], 0.5)

    # Node 3 is a sink, out-degree was 0 (no transitions in row 3)
    assert P[3].sum() == 0.0

def test_2hop_cycle_detection():
    """Verify that a direct reciprocal loop (0 -> 1 -> 0) produces r1 > 0."""
    num_nodes = 2
    src = np.array([0, 1], dtype=np.int64)
    dst = np.array([1, 0], dtype=np.int64)

    # Evaluate on the edge 0 -> 1
    eval_src = np.array([0], dtype=np.int64)
    eval_dst = np.array([1], dtype=np.int64)

    feats = compute_cycle_features_for_edges(num_nodes, src, dst, eval_src, eval_dst)
    assert feats.shape == (1, 8)

    # r1 should be 1.0 (from 1 back to 0 in 1 step)
    r1 = feats[0, 0]
    has_2 = feats[0, 5]
    assert np.isclose(r1, 1.0)
    assert has_2 == 1.0

def test_3hop_cycle_detection():
    """Verify that a 3-node ring (0 -> 1 -> 2 -> 0) produces r2 > 0 (3-hop cycle)."""
    num_nodes = 3
    src = np.array([0, 1, 2], dtype=np.int64)
    dst = np.array([1, 2, 0], dtype=np.int64)

    # Evaluate on edge 0 -> 1
    eval_src = np.array([0], dtype=np.int64)
    eval_dst = np.array([1], dtype=np.int64)

    feats = compute_cycle_features_for_edges(num_nodes, src, dst, eval_src, eval_dst)

    r1 = feats[0, 0] # 2-hop (1 -> 0): does not exist
    r2 = feats[0, 1] # 3-hop (1 -> 2 -> 0): exists with prob 1.0
    has_3 = feats[0, 6]

    assert np.isclose(r1, 0.0)
    assert np.isclose(r2, 1.0)
    assert has_3 == 1.0

def test_acyclic_path_zero_probabilities():
    """Verify that a strictly directed DAG path has 0 cycle return probabilities."""
    num_nodes = 4
    # 0 -> 1 -> 2 -> 3 (no back edges)
    src = np.array([0, 1, 2], dtype=np.int64)
    dst = np.array([1, 2, 3], dtype=np.int64)

    eval_src = np.array([0, 1], dtype=np.int64)
    eval_dst = np.array([1, 2], dtype=np.int64)

    feats = compute_cycle_features_for_edges(num_nodes, src, dst, eval_src, eval_dst)
    assert feats.shape == (2, 8)

    # All return probabilities and indicators should be 0.0
    assert np.all(feats == 0.0)

def test_cycle_feature_columns_count():
    """Verify CYCLE_FEAT_COLS aligns with 8 extracted metrics."""
    assert len(CYCLE_FEAT_COLS) == 8
