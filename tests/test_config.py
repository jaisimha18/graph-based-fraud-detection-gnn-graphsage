"""
Unit tests for configuration constants and paths in aml_config.py.
"""
import os
import pytest
from src.aml_pipeline import aml_config

def test_feature_dimensions():
    """Verify standard node and edge feature dimensions."""
    assert aml_config.ACCOUNT_FEAT_DIM == 8, "Expected 8 account node features"
    assert aml_config.EDGE_FEAT_DIM == 14, "Expected 14 base edge features"

def test_split_strategy_ordering():
    """Verify chronological split invariants: train < val < test."""
    assert aml_config.TRAIN_END_DAY < aml_config.VAL_DAY
    assert aml_config.VAL_DAY < aml_config.TEST_START_DAY

def test_hyperparameters():
    """Verify model hyperparameters are valid and within expected bounds."""
    assert aml_config.HIDDEN_DIM > 0
    assert aml_config.OUTPUT_DIM > 0
    assert 0.0 <= aml_config.DROPOUT < 1.0
    assert aml_config.NUM_SAGE_LAYERS >= 1
    assert aml_config.BATCH_SIZE >= 32
    assert aml_config.LEARNING_RATE > 0
    assert aml_config.EPOCHS > 0
    assert aml_config.SEED == 42

def test_threshold_sweep():
    """Verify threshold sweep list is ascending and bounded between (0, 1)."""
    sweep = aml_config.THRESHOLD_SWEEP
    assert len(sweep) > 0
    assert all(0.0 < t < 1.0 for t in sweep)
    assert sweep == sorted(sweep)
