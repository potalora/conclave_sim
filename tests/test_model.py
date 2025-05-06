# # AI: Unit tests for the model module

import pytest
import numpy as np
import pandas as pd

# Assuming the test is run from the root directory or pytest handles paths
from src.model import TransitionModel

# --- Test Fixtures ---
@pytest.fixture
def valid_params():
    """Provides valid parameters for TransitionModel."""
    return {'beta_weight': 0.5, 'other_param': 'value'}

@pytest.fixture
def invalid_params():
    """Provides invalid parameters (missing beta_weight)."""
    return {'other_param': 'value'}

@pytest.fixture
def sample_elector_data():
    """Provides sample elector data DataFrame for testing."""
    return pd.DataFrame({
        'elector_id': [1, 2, 3, 4],
        'name': ['Elector A', 'Elector B', 'Elector C', 'Elector D'],
        'region': ['Europe', 'Asia', 'Africa', 'Americas'],
        'ideology_score': [-0.8, -0.5, 0.1, 0.9]
    })

@pytest.fixture
def elector_data_missing_ideology():
    """Provides elector data missing the ideology_score column."""
    return pd.DataFrame({
        'elector_id': [1, 2],
        'name': ['Elector A', 'Elector B']
    })

# --- Test Cases ---

def test_transition_model_init_success(valid_params):
    """Tests successful initialization of TransitionModel."""
    model = TransitionModel(parameters=valid_params)
    assert model.parameters == valid_params

def test_transition_model_init_fail_missing_beta(invalid_params):
    """Tests ValueError on initialization if beta_weight is missing."""
    with pytest.raises(ValueError, match="must include 'beta_weight'"):
        TransitionModel(parameters=invalid_params)

def test_calculate_probabilities_success(valid_params, sample_elector_data):
    """Tests successful calculation of transition probabilities."""
    model = TransitionModel(parameters=valid_params)
    dummy_votes = pd.DataFrame() # Not used in current implementation
    probabilities = model.calculate_transition_probabilities(dummy_votes, sample_elector_data)

    num_electors = len(sample_elector_data)
    assert isinstance(probabilities, np.ndarray)
    assert probabilities.shape == (num_electors, num_electors)
    # Check that probabilities are non-negative
    assert np.all(probabilities >= 0)
    # Check that probabilities sum to approximately 1 for each elector (row-wise)
    assert np.allclose(probabilities.sum(axis=1), 1.0)

def test_calculate_probabilities_empty_data(valid_params):
    """Tests ValueError when elector data is empty."""
    model = TransitionModel(parameters=valid_params)
    empty_df = pd.DataFrame(columns=['elector_id', 'ideology_score'])
    dummy_votes = pd.DataFrame()
    with pytest.raises(ValueError, match="Cannot calculate probabilities with zero electors"):
        model.calculate_transition_probabilities(dummy_votes, empty_df)

def test_calculate_probabilities_missing_column(valid_params, elector_data_missing_ideology):
    """Tests ValueError when elector data is missing required columns."""
    model = TransitionModel(parameters=valid_params)
    dummy_votes = pd.DataFrame()
    with pytest.raises(ValueError, match="must contain 'elector_id' and 'ideology_score'"):
        model.calculate_transition_probabilities(dummy_votes, elector_data_missing_ideology)

# Optional: Test specific probability values based on known ideologies
def test_calculate_probabilities_values(valid_params):
    """Tests the relative probability values based on ideology."""
    model = TransitionModel(parameters=valid_params)
    # Simple data: Elector 1 is far left, Elector 2 is center, Elector 3 is far right
    test_data = pd.DataFrame({
        'elector_id': [1, 2, 3],
        'ideology_score': [-1.0, 0.0, 1.0]
    })
    dummy_votes = pd.DataFrame()
    probabilities = model.calculate_transition_probabilities(dummy_votes, test_data)

    # Expect Elector 1 (idx 0) to have highest probability of voting for self (col 0)
    assert np.argmax(probabilities[0, :]) == 0
    # Expect Elector 1 (idx 0) to have lower probability for Elector 3 (col 2) than Elector 2 (col 1)
    assert probabilities[0, 2] < probabilities[0, 1]

    # Expect Elector 2 (idx 1) to have highest probability for self (col 1)
    assert np.argmax(probabilities[1, :]) == 1
    # Expect Elector 2 (idx 1) to have similar (and lower than self) probs for 1 (col 0) and 3 (col 2)
    assert np.isclose(probabilities[1, 0], probabilities[1, 2])
    assert probabilities[1, 0] < probabilities[1, 1]
