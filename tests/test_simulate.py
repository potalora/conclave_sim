# # AI: Unit tests for the simulate module

import pytest
import pandas as pd
import numpy as np

from src.simulate import run_monte_carlo_simulation, REQUIRED_MAJORITY_FRACTION, MAX_ROUNDS
# Mock TransitionModel to avoid dependency on its internal logic during simulate tests
from unittest.mock import MagicMock, patch
import pandas as pd # Ensure pandas is imported for patching Series

# --- Test Fixtures ---
@pytest.fixture
def valid_params():
    """Provides valid parameters for TransitionModel (mocked)."""
    return {'beta_weight': 0.1} # Low beta for more randomness

@pytest.fixture
def sample_elector_data_small():
    """Provides a small elector data set for faster tests."""
    return pd.DataFrame({
        'elector_id': [1, 2, 3], # 3 electors
        'name': ['Elector A', 'Elector B', 'Elector C'],
        'ideology_score': [-0.5, 0.0, 0.5]
    })

@pytest.fixture
def empty_elector_data():
    """Provides an empty elector DataFrame."""
    return pd.DataFrame(columns=['elector_id', 'ideology_score'])

@pytest.fixture
def elector_data_no_id():
    """Provides elector data missing the elector_id column."""
    return pd.DataFrame({'ideology_score': [0.1, 0.2]})

# --- Test Cases ---

# Use patch to replace the actual TransitionModel with a mock
@patch('src.simulate.TransitionModel')
def test_simulation_success_basic(MockTransitionModel, valid_params, sample_elector_data_small):
    """Tests a basic successful run of the simulation."""
    # Configure the mock model to return predictable probabilities
    num_electors = len(sample_elector_data_small)
    mock_instance = MockTransitionModel.return_value
    # Return uniform probabilities for simplicity in this test
    uniform_probs = np.ones((num_electors, num_electors)) / num_electors
    mock_instance.calculate_transition_probabilities.return_value = uniform_probs

    num_simulations = 3
    results_df, aggregate_stats = run_monte_carlo_simulation(
        num_simulations=num_simulations,
        elector_data=sample_elector_data_small,
        model_parameters=valid_params
    )

    # Check DataFrame structure
    assert isinstance(results_df, pd.DataFrame)
    assert len(results_df) == num_simulations
    assert 'simulation_id' in results_df.columns
    assert 'winner_id' in results_df.columns
    assert 'rounds_taken' in results_df.columns
    assert 'status' in results_df.columns

    # Check aggregate stats structure
    assert isinstance(aggregate_stats, dict)
    assert 'success_rate' in aggregate_stats
    assert 'avg_rounds_success' in aggregate_stats # May be None if no success

    # Check model was called
    MockTransitionModel.assert_called_once_with(parameters=valid_params)
    # Check probability calculation was called (at least once per round per sim)
    assert mock_instance.calculate_transition_probabilities.call_count >= num_simulations

# Patch TransitionModel again for error tests
@patch('src.simulate.TransitionModel')
def test_simulation_empty_data(MockTransitionModel, valid_params, empty_elector_data):
    """Tests ValueError when elector data is empty."""
    with pytest.raises(ValueError, match="Elector data cannot be empty"):
        run_monte_carlo_simulation(1, empty_elector_data, valid_params)

@patch('src.simulate.TransitionModel')
def test_simulation_missing_id_column(MockTransitionModel, valid_params, elector_data_no_id):
    """Tests ValueError when elector data is missing elector_id."""
    with pytest.raises(ValueError, match="must contain an 'elector_id' column"):
        run_monte_carlo_simulation(1, elector_data_no_id, valid_params)

# Test specific scenario: Guaranteed winner in one round
@patch('src.simulate.TransitionModel')
def test_simulation_immediate_winner(MockTransitionModel, valid_params, sample_elector_data_small):
    """Tests scenario where a winner is determined in the first round."""
    num_electors = len(sample_elector_data_small)
    required_votes = int(np.ceil(num_electors * REQUIRED_MAJORITY_FRACTION))
    mock_instance = MockTransitionModel.return_value

    # Create probabilities ensuring elector 1 gets all votes
    win_probs = np.zeros((num_electors, num_electors))
    win_probs[:, 0] = 1.0 # All electors vote for candidate with id 1 (index 0)
    mock_instance.calculate_transition_probabilities.return_value = win_probs

    results_df, aggregate_stats = run_monte_carlo_simulation(
        num_simulations=1,
        elector_data=sample_elector_data_small,
        model_parameters=valid_params
    )

    assert len(results_df) == 1
    assert results_df.iloc[0]['winner_id'] == sample_elector_data_small['elector_id'].iloc[0] # Elector ID 1
    assert results_df.iloc[0]['rounds_taken'] == 1
    assert results_df.iloc[0]['status'] == 'Success'
    assert aggregate_stats['success_rate'] == 1.0

# Test specific scenario: Max rounds reached
@patch('src.simulate.TransitionModel')
@patch('pandas.Series.value_counts') # AI: Add patch for value_counts
def test_simulation_max_rounds(
    MockValueCounts, MockTransitionModel, valid_params, sample_elector_data_small
): # AI: Add MockValueCounts arg
    """Tests scenario where max rounds are reached without a winner (forced)."""
    num_electors = len(sample_elector_data_small)
    required_votes = int(np.ceil(num_electors * REQUIRED_MAJORITY_FRACTION))
    mock_tm_instance = MockTransitionModel.return_value

    # Configure TM mock (still needed, can return anything valid)
    uniform_probs = np.ones((num_electors, num_electors)) / num_electors
    mock_tm_instance.calculate_transition_probabilities.return_value = uniform_probs

    # AI: Configure the value_counts mock to *always* return non-winning counts
    # For N=3, required=2. Return counts like {1:1, 2:1, 3:1}
    non_winning_counts = pd.Series({1: 1, 2: 1, 3: 1})
    MockValueCounts.return_value = non_winning_counts

    # AI: Run only one simulation, as the outcome is now forced
    results_df, aggregate_stats = run_monte_carlo_simulation(
        num_simulations=1,
        elector_data=sample_elector_data_small,
        model_parameters=valid_params # Params don't matter much now
    )

    # AI: Check the single simulation result
    assert len(results_df) == 1
    result = results_df.iloc[0]
    assert result['status'] == 'Max Rounds Reached'
    assert result['rounds_taken'] == MAX_ROUNDS
    assert pd.isna(result['winner_id'])
    assert aggregate_stats['num_max_rounds_reached'] == 1
    assert aggregate_stats['success_rate'] == 0.0

    # Ensure value_counts was called MAX_ROUNDS times
    assert MockValueCounts.call_count == MAX_ROUNDS

    # Ensure TransitionModel was still called
    assert mock_tm_instance.calculate_transition_probabilities.call_count == MAX_ROUNDS
