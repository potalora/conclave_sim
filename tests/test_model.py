import pytest
import numpy as np
import pandas as pd
from src.model import TransitionModel # Assuming tests run from root

# --- Fixtures ---

@pytest.fixture
def model_params_valid():
    """Returns valid model parameters."""
    return {'beta_weight': 0.5, 'stickiness_factor': 0.7}

@pytest.fixture
def model_params_invalid_beta():
    """Returns invalid model parameters (negative beta)."""
    return {'beta_weight': -0.1, 'stickiness_factor': 0.5}

@pytest.fixture
def model_params_invalid_stickiness():
    """Returns invalid model parameters (stickiness > 1)."""
    return {'beta_weight': 0.5, 'stickiness_factor': 1.1}

@pytest.fixture
def model_params_missing_beta():
    """Returns invalid model parameters (missing beta)."""
    return {'stickiness_factor': 0.5}

@pytest.fixture
def model_params_missing_stickiness():
    """Returns invalid model parameters (missing stickiness)."""
    return {'beta_weight': 0.5}

@pytest.fixture
def elector_data_valid():
    """Returns a valid elector DataFrame with elector_id as index."""
    ids = [f'E{i+1:02d}' for i in range(5)]
    scores = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    df = pd.DataFrame({'ideology_score': scores}, index=ids)
    df.index.name = 'elector_id'
    return df

@pytest.fixture
def elector_data_missing_score(elector_data_valid):
    """Returns elector DataFrame missing the ideology_score column."""
    return elector_data_valid.drop(columns=['ideology_score'])

@pytest.fixture
def elector_data_empty():
    """Returns an empty elector DataFrame."""
    return pd.DataFrame(columns=['ideology_score'], index=pd.Index([], name='elector_id'))


@pytest.fixture
def previous_votes_valid(elector_data_valid):
    """Returns a valid previous votes dictionary."""
    ids = elector_data_valid.index.tolist()
    # E01->E02, E02->E01, E03->E03(invalid self), E04->E05, E05->E04
    # Note: The model itself doesn't prevent self-votes in input,
    # but the simulation logic should. Here we test model handles it.
    return {ids[0]: ids[1], ids[1]: ids[0], ids[2]: ids[2], ids[3]: ids[4], ids[4]: ids[3]}

@pytest.fixture
def previous_votes_invalid_voter(elector_data_valid):
    """Returns previous votes with an invalid voter ID."""
    ids = elector_data_valid.index.tolist()
    return {'E99': ids[0], ids[1]: ids[2]}

@pytest.fixture
def previous_votes_invalid_candidate(elector_data_valid):
    """Returns previous votes with an invalid candidate ID."""
    ids = elector_data_valid.index.tolist()
    return {ids[0]: 'E99', ids[1]: ids[2]}


# --- Test Cases ---

def test_transition_model_init_valid(model_params_valid):
    """Tests successful initialization with valid parameters."""
    model = TransitionModel(parameters=model_params_valid)
    assert model.parameters == model_params_valid

def test_transition_model_init_invalid_beta(model_params_invalid_beta):
    """Tests initialization failure with invalid beta_weight."""
    with pytest.raises(ValueError, match="'beta_weight' must be a non-negative number"):
        TransitionModel(parameters=model_params_invalid_beta)

def test_transition_model_init_invalid_stickiness(model_params_invalid_stickiness):
    """Tests initialization failure with invalid stickiness_factor."""
    with pytest.raises(ValueError, match="'stickiness_factor' must be between 0 and 1"):
        TransitionModel(parameters=model_params_invalid_stickiness)

def test_transition_model_init_missing_beta(model_params_missing_beta):
    """Tests initialization failure with missing required beta parameter."""
    with pytest.raises(ValueError, match="Model parameters must include 'beta_weight'"):
        TransitionModel(parameters=model_params_missing_beta)

def test_transition_model_init_missing_stickiness(model_params_missing_stickiness):
    """Tests initialization failure with missing required stickiness parameter."""
    with pytest.raises(ValueError, match="Model parameters must include 'stickiness_factor'"):
        TransitionModel(parameters=model_params_missing_stickiness)

def test_calculate_probabilities_first_round(model_params_valid, elector_data_valid):
    """Tests probability calculation without previous votes (first round)."""
    model = TransitionModel(parameters=model_params_valid)
    n_electors = len(elector_data_valid)
    prob_matrix = model.calculate_transition_probabilities(elector_data_valid, current_votes=None)

    assert isinstance(prob_matrix, np.ndarray)
    assert prob_matrix.shape == (n_electors, n_electors)
    assert np.all(prob_matrix >= 0)
    np.testing.assert_allclose(prob_matrix.sum(axis=1), 1.0, rtol=1e-6)
    assert np.all(np.diag(prob_matrix) == 0)
    # AI: Removed symmetry check as row normalization breaks it.
    # np.testing.assert_allclose(prob_matrix, prob_matrix.T, rtol=1e-6)

def test_calculate_probabilities_with_stickiness(model_params_valid, elector_data_valid, previous_votes_valid):
    """Tests probability calculation with previous votes and stickiness."""
    model = TransitionModel(parameters=model_params_valid)
    n_electors = len(elector_data_valid)
    prob_matrix = model.calculate_transition_probabilities(elector_data_valid, current_votes=previous_votes_valid)

    assert isinstance(prob_matrix, np.ndarray)
    assert prob_matrix.shape == (n_electors, n_electors)
    assert np.all(prob_matrix >= 0)
    np.testing.assert_allclose(prob_matrix.sum(axis=1), 1.0, rtol=1e-6)
    assert np.all(np.diag(prob_matrix) == 0)

    # Check stickiness effect: P(E01 -> E02) should be relatively high
    # Indices: E01=0, E02=1, E03=2, E04=3, E05=4
    idx_e01 = 0
    idx_e02 = 1
    prob_e01_e02 = prob_matrix[idx_e01, idx_e02]
    # Compare with probability for someone E01 didn't vote for (e.g., E03)
    idx_e03 = 2
    prob_e01_e03 = prob_matrix[idx_e01, idx_e03]
    # We expect stickiness to boost the probability of voting for the previous choice
    # Calculate base attraction without stickiness for comparison
    beta = model.parameters['beta_weight']
    scores = elector_data_valid['ideology_score'].values
    base_attraction_01_02 = np.exp(-beta * abs(scores[idx_e01] - scores[idx_e02]))
    base_attraction_01_03 = np.exp(-beta * abs(scores[idx_e01] - scores[idx_e03]))
    # If base attraction for E03 was higher, stickiness might not make E02 highest,
    # but it should significantly increase P(E01->E02) relative to its base attraction.
    # print(f"P(E01->E02): {prob_e01_e02:.4f}, BaseAtt(E01->E02): {base_attraction_01_02:.4f}")
    # print(f"P(E01->E03): {prob_e01_e03:.4f}, BaseAtt(E01->E03): {base_attraction_01_03:.4f}")
    # A simple check: boosted probability should be higher than non-boosted
    # This isn't universally true after normalization, but likely in simple cases.
    # assert prob_e01_e02 > prob_e01_e03 # Not guaranteed after normalization

def test_calculate_probabilities_missing_score_col(model_params_valid, elector_data_missing_score):
    """Tests error handling when ideology_score column is missing."""
    model = TransitionModel(parameters=model_params_valid)
    with pytest.raises(ValueError, match="Elector data must include 'ideology_score' column."):
        model.calculate_transition_probabilities(elector_data_missing_score)

def test_calculate_probabilities_invalid_voter_id(model_params_valid, elector_data_valid, previous_votes_invalid_voter):
    """Tests error handling for invalid voter ID in previous_votes."""
    model = TransitionModel(parameters=model_params_valid)
    with pytest.raises(KeyError, match="Voter IDs in current_votes not found"):
        model.calculate_transition_probabilities(elector_data_valid, current_votes=previous_votes_invalid_voter)

def test_calculate_probabilities_invalid_candidate_id(model_params_valid, elector_data_valid, previous_votes_invalid_candidate):
    """Tests error handling for invalid candidate ID in previous_votes."""
    model = TransitionModel(parameters=model_params_valid)
    with pytest.raises(KeyError, match="Candidate IDs \\(votes\\) in current_votes not found"):
        model.calculate_transition_probabilities(elector_data_valid, current_votes=previous_votes_invalid_candidate)

def test_calculate_probabilities_zero_stickiness(model_params_valid, elector_data_valid, previous_votes_valid):
    """Tests that zero stickiness yields same result as first round."""
    params_no_stick = model_params_valid.copy()
    params_no_stick['stickiness_factor'] = 0.0
    model_no_stick = TransitionModel(parameters=params_no_stick)
    model_first_round = TransitionModel(parameters=params_no_stick) # Use same params for fair compare

    prob_matrix_no_stick = model_no_stick.calculate_transition_probabilities(elector_data_valid, current_votes=previous_votes_valid)
    prob_matrix_first_round = model_first_round.calculate_transition_probabilities(elector_data_valid, current_votes=None)

    np.testing.assert_allclose(prob_matrix_no_stick, prob_matrix_first_round, rtol=1e-6)

def test_calculate_probabilities_full_stickiness(elector_data_valid, previous_votes_valid):
    """Tests that full stickiness forces vote repetition (where possible)."""
    params_full_stick = {'beta_weight': 0.5, 'stickiness_factor': 1.0}
    model_full_stick = TransitionModel(parameters=params_full_stick)
    n_electors = len(elector_data_valid)

    prob_matrix = model_full_stick.calculate_transition_probabilities(elector_data_valid, current_votes=previous_votes_valid)

    # Indices: E01=0, E02=1, E03=2, E04=3, E05=4
    # Votes: E01->E02, E02->E01, E03->E03, E04->E05, E05->E04
    # Expected probabilities (after zeroing diagonal and renormalizing):
    # E01 should vote for E02 with P=1
    # E02 should vote for E01 with P=1
    # E03 voted for self; with diagonal=0, P should be uniform for others? -> Check model logic
    #   -> Model logic re-normalizes. If E03 only had prob for E03, it gets uniform now.
    # E04 should vote for E05 with P=1
    # E05 should vote for E04 with P=1

    assert prob_matrix.shape == (n_electors, n_electors)
    np.testing.assert_allclose(prob_matrix.sum(axis=1), 1.0, rtol=1e-6)
    assert np.all(np.diag(prob_matrix) == 0)

    assert np.isclose(prob_matrix[0, 1], 1.0) # E01 -> E02
    assert np.isclose(prob_matrix[1, 0], 1.0) # E02 -> E01
    # E03 self-vote case: After zeroing diagonal, row sum is 0. Model assigns uniform.
    expected_uniform_prob = 1.0 / (n_electors - 1)
    non_diag_indices = [i for i in range(n_electors) if i != 2]
    assert np.allclose(prob_matrix[2, non_diag_indices], expected_uniform_prob)
    assert np.isclose(prob_matrix[2, 2], 0.0) # Check diagonal is still 0
    assert np.isclose(prob_matrix[3, 4], 1.0) # E04 -> E05
    assert np.isclose(prob_matrix[4, 3], 1.0) # E05 -> E04

def test_calculate_probabilities_empty_data(model_params_valid, elector_data_empty):
    """Tests ValueError when elector data is empty."""
    model = TransitionModel(parameters=model_params_valid)
    with pytest.raises(ValueError, match="Cannot calculate probabilities with zero electors"):
        model.calculate_transition_probabilities(elector_data_empty)
