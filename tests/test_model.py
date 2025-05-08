import pytest
import numpy as np
import pandas as pd
import logging
from src.model import TransitionModel  # Assuming tests run from root

# --- Fixtures ---


@pytest.fixture
def elector_data_valid():
    """Returns a valid elector DataFrame with elector_id as index, ideology_score, region, and is_papabile."""
    ids = [f"E{i+1:02d}" for i in range(5)]  # E01, E02, E03, E04, E05
    scores = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    regions = ["Europe", "Asia", "Europe", "Americas", "Asia"]
    papabile_status = [True, False, True, False, True]  # E01, E03, E05 are papabile
    df = pd.DataFrame(
        {"ideology_score": scores, "region": regions, "is_papabile": papabile_status},
        index=ids,
    )
    df.index.name = "elector_id"
    return df


@pytest.fixture
def elector_data_missing_ideology_score(elector_data_valid):
    """Returns elector DataFrame missing the ideology_score column."""
    return elector_data_valid.drop(columns=["ideology_score"])


@pytest.fixture
def elector_data_missing_region(elector_data_valid):
    """Returns elector DataFrame missing the region column."""
    return elector_data_valid.drop(columns=["region"])


@pytest.fixture
def elector_data_missing_papabile(elector_data_valid):
    """Returns elector DataFrame missing the is_papabile column."""
    return elector_data_valid.drop(columns=["is_papabile"])


@pytest.fixture
def elector_data_empty():
    """Returns an empty elector DataFrame."""
    return pd.DataFrame(
        columns=["ideology_score", "region", "is_papabile"],
        index=pd.Index([], name="elector_id"),
    )


@pytest.fixture
def previous_votes_valid(elector_data_valid):
    """Returns a valid previous votes dictionary."""
    ids = elector_data_valid.index.tolist()
    # E01->E02, E02->E01, E03->E03(invalid self), E04->E05, E05->E04
    # Note: The model itself doesn't prevent self-votes in input,
    # but the simulation logic should. Here we test model handles it.
    return {
        ids[0]: ids[1],
        ids[1]: ids[0],
        ids[2]: ids[2],
        ids[3]: ids[4],
        ids[4]: ids[3],
    }


@pytest.fixture
def previous_votes_invalid_voter(elector_data_valid):
    """Returns previous votes with an invalid voter ID."""
    ids = elector_data_valid.index.tolist()
    return {"E99": ids[0], ids[1]: ids[2]}


@pytest.fixture
def previous_votes_invalid_candidate(elector_data_valid):
    """Returns previous votes with an invalid candidate ID."""
    ids = elector_data_valid.index.tolist()
    return {ids[0]: "E99", ids[1]: ids[2]}


# --- Test Cases ---


def test_transition_model_init_valid_params(elector_data_valid):
    """Tests successful initialization with valid parameters, including defaults and overrides."""
    # Test with minimal required parameters, relying on defaults for others
    # beta_weight and stickiness_factor are technically not *required* if defaults are desired,
    # but the constructor requires them explicitly for now.
    model_defaults = TransitionModel(
        elector_data=elector_data_valid, beta_weight=0.5, stickiness_factor=0.7
    )
    assert model_defaults.beta_weight == 0.5
    assert model_defaults.stickiness_factor == 0.7
    assert model_defaults.bandwagon_strength == 0.0  # Default
    assert model_defaults.regional_affinity_bonus == 0.1  # Default
    assert model_defaults.papabile_weight_factor == 1.5  # Default
    pd.testing.assert_frame_equal(model_defaults.elector_data_full, elector_data_valid)

    # Test with all parameters specified
    model_all_params = TransitionModel(
        elector_data=elector_data_valid,
        beta_weight=1.0,
        stickiness_factor=0.2,
        bandwagon_strength=0.3,
        regional_affinity_bonus=0.4,
        papabile_weight_factor=1.8,
    )
    assert model_all_params.beta_weight == 1.0
    assert model_all_params.stickiness_factor == 0.2
    assert model_all_params.bandwagon_strength == 0.3
    assert model_all_params.regional_affinity_bonus == 0.4
    assert model_all_params.papabile_weight_factor == 1.8


@pytest.mark.parametrize(
    "param_name,invalid_value,error_message_match",
    [
        ("beta_weight", -0.1, "beta_weight must be non-negative."),
        ("stickiness_factor", -0.1, "stickiness_factor must be between 0 and 1."),
        ("stickiness_factor", 1.1, "stickiness_factor must be between 0 and 1."),
        ("bandwagon_strength", -0.1, "bandwagon_strength must be non-negative."),
        (
            "regional_affinity_bonus",
            -0.1,
            "regional_affinity_bonus must be non-negative.",
        ),
        (
            "papabile_weight_factor",
            -0.1,
            "papabile_weight_factor must be non-negative.",
        ),
    ],
)
def test_transition_model_init_invalid_numeric_params(
    elector_data_valid, param_name, invalid_value, error_message_match
):
    """Tests initialization failure with invalid numeric model parameters."""
    params = {
        "elector_data": elector_data_valid,
        "beta_weight": 1.0,
        "stickiness_factor": 0.5,
        "bandwagon_strength": 0.0,
        "regional_affinity_bonus": 0.1,
        "papabile_weight_factor": 1.5,
    }
    params[param_name] = invalid_value
    with pytest.raises(ValueError, match=error_message_match):
        TransitionModel(**params)


def test_transition_model_init_elector_data_not_dataframe():
    """Tests initialization failure if elector_data is not a DataFrame."""
    with pytest.raises(ValueError, match="elector_data must be a pandas DataFrame."):
        # Provide all required numeric params to isolate the elector_data type error
        TransitionModel(
            elector_data="not_a_dataframe", beta_weight=1.0, stickiness_factor=0.5
        )


@pytest.mark.parametrize(
    "missing_col_fixture_name,missing_col_name",
    [
        ("elector_data_missing_ideology_score", "ideology_score"),
        ("elector_data_missing_region", "region"),
        ("elector_data_missing_papabile", "is_papabile"),
    ],
)
def test_transition_model_init_elector_data_missing_column(
    request, missing_col_fixture_name, missing_col_name, elector_data_valid
):
    """Tests initialization failure if elector_data is missing required columns."""
    elector_data_missing_col = request.getfixturevalue(missing_col_fixture_name)
    with pytest.raises(
        ValueError, match=f"Elector data must contain a '{missing_col_name}' column."
    ):
        # Provide all required numeric params to isolate the column error
        TransitionModel(
            elector_data=elector_data_missing_col,
            beta_weight=1.0,
            stickiness_factor=0.5,
        )


def test_transition_model_init_elector_data_index_name_warning(
    elector_data_valid, caplog
):
    """Tests warning if elector_data index is not named 'elector_id'."""
    elector_data_wrong_index_name = elector_data_valid.copy()
    elector_data_wrong_index_name.index.name = "wrong_name"
    with caplog.at_level(logging.WARNING):
        TransitionModel(
            elector_data=elector_data_wrong_index_name,
            beta_weight=1.0,
            stickiness_factor=0.5,
        )
    assert "Elector data index is not named 'elector_id'." in caplog.text


def test_calculate_probabilities_first_round(elector_data_valid):
    """Tests probability calculation without previous votes (first round).
    Uses default model parameters for bandwagon, regional, and papabile effects.
    """
    model = TransitionModel(
        elector_data=elector_data_valid, beta_weight=0.5, stickiness_factor=0.7
    )
    n_electors = len(elector_data_valid)
    prob_matrix, _ = model.calculate_transition_probabilities(
        elector_data_runtime=elector_data_valid, current_votes=None
    )

    assert isinstance(prob_matrix, np.ndarray)
    assert prob_matrix.shape == (n_electors, n_electors)
    assert np.all(prob_matrix >= 0)
    np.testing.assert_allclose(prob_matrix.sum(axis=1), 1.0, rtol=1e-6)
    # Check that diagonal is zero (no self-votes)
    for i in range(n_electors):
        assert np.isclose(
            prob_matrix[i, i], 0
        ), f"Diagonal element at ({i},{i}) is not zero."


def test_calculate_probabilities_with_stickiness(
    elector_data_valid, previous_votes_valid
):
    """Tests probability calculation with previous votes and stickiness.
    Uses default model parameters for bandwagon, regional, and papabile effects.
    """
    model = TransitionModel(
        elector_data=elector_data_valid, beta_weight=0.5, stickiness_factor=0.7
    )
    n_electors = len(elector_data_valid)
    prob_matrix, _ = model.calculate_transition_probabilities(
        elector_data_runtime=elector_data_valid, current_votes=previous_votes_valid
    )

    assert isinstance(prob_matrix, np.ndarray)
    assert prob_matrix.shape == (n_electors, n_electors)
    assert np.all(prob_matrix >= 0)
    np.testing.assert_allclose(prob_matrix.sum(axis=1), 1.0, rtol=1e-6)
    for i in range(n_electors):
        assert np.isclose(
            prob_matrix[i, i], 0
        ), f"Diagonal element at ({i},{i}) is not zero."

    # Qualitative check for stickiness: E01 voted for E02. P(E01->E02) should be relatively high.
    # Elector E01 is at index 0, E02 at index 1.
    # This is a qualitative check, exact values depend on interplay of all factors.
    # A more robust check would compare with a no-stickiness scenario or verify amplification.
    # For now, just ensure the calculation runs and adheres to basic probability rules.
    # idx_e01 = elector_data_valid.index.get_loc("E01") # AI: Removing as unused
    # idx_e02 = elector_data_valid.index.get_loc("E02") # AI: Removing as unused
    # print(f"Stickiness test: P(E01->E02) = {prob_matrix[idx_e01, idx_e02]}")
    # Example: Check it's higher than if E01 voted for E03 (who E01 didn't vote for)
    # idx_e03 = elector_data_valid.index.get_loc('E03')
    # if previous_votes_valid.get('E01') != 'E03': # Ensure comparison is fair
    # assert prob_matrix[idx_e01, idx_e02] > prob_matrix[idx_e01, idx_e03] # Not guaranteed


def test_calculate_probabilities_invalid_voter_id(
    elector_data_valid, previous_votes_invalid_voter
):
    """Tests error handling for invalid voter ID in previous_votes.
    Note: The model's calculate_transition_probabilities itself might not raise KeyError
    if a voter ID is in current_votes but not in elector_data_full.index during iteration.
    It would skip stickiness for that voter. Let's verify behavior.
    The original model's design might silently ignore this.
    The current model.py line: `for elector_idx, elector_id in enumerate(self.elector_data_full.index):`
    means it only iterates through known electors. So an invalid voter in `current_votes` is ignored.
    This test might need to be re-evaluated based on desired strictness.
    For now, let's assume it runs without error if invalid voter is just extra in current_votes.
    If an elector in self.elector_data_full.index IS NOT in current_votes, that's fine (no stickiness).
    """
    model = TransitionModel(
        elector_data=elector_data_valid, beta_weight=0.5, stickiness_factor=0.7
    )
    # This should run without error, as the loop iterates over model's known electors.
    # Stickiness for 'E99' won't be applied as 'E99' is not in self.elector_data_full.index.
    prob_matrix, _ = model.calculate_transition_probabilities(
        elector_data_runtime=elector_data_valid,
        current_votes=previous_votes_invalid_voter,
    )
    assert prob_matrix is not None  # Basic check that calculation completed


def test_calculate_probabilities_invalid_candidate_id(
    elector_data_valid, previous_votes_invalid_candidate
):
    """Tests error handling for invalid candidate ID in previous_votes.
    The model should handle this gracefully by not finding the candidate and thus not applying stickiness for that vote.
    """
    model = TransitionModel(
        elector_data=elector_data_valid, beta_weight=0.5, stickiness_factor=0.7
    )
    # Should run without error. The try-except ValueError in stickiness logic handles this.
    prob_matrix, _ = model.calculate_transition_probabilities(
        elector_data_runtime=elector_data_valid,
        current_votes=previous_votes_invalid_candidate,
    )
    assert prob_matrix is not None  # Basic check that calculation completed


def test_calculate_probabilities_zero_stickiness(
    elector_data_valid, previous_votes_valid
):
    """Tests that zero stickiness (when other effects active) behaves as expected.
    Compare to first round, but note other effects (papabile, regional, bandwagon default 0) are still on.
    """
    model_zero_stick = TransitionModel(
        elector_data=elector_data_valid,
        beta_weight=0.5,
        stickiness_factor=0.0,  # Key: zero stickiness
        regional_affinity_bonus=0.1,  # Default
        papabile_weight_factor=1.5,  # Default
        bandwagon_strength=0.0,  # Default
    )
    prob_matrix_zero_stick, _ = model_zero_stick.calculate_transition_probabilities(
        elector_data_runtime=elector_data_valid, current_votes=previous_votes_valid
    )

    model_first_round_comparable = TransitionModel(
        elector_data=elector_data_valid,
        beta_weight=0.5,
        stickiness_factor=0.0,  # Effectively no stickiness for comparison
        regional_affinity_bonus=0.1,  # Default
        papabile_weight_factor=1.5,  # Default
        bandwagon_strength=0.0,  # Default
    )
    prob_matrix_first_round, _ = (
        model_first_round_comparable.calculate_transition_probabilities(
            elector_data_runtime=elector_data_valid, current_votes=None
        )
    )

    # With bandwagon_strength=0, regional and papabile effects active,
    # zero stickiness with votes should be identical to first round (no votes).
    np.testing.assert_allclose(
        prob_matrix_zero_stick, prob_matrix_first_round, rtol=1e-6
    )


def test_calculate_probabilities_full_stickiness(
    elector_data_valid, previous_votes_valid
):
    """Tests that full stickiness forces vote repetition (where possible and other factors allow).
    This is a qualitative check, as other factors (ideology, regional, papabile) also play a role.
    """
    model_full_stick = TransitionModel(
        elector_data=elector_data_valid,
        beta_weight=0.5,  # Non-zero ideology effect
        stickiness_factor=1.0,  # Full stickiness
        regional_affinity_bonus=0.0,  # Turn off for simplicity here
        papabile_weight_factor=1.0,  # Turn off for simplicity here
        bandwagon_strength=0.0,  # Turn off for simplicity here
    )
    n_electors = len(elector_data_valid)
    prob_matrix, _ = model_full_stick.calculate_transition_probabilities(
        elector_data_runtime=elector_data_valid, current_votes=previous_votes_valid
    )

    assert prob_matrix.shape == (n_electors, n_electors)
    np.testing.assert_allclose(prob_matrix.sum(axis=1), 1.0, rtol=1e-6)
    for i in range(n_electors):
        assert np.isclose(
            prob_matrix[i, i], 0
        ), f"Diagonal element at ({i},{i}) is not zero."

    # Elector E01 (idx 0) voted for E02 (idx 1). P(E01->E02) should be dominant.
    # Elector E02 (idx 1) voted for E01 (idx 0). P(E02->E01) should be dominant.
    # Elector E03 (idx 2) voted for E03 (self). Vote gets redistributed. Check row sum is 1 and prob_matrix[2,2]==0.
    # Elector E04 (idx 3) voted for E05 (idx 4). P(E04->E05) should be dominant.
    # Elector E05 (idx 4) voted for E04 (idx 3). P(E05->E04) should be dominant.

    e_map = {name: i for i, name in enumerate(elector_data_valid.index)}

    # Check E01 -> E02 is highest prob for E01
    if "E01" in previous_votes_valid and previous_votes_valid["E01"] == "E02":
        assert prob_matrix[e_map["E01"], e_map["E02"]] == np.max(
            prob_matrix[e_map["E01"], :]
        )

    # Check E02 -> E01 is highest prob for E02
    if "E02" in previous_votes_valid and previous_votes_valid["E02"] == "E01":
        assert prob_matrix[e_map["E02"], e_map["E01"]] == np.max(
            prob_matrix[e_map["E02"], :]
        )

    # For E03, who voted for self (E03): prob_matrix[e_map['E03'], e_map['E03']] must be 0.
    # The other probabilities in that row must sum to 1.
    assert np.isclose(prob_matrix[e_map["E03"], e_map["E03"]], 0.0)
    # Check if other probabilities in row E03 are plausible (e.g. not all zero if other cands exist)
    if n_electors > 1:
        assert not np.all(
            np.isclose(prob_matrix[e_map["E03"], : e_map["E03"]], 0.0)
        ) or not np.all(np.isclose(prob_matrix[e_map["E03"], e_map["E03"] + 1 :], 0.0))

    # Check E04 -> E05 is highest prob for E04
    if "E04" in previous_votes_valid and previous_votes_valid["E04"] == "E05":
        assert prob_matrix[e_map["E04"], e_map["E05"]] == np.max(
            prob_matrix[e_map["E04"], :]
        )

    # Check E05 -> E04 is highest prob for E05
    if "E05" in previous_votes_valid and previous_votes_valid["E05"] == "E04":
        assert prob_matrix[e_map["E05"], e_map["E04"]] == np.max(
            prob_matrix[e_map["E05"], :]
        )


@pytest.fixture
def elector_data_very_small():
    # Fixture for testing edge cases with 1 or 2 electors, if needed later.
    # For now, not used directly by the refactored test below.
    data = {
        "elector_id": ["E01"],
        "ideology_score": [0.5],
        "region": ["Europe"],
        "is_papabile": [False],
    }
    return pd.DataFrame(data).set_index("elector_id")


def test_transition_model_init_empty_elector_data_logs_warning(
    elector_data_empty, caplog
):
    """Tests TransitionModel initialization with empty elector data and subsequent calculation."""
    # __init__ should not raise an error for an empty DataFrame with correct columns.
    # The logging of empty data occurs in calculate_transition_probabilities.
    model = TransitionModel(
        elector_data=elector_data_empty, beta_weight=0.5, stickiness_factor=0.1
    )
    assert model is not None  # Check that model was created
    # No specific __init__ warning for *just* empty data if columns are present.
    # assert "TransitionModel initialized with empty elector data." in caplog.text # This log doesn't exist in current __init__

    # Test calculate_transition_probabilities with this model
    # It should handle the zero-elector case gracefully (e.g., return empty array and log)
    with caplog.at_level(
        logging.WARNING
    ):  # Capture warnings from calculate_transition_probabilities
        prob_matrix, details = model.calculate_transition_probabilities(
            elector_data_runtime=elector_data_empty,  # Pass it again for runtime consistency if needed by method signature
            current_votes=None,
        )
    assert prob_matrix.shape == (0, 0)  # Expecting a 0x0 array
    assert isinstance(details, list)
    assert not details
    assert (
        "No effective candidates to calculate transition probabilities for."
        in caplog.text
    )


def test_papabile_weight_factor_effect(elector_data_valid):
    """Tests that papabile_weight_factor correctly increases preference for papabile candidates."""
    beta_weight = 0.5
    papabile_factor_baseline = 1.0
    papabile_factor_increased = 1.5

    # Baseline model: papabile factor is neutral
    model_baseline = TransitionModel(
        elector_data=elector_data_valid,
        beta_weight=beta_weight,
        papabile_weight_factor=papabile_factor_baseline,
        regional_affinity_bonus=0.0,  # Isolate papabile effect
        bandwagon_strength=0.0,  # Isolate papabile effect
        stickiness_factor=0.0        # Isolate papabile effect (also no current_votes)
    )
    probs_baseline, _ = model_baseline.calculate_transition_probabilities(
        elector_data_runtime=elector_data_valid, current_votes=None
    )

    # Model with increased papabile factor
    model_increased_papabile = TransitionModel(
        elector_data=elector_data_valid,
        beta_weight=beta_weight,
        papabile_weight_factor=papabile_factor_increased,
        regional_affinity_bonus=0.0,
        bandwagon_strength=0.0,
        stickiness_factor=0.0        # Isolate papabile effect (also no current_votes)
    )
    probs_increased, _ = model_increased_papabile.calculate_transition_probabilities(
        elector_data_runtime=elector_data_valid, current_votes=None
    )

    # Based on the actual elector_data_valid fixture:
    # E01: ideology=-2.0, is_papabile=True  (Voter)
    # E02: ideology=-1.0, is_papabile=False (Non-Papabile Candidate)
    # E03: ideology= 0.0, is_papabile=True  (Papabile Candidate)
    # E04: ideology= 1.0, is_papabile=False
    # E05: ideology= 2.0, is_papabile=True

    idx_e01 = elector_data_valid.index.get_loc("E01")  # Voter
    idx_e02 = elector_data_valid.index.get_loc("E02")  # Non-Papabile Candidate
    idx_e03 = elector_data_valid.index.get_loc("E03")  # Papabile Candidate

    # For E01 voting for E03 (papabile):
    # Probability should be higher when papabile_weight_factor is increased.
    prob_e01_e03_baseline = probs_baseline[idx_e01, idx_e03]
    prob_e01_e03_increased = probs_increased[idx_e01, idx_e03]
    msg_e01_e03_prob = (
        f"P(E01->E03) with papabile factor {papabile_factor_increased} ({prob_e01_e03_increased:.4f}) "
        f"not greater than with factor {papabile_factor_baseline} ({prob_e01_e03_baseline:.4f})"
    )
    assert prob_e01_e03_increased > prob_e01_e03_baseline, msg_e01_e03_prob

    # For E01 voting for E02 (non-papabile):
    # Probability may decrease or stay same, but ratio P(papabile)/P(non-papabile) should increase.
    prob_e01_e02_baseline = probs_baseline[idx_e01, idx_e02]
    prob_e01_e02_increased = probs_increased[idx_e01, idx_e02]

    # Avoid division by zero if a probability is zero (e.g. if a candidate is the elector itself, though handled by fill_diagonal)
    # However, with varied ideologies, direct zero is unlikely unless beta is huge or scores are identical.
    # For this test, we assume non-zero probabilities for selected candidates prior to papabile factor.
    ratio_baseline = (
        prob_e01_e03_baseline / prob_e01_e02_baseline
        if prob_e01_e02_baseline > 1e-9
        else float("inf")
    )
    ratio_increased = (
        prob_e01_e03_increased / prob_e01_e02_increased
        if prob_e01_e02_increased > 1e-9
        else float("inf")
    )

    msg_ratio = (
        f"Ratio P(E01->E03)/P(E01->E02) with papabile factor {papabile_factor_increased} ({ratio_increased:.4f}) "
        f"not greater than with factor {papabile_factor_baseline} ({ratio_baseline:.4f})"
    )
    assert ratio_increased > ratio_baseline, msg_ratio


def test_papabile_factor_multiplicative_effect():
    """Tests that papabile_weight_factor is applied multiplicatively and can change preference order."""
    elector_data = pd.DataFrame(
        {
            "elector_id": ["E01", "C1", "C2"], # E01 is elector, C1/C2 are candidates
            "ideology_score": [0.0, 0.5, 0.1], # E01 at 0.0, C1 at 0.5, C2 at 0.1
            "region": ["Europe", "Europe", "Europe"], # Same region to neutralize regional effect for this test if bonus is non-zero
            "is_papabile": [False, True, False], # C1 is papabile, C2 is not
        }
    ).set_index("elector_id")

    # Model with beta=1, papabile_weight_factor=1.5 (default), other factors off or neutral
    model = TransitionModel(
        elector_data=elector_data,
        beta_weight=1.0,
        papabile_weight_factor=1.5, # Default, but explicit for clarity
        regional_affinity_bonus=0.0, # Neutralize regional effect
        bandwagon_strength=0.0,      # Neutralize bandwagon effect
        stickiness_factor=0.0        # Neutralize stickiness effect (also no current_votes)
    )

    # Calculate probabilities for Elector E01 voting for C1 or C2
    # We need to consider E01 as the only elector, and C1, C2 as the only candidates.
    # The model internally handles all individuals in elector_data as potential candidates.
    # We'll check the probabilities from E01's perspective.
    
    # Base ideological scores for E01 (ideology 0.0) vs C1 (ideology 0.5) and C2 (ideology 0.1):
    # Pref E01->C1_base = exp(-1.0 * |0.0 - 0.5|) = exp(-0.5) approx 0.6065
    # Pref E01->C2_base = exp(-1.0 * |0.0 - 0.1|) = exp(-0.1) approx 0.9048
    # Initially, E01 prefers C2 over C1 based on ideology.

    # Apply papabile_weight_factor (1.5) to C1 (papabile):
    # Pref E01->C1_adj = Pref E01->C1_base * 1.5 = 0.6065 * 1.5 approx 0.9098
    # Pref E01->C2_adj = Pref E01->C2_base (no change) approx 0.9048
    # Now, E01 prefers C1 over C2.

    prob_matrix, effective_candidates = model.calculate_transition_probabilities(
        elector_data_runtime=elector_data, # Provide runtime data
        active_candidates=["C1", "C2"] # Focus on C1 and C2
    )

    # Elector E01 is the first row in the original elector_data, so it's index 0 for probabilities
    # if all original electors are used. However, the model re-indexes internally for electors if needed.
    # For this setup, elector_data only has one *actual* elector if we filter it.
    # Let's assume the model's self.elector_data_full uses the order from input.
    # E01 is at index 0 in self.elector_data_full.
    # C1 is index 0, C2 is index 1 in effective_candidates list (due to active_candidates order)
    
    assert "C1" in effective_candidates
    assert "C2" in effective_candidates
    c1_idx_effective = effective_candidates.index("C1")
    c2_idx_effective = effective_candidates.index("C2")
    
    # Get probabilities for E01 (first elector in the input `elector_data`)
    e01_prob_for_c1 = prob_matrix[0, c1_idx_effective]
    e01_prob_for_c2 = prob_matrix[0, c2_idx_effective]

    # Expected scores (unnormalized):
    pref_e01_c1_base = np.exp(-1.0 * abs(0.0 - 0.5))
    pref_e01_c2_base = np.exp(-1.0 * abs(0.0 - 0.1))
    
    expected_score_c1 = pref_e01_c1_base * 1.5 # C1 is papabile
    expected_score_c2 = pref_e01_c2_base       # C2 is not

    # Check that C1 is now preferred over C2 due to papabile factor
    assert expected_score_c1 > expected_score_c2, \
        f"C1 score ({expected_score_c1:.4f}) should be > C2 score ({expected_score_c2:.4f})"

    # Check probabilities reflect this preference
    assert e01_prob_for_c1 > e01_prob_for_c2, \
        f"P(E01->C1) ({e01_prob_for_c1:.4f}) should be > P(E01->C2) ({e01_prob_for_c2:.4f})"

    # Verify the ratio of probabilities matches the ratio of calculated scores
    expected_total_score = expected_score_c1 + expected_score_c2
    expected_prob_c1 = expected_score_c1 / expected_total_score
    expected_prob_c2 = expected_score_c2 / expected_total_score

    assert np.isclose(e01_prob_for_c1, expected_prob_c1), \
        f"P(E01->C1) is {e01_prob_for_c1:.4f}, expected {expected_prob_c1:.4f}"
    assert np.isclose(e01_prob_for_c2, expected_prob_c2), \
        f"P(E01->C2) is {e01_prob_for_c2:.4f}, expected {expected_prob_c2:.4f}"


def test_regional_affinity_bonus_effect():
    """Tests that regional_affinity_bonus is applied additively and can change preference."""
    elector_data = pd.DataFrame(
        {
            "elector_id": ["E01", "C1", "C2"], # E01 elector; C1, C2 candidates
            "ideology_score": [0.0, 0.2, 0.1],  # E01 at 0.0; C1 at 0.2; C2 at 0.1 (C2 closer to E01)
            "region": ["RegionA", "RegionA", "RegionB"], # E01, C1 in RegionA; C2 in RegionB
            "is_papabile": [False, False, False], # Neutralize papabile effect for this test
        }
    ).set_index("elector_id")

    regional_bonus_value = 0.1
    beta = 1.0

    model = TransitionModel(
        elector_data=elector_data,
        beta_weight=beta,
        papabile_weight_factor=1.0,     # Neutralize papabile (multiplicative factor of 1)
        regional_affinity_bonus=regional_bonus_value,
        bandwagon_strength=0.0,         # Neutralize bandwagon
        stickiness_factor=0.0           # Neutralize stickiness
    )

    # Base ideological scores for E01 (ideology 0.0):
    # Pref E01->C1_base = exp(-beta * |0.0 - 0.2|) = exp(-0.2) approx 0.8187
    # Pref E01->C2_base = exp(-beta * |0.0 - 0.1|) = exp(-0.1) approx 0.9048
    # Initially, E01 prefers C2 over C1 based on ideology.

    # Apply regional_affinity_bonus (0.1) to C1 (same region as E01):
    # Score E01->C1_adj = Pref E01->C1_base + regional_bonus_value = 0.8187 + 0.1 = 0.9187
    # Score E01->C2_adj = Pref E01->C2_base (no change) = 0.9048
    # Now, E01 should prefer C1 over C2.

    prob_matrix, effective_candidates = model.calculate_transition_probabilities(
        elector_data_runtime=elector_data,
        active_candidates=["C1", "C2"]
    )

    c1_idx_effective = effective_candidates.index("C1")
    c2_idx_effective = effective_candidates.index("C2")

    e01_prob_for_c1 = prob_matrix[0, c1_idx_effective] # E01 is the first elector
    e01_prob_for_c2 = prob_matrix[0, c2_idx_effective]

    # Calculate expected scores (unnormalized)
    pref_e01_c1_base = np.exp(-beta * abs(0.0 - 0.2))
    pref_e01_c2_base = np.exp(-beta * abs(0.0 - 0.1))

    expected_score_c1 = pref_e01_c1_base + regional_bonus_value # C1 gets regional bonus
    expected_score_c2 = pref_e01_c2_base                        # C2 does not

    # Ensure scores are positive before normalization, as TransitionModel might clip at 0 implicitly
    # (though with exp and positive additive bonus, this is unlikely here)
    # This check is more relevant if bonuses could be negative or base scores near zero.
    # For this specific test case, raw scores `expected_score_c1` and `expected_score_c2` will be positive.

    assert expected_score_c1 > expected_score_c2, \
        f"C1 score ({expected_score_c1:.4f}) should be > C2 score ({expected_score_c2:.4f}) after regional bonus."

    assert e01_prob_for_c1 > e01_prob_for_c2, \
        f"P(E01->C1) ({e01_prob_for_c1:.4f}) should be > P(E01->C2) ({e01_prob_for_c2:.4f})"

    # Verify the ratio of probabilities matches the ratio of calculated scores
    expected_total_score = expected_score_c1 + expected_score_c2
    expected_prob_c1 = expected_score_c1 / expected_total_score
    expected_prob_c2 = expected_score_c2 / expected_total_score

    assert np.isclose(e01_prob_for_c1, expected_prob_c1), \
        f"P(E01->C1) is {e01_prob_for_c1:.4f}, expected {expected_prob_c1:.4f}"
    assert np.isclose(e01_prob_for_c2, expected_prob_c2), \
        f"P(E01->C2) is {e01_prob_for_c2:.4f}, expected {expected_prob_c2:.4f}"


def test_bandwagon_effect():
    """Tests that bandwagon_strength is applied additively based on previous vote counts."""
    # E01, E02, E03 are electors. C1, C2 are candidates.
    # We're interested in E01's choice between C1 and C2.
    elector_data = pd.DataFrame(
        {
            "elector_id": ["E01", "E02", "E03", "C1", "C2"],
            "ideology_score": [0.0, 0.0, 0.0, 0.1, 0.1], # C1 and C2 equally (ideologically) close to E01
            "region": ["R1", "R1", "R1", "R1", "R1"],   # All same region
            "is_papabile": [False, False, False, False, False], # No papabile effect
        }
    ).set_index("elector_id")

    bandwagon_val = 0.2
    beta = 1.0

    model = TransitionModel(
        elector_data=elector_data,
        beta_weight=beta,
        papabile_weight_factor=1.0,
        regional_affinity_bonus=0.0,
        bandwagon_strength=bandwagon_val,
        stickiness_factor=0.0 # No stickiness for this test
    )

    # Simulate previous votes: C1 gets 2 votes (from E02, E03), C2 gets 0 votes.
    # E01's previous vote doesn't matter for *itself* in bandwagon, only for stickiness (which is off).
    current_votes_for_bandwagon = {
        "E02": "C1",
        "E03": "C1",
        # E01 could have voted for anyone or abstained, stickiness is off.
    }

    # Base ideological scores for E01 (ideology 0.0) vs C1 (0.1) and C2 (0.1):
    # Pref E01->C1_base = exp(-1.0 * |0.0 - 0.1|) = exp(-0.1) approx 0.9048
    # Pref E01->C2_base = exp(-1.0 * |0.0 - 0.1|) = exp(-0.1) approx 0.9048
    # Papabile and Regional are off, so these are the scores before bandwagon.

    # Bandwagon calculation:
    # Total votes considered for active candidates C1, C2: 2 (for C1) + 0 (for C2) = 2
    # Vote share C1 = 2/2 = 1.0
    # Vote share C2 = 0/2 = 0.0
    # Bandwagon bonus for C1 = bandwagon_val * 1.0 = 0.2
    # Bandwagon bonus for C2 = bandwagon_val * 0.0 = 0.0

    # Expected scores after bandwagon for E01:
    # Score E01->C1_adj = Pref E01->C1_base + Bandwagon_C1 = 0.9048 + 0.2 = 1.1048
    # Score E01->C2_adj = Pref E01->C2_base + Bandwagon_C2 = 0.9048 + 0.0 = 0.9048
    # E01 should now prefer C1 over C2.

    prob_matrix, effective_candidates = model.calculate_transition_probabilities(
        elector_data_runtime=elector_data,
        current_votes=current_votes_for_bandwagon,
        active_candidates=["C1", "C2"]
    )

    e01_idx_in_model = model.candidate_names.index("E01")
    c1_idx_effective = effective_candidates.index("C1")
    c2_idx_effective = effective_candidates.index("C2")

    e01_prob_for_c1 = prob_matrix[e01_idx_in_model, c1_idx_effective]
    e01_prob_for_c2 = prob_matrix[e01_idx_in_model, c2_idx_effective]

    pref_e01_c_base = np.exp(-beta * abs(0.0 - 0.1)) # Same base for C1 and C2

    expected_bandwagon_score_c1 = bandwagon_val * 1.0 # C1 got all relevant votes
    expected_bandwagon_score_c2 = bandwagon_val * 0.0 # C2 got no relevant votes

    expected_score_c1 = pref_e01_c_base + expected_bandwagon_score_c1
    expected_score_c2 = pref_e01_c_base + expected_bandwagon_score_c2

    assert expected_score_c1 > expected_score_c2, \
        f"C1 score ({expected_score_c1:.4f}) should be > C2 score ({expected_score_c2:.4f}) due to bandwagon."
    
    assert e01_prob_for_c1 > e01_prob_for_c2, \
        f"P(E01->C1) ({e01_prob_for_c1:.4f}) should be > P(E01->C2) ({e01_prob_for_c2:.4f})"

    expected_total_score = expected_score_c1 + expected_score_c2
    expected_prob_c1 = expected_score_c1 / expected_total_score
    expected_prob_c2 = expected_score_c2 / expected_total_score

    assert np.isclose(e01_prob_for_c1, expected_prob_c1), \
        f"P(E01->C1) is {e01_prob_for_c1:.4f}, expected {expected_prob_c1:.4f}"
    assert np.isclose(e01_prob_for_c2, expected_prob_c2), \
        f"P(E01->C2) is {e01_prob_for_c2:.4f}, expected {expected_prob_c2:.4f}"


def test_stickiness_factor_effect():
    """Tests that stickiness_factor is applied multiplicatively to a voter's previous choice."""
    # E01 is the elector. C1, C2 are candidates.
    elector_data = pd.DataFrame(
        {
            "elector_id": ["E01", "C1", "C2"],
            "ideology_score": [0.0, 0.2, 0.1],  # E01 at 0.0; C1 at 0.2; C2 at 0.1 (C2 closer to E01 initially)
            "region": ["R1", "R1", "R1"],       # All same region to neutralize regional effect
            "is_papabile": [False, False, False], # No papabile effect
        }
    ).set_index("elector_id")

    stickiness_val = 0.15 # Enough to overcome C2's initial ideological advantage
    beta = 1.0

    model = TransitionModel(
        elector_data=elector_data,
        beta_weight=beta,
        papabile_weight_factor=1.0,     # Neutralize papabile
        regional_affinity_bonus=0.0,    # Neutralize regional bonus
        bandwagon_strength=0.0,         # Neutralize bandwagon
        stickiness_factor=stickiness_val
    )

    # E01 voted for C1 in the previous round.
    current_votes_for_stickiness = {
        "E01": "C1"
    }

    # Base ideological scores for E01 (ideology 0.0):
    # Pref E01->C1_base = exp(-beta * |0.0 - 0.2|) = exp(-0.2) approx 0.8187
    # Pref E01->C2_base = exp(-beta * |0.0 - 0.1|) = exp(-0.1) approx 0.9048
    # Initially, E01 prefers C2 over C1 (0.9048 > 0.8187).

    # Stickiness calculation for E01:
    # Stickiness bonus for C1 (previous vote) = stickiness_val = 0.15
    # Stickiness bonus for C2 (not previous vote) = 0.0

    # Expected scores after stickiness for E01:
    # Score E01->C1_adj = Pref E01->C1_base * (1 + stickiness_val) = 0.8187 * (1 + 0.15) = 0.9406
    # Score E01->C2_adj = Pref E01->C2_base * (1 + 0.0) = 0.9048 * 1 = 0.9048
    # E01 should now prefer C1 over C2.

    prob_matrix, effective_candidates = model.calculate_transition_probabilities(
        elector_data_runtime=elector_data,
        current_votes=current_votes_for_stickiness,
        active_candidates=["C1", "C2"]
    )

    # E01 is the first elector in the original elector_data, so its index is 0 in prob_matrix
    e01_idx_in_model = model.candidate_names.index("E01")
    c1_idx_effective = effective_candidates.index("C1")
    c2_idx_effective = effective_candidates.index("C2")

    e01_prob_for_c1 = prob_matrix[e01_idx_in_model, c1_idx_effective]
    e01_prob_for_c2 = prob_matrix[e01_idx_in_model, c2_idx_effective]

    pref_e01_c1_base = np.exp(-beta * abs(0.0 - 0.2))
    pref_e01_c2_base = np.exp(-beta * abs(0.0 - 0.1))

    expected_stickiness_score_c1 = pref_e01_c1_base * (1 + stickiness_val) # E01 voted for C1
    expected_stickiness_score_c2 = pref_e01_c2_base                        # E01 did not vote for C2, no stickiness bonus/malus

    expected_score_c1 = expected_stickiness_score_c1
    expected_score_c2 = expected_stickiness_score_c2

    assert expected_score_c1 > expected_score_c2, \
        f"C1 score ({expected_score_c1:.4f}) should be > C2 score ({expected_score_c2:.4f}) due to stickiness."

    assert e01_prob_for_c1 > e01_prob_for_c2, \
        f"P(E01->C1) ({e01_prob_for_c1:.4f}) should be > P(E01->C2) ({e01_prob_for_c2:.4f})"

    expected_total_score = expected_score_c1 + expected_score_c2
    expected_prob_c1 = expected_score_c1 / expected_total_score
    expected_prob_c2 = expected_score_c2 / expected_total_score

    assert np.isclose(e01_prob_for_c1, expected_prob_c1), \
        f"P(E01->C1) is {e01_prob_for_c1:.4f}, expected {expected_prob_c1:.4f}"
    assert np.isclose(e01_prob_for_c2, expected_prob_c2), \
        f"P(E01->C2) is {e01_prob_for_c2:.4f}, expected {expected_prob_c2:.4f}"


# TODO: Add test for the combined order of operations

def test_combined_effects_order_of_operations():
    """Tests the combined effect and correct order of operations for all factors."""
    elector_data = pd.DataFrame(
        {
            "elector_id": ["E01", "E02", "E03", "C1", "C2"],
            "ideology_score": [0.0, 0.5, -0.5, 0.2, -0.1],
            "region": ["North", "North", "South", "North", "South"],
            "is_papabile": [False, False, False, True, False], # C1 is papabile
        }
    ).set_index("elector_id")

    beta = 1.0
    papabile_factor = 1.5
    regional_bonus = 0.1
    bandwagon_strength_val = 0.2
    stickiness_val = 0.1

    model = TransitionModel(
        elector_data=elector_data,
        beta_weight=beta,
        papabile_weight_factor=papabile_factor,
        regional_affinity_bonus=regional_bonus,
        bandwagon_strength=bandwagon_strength_val,
        stickiness_factor=stickiness_val
    )

    current_votes_dict = {
        "E01": "C1", # For E01's stickiness to C1
        "E02": "C2", # Contributes to C2's bandwagon
        "E03": "C2", # Contributes to C2's bandwagon
    }
    # For bandwagon calculation active_candidates C1, C2:
    # C1 gets 1 vote (from E01), C2 gets 2 votes (from E02, E03)

    # --- Manual Calculation for E01's preference scores for C1 and C2 ---
    # E01: ideology 0.0, region "North"
    # C1: ideology 0.2, region "North", papabile True
    # C2: ideology -0.1, region "South", papabile False

    # Step 1: Ideology
    score_c1_ideology = np.exp(-beta * abs(0.0 - 0.2)) # exp(-0.2) approx 0.81873
    score_c2_ideology = np.exp(-beta * abs(0.0 - (-0.1))) # exp(-0.1) approx 0.90484

    # Step 2: Papabile (C1 is papabile)
    score_c1_papabile = score_c1_ideology * papabile_factor
    score_c2_papabile = score_c2_ideology

    # Step 3: Regional (E01 & C1 in "North", C2 in "South")
    score_c1_regional = score_c1_papabile + regional_bonus # E01 and C1 same region
    score_c2_regional = score_c2_papabile # E01 and C2 different region

    # Step 4: Bandwagon
    # Votes for C1 among active = 1 (E01->C1)
    # Votes for C2 among active = 2 (E02->C2, E03->C2)
    # Total votes for C1,C2 = 3
    bandwagon_share_c1 = 1/3
    bandwagon_share_c2 = 2/3
    bonus_c1_bandwagon = bandwagon_strength_val * bandwagon_share_c1
    bonus_c2_bandwagon = bandwagon_strength_val * bandwagon_share_c2
    score_c1_bandwagon = score_c1_regional + bonus_c1_bandwagon
    score_c2_bandwagon = score_c2_regional + bonus_c2_bandwagon

    # Step 5: Stickiness (E01 voted for C1)
    expected_score_c1 = score_c1_bandwagon * (1 + stickiness_val)
    expected_score_c2 = score_c2_bandwagon

    # --- Get model's probabilities ---
    prob_matrix, effective_candidates = model.calculate_transition_probabilities(
        elector_data_runtime=elector_data, 
        current_votes=current_votes_dict,
        active_candidates=["C1", "C2"]
    )

    e01_idx_in_model = model.candidate_names.index("E01")
    c1_idx_effective = effective_candidates.index("C1")
    c2_idx_effective = effective_candidates.index("C2")

    model_prob_e01_c1 = prob_matrix[e01_idx_in_model, c1_idx_effective]
    model_prob_e01_c2 = prob_matrix[e01_idx_in_model, c2_idx_effective]

    # --- Compare with expected probabilities ---
    total_expected_score = expected_score_c1 + expected_score_c2
    expected_prob_c1 = expected_score_c1 / total_expected_score
    expected_prob_c2 = expected_score_c2 / total_expected_score

    assert np.isclose(model_prob_e01_c1, expected_prob_c1), \
        f"P(E01->C1) model: {model_prob_e01_c1:.5f}, expected: {expected_prob_c1:.5f}"
    assert np.isclose(model_prob_e01_c2, expected_prob_c2), \
        f"P(E01->C2) model: {model_prob_e01_c2:.5f}, expected: {expected_prob_c2:.5f}"
