import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union


def calculate_preference_probabilities(
    elector_data: pd.DataFrame,
    beta_weight: float
) -> np.ndarray:
    """Calculates initial vote preference probabilities based on ideology.

    Probability P(i -> j) is proportional to exp(-beta * |score_i - score_j|).
    Higher beta means stronger preference for ideologically similar candidates.

    Args:
        elector_data: DataFrame with 'elector_id' as index and 'ideology_score'.
        beta_weight: Sensitivity parameter for ideological difference.

    Returns:
        An N x N numpy array where N is the number of electors. Matrix[i, j]
        is the probability of elector i voting for elector j.
        The diagonal is always zero.

    Raises:
        ValueError: If input data is invalid or normalization fails.
    """
    if not isinstance(elector_data, pd.DataFrame) or elector_data.empty:
        raise ValueError("elector_data must be a non-empty DataFrame.")
    if 'ideology_score' not in elector_data.columns:
        raise ValueError("elector_data must contain 'ideology_score' column.")
    if elector_data.index.name != 'elector_id':
        # Check if 'elector_id' column exists to provide a better error message
        if 'elector_id' in elector_data.columns:
             raise ValueError("elector_data must have 'elector_id' set as the index.")
        else:
             raise ValueError("elector_data must have an 'elector_id' index or column.")
    if not isinstance(beta_weight, (int, float)) or beta_weight < 0:
        raise ValueError("beta_weight must be a non-negative number.")

    n_electors = len(elector_data)
    if n_electors == 0:
        # Return empty array if no electors (though caught by .empty() check)
        return np.array([]).reshape(0, 0)

    scores = elector_data['ideology_score'].values
    elector_ids = elector_data.index.values

    # Calculate pairwise absolute differences in ideology scores
    # diffs[i, j] = |score_i - score_j|
    ideology_diffs = np.abs(scores[:, None] - scores[None, :])

    # Calculate preference weights: exp(-beta * |difference|)
    # High score = high preference (low difference)
    pref_weights = np.exp(-beta_weight * ideology_diffs)

    # --- Normalization --- Ensure probabilities sum to 1 for each voter

    # Set diagonal to 0 (cannot vote for self)
    np.fill_diagonal(pref_weights, 0)

    # Calculate row sums
    row_sums = pref_weights.sum(axis=1, keepdims=True)

    # Handle rows where the sum is zero (e.g., single elector, or all others have infinite distance)
    # If sum is 0, division results in NaN. Replace 0 sums with 1 to avoid NaN,
    # the resulting row will be all zeros, which is correct (no valid candidates).
    # Or, if N=1, row_sum is 0, results in [0].
    # If N > 1 and row_sum is 0, means all other candidates were infinitely distant?
    # This case might need uniform probability assignment like in TransitionModel.

    prob_matrix = np.zeros_like(pref_weights)
    non_zero_sum_rows = (row_sums > 1e-9).flatten() # Avoid floating point issues

    if np.any(non_zero_sum_rows):
        safe_sums = np.where(row_sums[non_zero_sum_rows] == 0, 1, row_sums[non_zero_sum_rows])
        prob_matrix[non_zero_sum_rows, :] = pref_weights[non_zero_sum_rows, :] / safe_sums

    # Handle rows that had zero sums (assign uniform probability if N > 1)
    zero_sum_rows = ~non_zero_sum_rows
    if np.any(zero_sum_rows) and n_electors > 1:
         print(f"Warning: Zero preference sum for electors (indices: {np.where(zero_sum_rows)[0]}). Assigning uniform vote.")
         uniform_prob = 1.0 / (n_electors - 1)
         prob_matrix[zero_sum_rows, :] = uniform_prob
         np.fill_diagonal(prob_matrix, 0) # Ensure diagonal is still zero

    # --- Final Validation ---
    final_sums = prob_matrix.sum(axis=1)
    valid_sums = np.isclose(final_sums, 1.0, rtol=1e-5, atol=1e-7) | np.isclose(final_sums, 0.0, atol=1e-7)
    if not np.all(valid_sums):
        invalid_rows = np.where(~valid_sums)[0]
        problematic_elector_ids = elector_data.iloc[invalid_rows].index.tolist()
        print(f"Error: Preference probability matrix normalization failed for electors: {problematic_elector_ids}")
        print(f"Row indices: {invalid_rows}")
        print(f"Problematic sums: {final_sums[invalid_rows]}")
        raise ValueError("Preference probability matrix rows do not sum correctly to 1.0 or 0.0 after normalization.")

    return prob_matrix


class TransitionModel:
    """Models the transition probabilities of elector votes between rounds.

    Attributes:
        parameters: A dictionary containing model parameters.
            Expected keys:
            - 'beta_weight': Controls influence of ideological distance.
            - 'stickiness_factor': Controls tendency to repeat previous vote (0-1).
    """

    def __init__(self, parameters: Dict[str, Any]):
        """Initializes the TransitionModel.

        Args:
            parameters: Dictionary of model parameters.

        Raises:
            ValueError: If required parameters are missing or invalid.
        """
        self.parameters = parameters
        print(f"TransitionModel initialized with parameters: {self.parameters}")

        # --- Parameter Validation ---
        required_params = ['beta_weight', 'stickiness_factor']
        for param in required_params:
            if param not in self.parameters:
                raise ValueError(f"Model parameters must include '{param}'.")

        beta = self.parameters['beta_weight']
        stickiness = self.parameters['stickiness_factor']

        if not isinstance(beta, (int, float)) or beta < 0:
            raise ValueError(f"'beta_weight' must be a non-negative number, got {beta}")
        if not isinstance(stickiness, (int, float)) or not (0 <= stickiness <= 1):
            raise ValueError(f"'stickiness_factor' must be between 0 and 1, got {stickiness}")

        # TODO: Add more parameter validation as model complexity increases

    def calculate_transition_probabilities(
        self,
        elector_data: pd.DataFrame,
        current_votes: Optional[Dict[Any, Any]] = None
    ) -> np.ndarray:
        """Calculates the probability of each elector voting for each candidate.

        Combines ideological distance with vote stickiness from the previous round.

        Args:
            elector_data: DataFrame with elector information, including an
                          'ideology_score' column and elector IDs as index.
            current_votes: Dictionary mapping elector IDs (from index) to their
                           vote (candidate ID, also from index) in the previous round.
                           If None (e.g., first round), only ideology is used.

        Returns:
            np.ndarray: An N x N matrix where P[i, j] is the probability of
                        elector i voting for elector j. Rows sum to 1.
                        The diagonal is zero (electors cannot vote for themselves).

        Raises:
            ValueError: If 'ideology_score' column is missing.
            KeyError: If IDs in current_votes don't match elector_data index.
        """
        # AI: Implemented transition probability calculation
        if 'ideology_score' not in elector_data.columns:
            raise ValueError("Elector data must include 'ideology_score' column.")

        n_electors = len(elector_data)
        # AI: Add explicit check for empty input DataFrame
        if n_electors == 0:
            raise ValueError("Cannot calculate probabilities with zero electors")

        elector_ids = elector_data.index.to_numpy() # Get IDs in DataFrame order
        ideology_scores = elector_data['ideology_score'].to_numpy()

        beta = self.parameters['beta_weight']
        stickiness = self.parameters['stickiness_factor']

        # 1. Calculate Ideological Attraction Matrix
        # Reshape for broadcasting: (N, 1) vs (1, N) -> (N, N) difference matrix
        ideology_diff = np.abs(ideology_scores[:, np.newaxis] - ideology_scores)
        attraction_matrix = np.exp(-beta * ideology_diff)
        # Initialize transition probability matrix
        prob_matrix = attraction_matrix.copy()

        # 2. Incorporate Stickiness (if applicable)
        if current_votes is not None:
            if not set(current_votes.keys()).issubset(set(elector_ids)):
                 missing_voters = set(current_votes.keys()) - set(elector_ids)
                 raise KeyError(f"Voter IDs in current_votes not found in elector_data index: {missing_voters}")
            if not set(current_votes.values()).issubset(set(elector_ids)):
                 missing_candidates = set(current_votes.values()) - set(elector_ids)
                 raise KeyError(f"Candidate IDs (votes) in current_votes not found in elector_data index: {missing_candidates}")

            # Create stickiness matrix (N x N)
            # stickiness_matrix[i, j] = 1 if elector i voted for j previously, else 0
            stickiness_matrix = np.zeros((n_electors, n_electors))
            # Map elector IDs back to their integer index position for array indexing
            id_to_index = {id_val: idx for idx, id_val in enumerate(elector_ids)}

            for voter_id, candidate_id in current_votes.items():
                 voter_idx = id_to_index[voter_id]
                 candidate_idx = id_to_index[candidate_id]
                 stickiness_matrix[voter_idx, candidate_idx] = 1.0 # Elector i previously voted for j

            # Combine attraction and stickiness
            prob_matrix = (1 - stickiness) * attraction_matrix + stickiness * stickiness_matrix

        # 3. Normalize probabilities (rows sum to 1 initially)
        # Avoid division by zero if a row is all zeros (though unlikely with exp)
        row_sums = prob_matrix.sum(axis=1, keepdims=True)
        # Replace zeros in row_sums with 1 to avoid division by zero; probabilities will be 0 anyway.
        safe_row_sums = np.where(row_sums == 0, 1, row_sums)
        prob_matrix /= safe_row_sums

        # 4. Set self-vote probability to zero
        np.fill_diagonal(prob_matrix, 0)

        # 5. Re-normalize rows after removing self-vote probability
        row_sums_after_diag = prob_matrix.sum(axis=1, keepdims=True)
        # Handle electors who only had non-zero probability for themselves (now all zero)
        # Assign uniform probability to others in this rare case.
        zero_sum_rows = (row_sums_after_diag == 0).flatten()
        if np.any(zero_sum_rows):
            print(f"Warning: Found {sum(zero_sum_rows)} electors with zero probability "
                  "to vote for others after removing self-vote. Assigning uniform probability.")
            # Assign uniform probability (1 / (N-1)) to others for these rows
            uniform_prob = 1.0 / (n_electors - 1) if n_electors > 1 else 1.0
            prob_matrix[zero_sum_rows, :] = uniform_prob
            # Ensure diagonal is still zero
            np.fill_diagonal(prob_matrix, 0)
            # Re-calculate sums for these specific rows before final normalization
            row_sums_after_diag[zero_sum_rows] = prob_matrix[zero_sum_rows].sum(axis=1, keepdims=True)

        # Avoid division by zero for rows that became all zero
        safe_row_sums_after_diag = np.where(row_sums_after_diag == 0, 1, row_sums_after_diag)
        prob_matrix /= safe_row_sums_after_diag

        # Final check for row sums (should be very close to 1 or 0)
        final_sums = prob_matrix.sum(axis=1)
        # Check if all sums are close to 1.0 OR 0.0 (for rows that might be all zero if n_electors=1 or all probabilities became zero)
        # Increased tolerance slightly for floating point comparisons with many electors
        valid_sums = np.isclose(final_sums, 1.0, rtol=1e-5, atol=1e-7) | np.isclose(final_sums, 0.0, atol=1e-7)
        if not np.all(valid_sums):
            invalid_rows = np.where(~valid_sums)[0]
            # Use .iloc to get elector_ids based on integer position
            problematic_elector_ids = elector_data.iloc[invalid_rows].index.tolist()
            print(f"Error: Probability matrix normalization failed for electors: {problematic_elector_ids}")
            print(f"Row indices: {invalid_rows}")
            print(f"Problematic sums: {final_sums[invalid_rows]}")
            # Optional: Could print the offending rows of prob_matrix for debugging
            # print(f"Problematic rows:\n{prob_matrix[invalid_rows, :]}")
            raise ValueError("Probability matrix rows do not sum correctly to 1.0 or 0.0 after normalization.")

        return prob_matrix
