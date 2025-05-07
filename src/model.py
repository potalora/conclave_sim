import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, List, Tuple
import logging
log = logging.getLogger(__name__)

def calculate_preference_probabilities(
    elector_data: pd.DataFrame,
    beta_weight: float,
    regional_affinity_bonus: float = 0.1,
    papabile_candidate_bonus: float = 0.1
) -> np.ndarray:
    """Calculates initial vote preference probabilities.

    Combines ideology, regional affinity, and papabile status of candidates.
    Probability P(i -> j) is based on:
    exp(-beta * |score_i - score_j|) + regional_bonus + papabile_bonus.
    Higher beta means stronger preference for ideologically similar candidates.

    Args:
        elector_data: DataFrame with 'elector_id' as index and columns
                      'ideology_score', 'region', and 'is_papabile'.
        beta_weight: Sensitivity parameter for ideological difference.
        regional_affinity_bonus: Additive bonus if elector and candidate share a region.
        papabile_candidate_bonus: Additive bonus if the candidate is papabile.

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
    if 'region' not in elector_data.columns:
        raise ValueError("elector_data must contain 'region' column.")
    if 'is_papabile' not in elector_data.columns:
        raise ValueError("elector_data must contain 'is_papabile' column.")
    if elector_data.index.name != 'elector_id':
        # Check if 'elector_id' column exists to provide a better error message
        if 'elector_id' in elector_data.columns:
             raise ValueError("elector_data must have 'elector_id' set as the index.")
        else:
             raise ValueError("elector_data must have an 'elector_id' index or column.")
    if not isinstance(beta_weight, (int, float)) or beta_weight < 0:
        raise ValueError("beta_weight must be a non-negative number.")
    if not isinstance(regional_affinity_bonus, (int, float)) or regional_affinity_bonus < 0:
        raise ValueError("regional_affinity_bonus must be a non-negative number.")
    if not isinstance(papabile_candidate_bonus, (int, float)) or papabile_candidate_bonus < 0:
        raise ValueError("papabile_candidate_bonus must be a non-negative number.")

    n_electors = len(elector_data)
    if n_electors == 0:
        # Return empty array if no electors (though caught by .empty() check)
        return np.array([]).reshape(0, 0)

    scores = elector_data['ideology_score'].values
    elector_ids = elector_data.index.values

    # Calculate pairwise absolute differences in ideology scores
    # diffs[i, j] = |score_i - score_j|
    ideology_diffs = np.abs(scores[:, None] - scores[None, :])

    # Calculate base preference weights: exp(-beta * |difference|)
    pref_weights = np.exp(-beta_weight * ideology_diffs)

    # Apply regional affinity bonus
    regions = elector_data['region'].values
    # Create a matrix where same_region_matrix[i, j] is True if elector i and candidate j are from the same region
    # Candidate j is also an elector, so we compare regions[i] with regions[j]
    same_region_matrix = (regions[:, None] == regions[None, :])
    pref_weights[same_region_matrix] += regional_affinity_bonus
    log.debug(f"Applied regional affinity bonus. Affected elements: {same_region_matrix.sum()}")

    # Apply papabile candidate bonus
    # This bonus applies if the *candidate* (column j) is papabile.
    is_papabile_status = elector_data['is_papabile'].values.astype(bool)
    # Add bonus to columns corresponding to papabile candidates
    # pref_weights[:, is_papabile_status] should add the bonus to all rows for columns where is_papabile_status is True
    for j in range(n_electors):
        if is_papabile_status[j]:
            pref_weights[:, j] += papabile_candidate_bonus
    log.debug(f"Applied papabile candidate bonus. Papabile candidates count: {is_papabile_status.sum()}")

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
         log.warning(f"Warning: Zero preference sum for electors (indices: {np.where(zero_sum_rows)[0]}). Assigning uniform vote.")
         uniform_prob = 1.0 / (n_electors - 1)
         prob_matrix[zero_sum_rows, :] = uniform_prob
         np.fill_diagonal(prob_matrix, 0) # Ensure diagonal is still zero

    # --- Final Validation ---
    final_sums = prob_matrix.sum(axis=1)
    valid_sums = np.isclose(final_sums, 1.0, rtol=1e-5, atol=1e-7) | np.isclose(final_sums, 0.0, atol=1e-7)
    if not np.all(valid_sums):
        invalid_rows = np.where(~valid_sums)[0]
        problematic_elector_ids = elector_data.iloc[invalid_rows].index.tolist()
        log.error(f"Error: Preference probability matrix normalization failed for electors: {problematic_elector_ids}")
        log.error(f"Row indices: {invalid_rows}")
        log.error(f"Problematic sums: {final_sums[invalid_rows]}")
        raise ValueError("Preference probability matrix rows do not sum correctly to 1.0 or 0.0 after normalization.")

    return prob_matrix


class TransitionModel:
    """Models the transition probabilities of elector votes between rounds.

    Attributes:
        parameters: A dictionary containing model parameters.
            Expected keys:
            - 'beta_weight': Controls influence of ideological distance.
            - 'stickiness_factor': Controls tendency to repeat previous vote (0-1).
            - 'bandwagon_strength': Controls influence of previous round's vote counts (>=0).
            - 'regional_affinity_bonus': Additive bonus for same region.
            - 'papabile_candidate_bonus': Additive bonus for papabile candidate.
    """

    def __init__(self, elector_data: pd.DataFrame, beta_weight: float = 1.0, 
                 stickiness_factor: float = 0.5, bandwagon_strength: float = 0.0,
                 regional_affinity_bonus: float = 0.1, papabile_candidate_bonus: float = 0.1):
        """Initialize the TransitionModel.

        Args:
            elector_data: DataFrame with elector profiles, must include 'elector_id' (as index),
                          'ideology_score', 'region', and 'is_papabile'.
            beta_weight: Sensitivity to ideological differences. Higher values mean stronger preference for closer candidates.
            stickiness_factor: Tendency to repeat the previous vote (0 to 1).
            bandwagon_strength: Strength of the bandwagon effect (>=0).
            regional_affinity_bonus: Additive bonus if elector and candidate share a region.
            papabile_candidate_bonus: Additive bonus if the candidate is papabile.

        Raises:
            ValueError: If required columns are missing or parameters are invalid.
        """
        if not isinstance(elector_data, pd.DataFrame):
            raise ValueError("elector_data must be a pandas DataFrame.")
        required_cols = ['ideology_score', 'region', 'is_papabile']
        for col in required_cols:
            if col not in elector_data.columns:
                raise ValueError(f"Elector data must contain a '{col}' column.")
        if elector_data.index.name != 'elector_id':
            log.warning("Elector data index is not named 'elector_id'. Ensure IDs are properly handled or set as index.")
            # Potentially raise ValueError if 'elector_id' is critical as index here.
            # For now, we assume it's correctly indexed if this check passes or user is warned.

        if not (0 <= stickiness_factor <= 1):
            raise ValueError("stickiness_factor must be between 0 and 1.")
        if beta_weight < 0:
            raise ValueError("beta_weight must be non-negative.")
        if bandwagon_strength < 0:
            raise ValueError("bandwagon_strength must be non-negative.")
        if regional_affinity_bonus < 0:
            raise ValueError("regional_affinity_bonus must be non-negative.")
        if papabile_candidate_bonus < 0:
            raise ValueError("papabile_candidate_bonus must be non-negative.")

        self.elector_data_full = elector_data.copy() # Store full data for region/papabile access
        self.elector_ideologies = self.elector_data_full['ideology_score'].values.astype(float)
        self.candidate_names: List[str] = self.elector_data_full.index.tolist()
        self.candidate_ideologies: np.ndarray = self.elector_ideologies
        self.candidate_regions: np.ndarray = self.elector_data_full['region'].values
        self.candidate_is_papabile: np.ndarray = self.elector_data_full['is_papabile'].values.astype(bool)
        
        self.num_electors = len(self.elector_ideologies)
        self.num_candidates = len(self.candidate_names)

        self.beta_weight = float(beta_weight)
        self.stickiness_factor = float(stickiness_factor)
        self.bandwagon_strength = float(bandwagon_strength)
        self.regional_affinity_bonus = float(regional_affinity_bonus)
        self.papabile_candidate_bonus = float(papabile_candidate_bonus)

        # Precompute base ideological preference scores (electors x all candidates)
        # These will be further adjusted by region and papabile status per call.
        self.base_ideological_pref_scores = np.exp(-self.beta_weight * np.abs(self.elector_ideologies[:, np.newaxis] - self.candidate_ideologies[np.newaxis, :]))
        
        log.debug(f"TransitionModel initialized with {self.num_electors} electors and {self.num_candidates} potential candidates.")
        log.debug(f"Params: beta={self.beta_weight}, stickiness={self.stickiness_factor}, bandwagon={self.bandwagon_strength}, region_bonus={self.regional_affinity_bonus}, papabile_bonus={self.papabile_candidate_bonus}")


    def _get_effective_candidate_data(self, active_candidates: Optional[List[str]] = None) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Helper to get data for effective (active or all) candidates."""
        if active_candidates:
            try:
                active_candidate_indices = [self.candidate_names.index(name) for name in active_candidates]
            except ValueError as e:
                missing_candidate = str(e).split("'", 2)[1]
                log.error(f"Error: Candidate '{missing_candidate}' from active_candidates not found in model's master list.")
                raise ValueError(f"Active candidate '{missing_candidate}' not in master candidate list.") from e
            
            effective_candidate_names = [self.candidate_names[i] for i in active_candidate_indices]
            effective_candidate_ideologies = self.candidate_ideologies[active_candidate_indices]
            effective_candidate_regions = self.candidate_regions[active_candidate_indices]
            effective_candidate_is_papabile = self.candidate_is_papabile[active_candidate_indices]
            # Slice the base ideological preferences for these active candidates
            effective_base_ideological_pref_scores = self.base_ideological_pref_scores[:, active_candidate_indices]
        else:
            effective_candidate_names = self.candidate_names
            effective_candidate_ideologies = self.candidate_ideologies
            effective_candidate_regions = self.candidate_regions
            effective_candidate_is_papabile = self.candidate_is_papabile
            effective_base_ideological_pref_scores = self.base_ideological_pref_scores
        
        return effective_candidate_names, effective_candidate_ideologies, effective_candidate_regions, effective_candidate_is_papabile, effective_base_ideological_pref_scores

    def calculate_transition_probabilities(
        self,
        elector_data_runtime: pd.DataFrame, # Full elector data available at runtime for current state
        current_votes: Optional[Dict[str, Union[str, int]]] = None, # elector_id -> candidate_id
        active_candidates: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """Calculates the probability of each elector voting for each (active) candidate.

        Combines ideological preference, regional affinity, papabile status, vote stickiness, and bandwagon effect.

        Args:
            elector_data_runtime: DataFrame of all electors in the simulation (index 'elector_id', cols 'region', 'is_papabile').
                                  Used to get current region of electors if needed, though self.elector_data_full is primary.
            current_votes: Dictionary mapping elector_id to their chosen candidate_id (str or int) in the previous round.
                           If None (e.g., first round), stickiness and bandwagon are not applied.
            active_candidates: Optional list of candidate_ids (str). If provided, probabilities are calculated
                               only for these candidates. If None, all candidates are considered.

        Returns:
            A tuple (prob_matrix, effective_candidate_names):
            - prob_matrix: A 2D numpy array (num_electors x num_effective_candidates) where each row sums to 1.
            - effective_candidate_names: List of candidate names corresponding to columns in prob_matrix.
        """
        
        effective_candidate_names, _, effective_candidate_regions, \
        effective_candidate_is_papabile, base_pref_scores = self._get_effective_candidate_data(active_candidates)
        
        num_effective_candidates = len(effective_candidate_names)
        if num_effective_candidates == 0:
            log.warning("No effective candidates to calculate transition probabilities for.")
            return np.array([]).reshape(self.num_electors, 0), []

        # Start with base ideological preference scores (electors x effective_candidates)
        current_pref_scores = base_pref_scores.copy()

        # Apply regional affinity bonus
        # Elector regions are from self.elector_data_full.index which maps to self.elector_data_full['region']
        elector_regions = self.elector_data_full['region'].values # Shape: (num_electors,)
        # Compare each elector's region with each *effective* candidate's region
        # same_region_matrix[i, j] is True if elector i and (effective) candidate j have same region
        same_region_matrix = (elector_regions[:, np.newaxis] == effective_candidate_regions[np.newaxis, :])
        current_pref_scores[same_region_matrix] += self.regional_affinity_bonus

        # Apply papabile candidate bonus
        # This bonus applies if the *effective_candidate* is papabile.
        # effective_candidate_is_papabile is a boolean array of shape (num_effective_candidates,)
        current_pref_scores[:, effective_candidate_is_papabile] += self.papabile_candidate_bonus
        
        # --- Stickiness and Bandwagon (if not first round) ---
        if current_votes is not None and self.num_electors > 0:
            # Stickiness: Increase preference for previously chosen candidate
            for elector_idx, elector_id in enumerate(self.elector_data_full.index):
                previous_vote_candidate_id = current_votes.get(str(elector_id)) # Ensure elector_id is string for dict key
                if previous_vote_candidate_id is not None:
                    try:
                        # Find index of previously voted candidate among effective candidates
                        previous_vote_cand_idx_effective = effective_candidate_names.index(str(previous_vote_candidate_id))
                        current_pref_scores[elector_idx, previous_vote_cand_idx_effective] *= (1 + self.stickiness_factor)
                    except ValueError:
                        # Previous candidate not in active list, stickiness doesn't apply to them
                        pass 

            # Bandwagon effect: Increase preference based on overall vote counts for candidates
            if self.bandwagon_strength > 0 and num_effective_candidates > 0:
                # Calculate vote counts for each effective candidate from current_votes
                vote_counts = np.zeros(num_effective_candidates, dtype=float)
                for elector_id_key, voted_candidate_id_val in current_votes.items():
                    try:
                        voted_cand_idx_effective = effective_candidate_names.index(str(voted_candidate_id_val))
                        vote_counts[voted_cand_idx_effective] += 1
                    except ValueError:
                        pass # Vote for a candidate not in the current active list
                
                # Normalize vote counts to get a bandwagon score (0 to 1)
                if np.sum(vote_counts) > 0:
                    bandwagon_scores = vote_counts / np.sum(vote_counts) 
                    current_pref_scores += self.bandwagon_strength * bandwagon_scores[np.newaxis, :]
        
        # --- Final Normalization ---
        # Ensure no self-voting (diagonal of original square matrix, if all candidates are electors)
        # If active_candidates is a subset, self-voting might not be on the diagonal of current_pref_scores.
        # Need to identify if an elector IS one of the active_candidates.
        for elector_idx, elector_id_str in enumerate(self.elector_data_full.index.astype(str)):
            if elector_id_str in effective_candidate_names:
                cand_idx_in_effective = effective_candidate_names.index(elector_id_str)
                current_pref_scores[elector_idx, cand_idx_in_effective] = 0
        
        row_sums = current_pref_scores.sum(axis=1, keepdims=True)
        prob_matrix = np.zeros_like(current_pref_scores)
        
        non_zero_sum_rows = (row_sums > 1e-9).flatten()
        if np.any(non_zero_sum_rows):
            # Ensure sums are not zero before division
            safe_sums = np.where(row_sums[non_zero_sum_rows] == 0, 1, row_sums[non_zero_sum_rows])
            prob_matrix[non_zero_sum_rows, :] = current_pref_scores[non_zero_sum_rows, :] / safe_sums

        # For electors with zero sum of preferences (e.g., all candidates filtered out or infinitely distant)
        # assign uniform probability if there are candidates, otherwise leave as zeros.
        zero_sum_rows = ~non_zero_sum_rows
        if np.any(zero_sum_rows) and num_effective_candidates > 0:
            num_votable_for_row = np.sum(current_pref_scores[zero_sum_rows, :] > 0, axis=1, keepdims=True)
            # Create a mask for rows that truly have no positive preference scores left
            truly_zero_preference_rows = (num_votable_for_row == 0).flatten()
            
            if np.any(truly_zero_preference_rows):
                indices_truly_zero = np.where(zero_sum_rows)[0][truly_zero_preference_rows]
                log.warning(f"Electors at indices {indices_truly_zero} have zero preference for all {num_effective_candidates} active candidates. Assigning uniform probability.")
                uniform_prob = 1.0 / num_effective_candidates
                prob_matrix[indices_truly_zero, :] = uniform_prob
                 # Re-apply self-vote zeroing for these uniform rows if elector is a candidate
                for elector_idx in indices_truly_zero:
                    elector_id_str = self.elector_data_full.index[elector_idx].astype(str)
                    if elector_id_str in effective_candidate_names:
                        cand_idx_in_effective = effective_candidate_names.index(elector_id_str)
                        prob_matrix[elector_idx, cand_idx_in_effective] = 0
                        # Renormalize this specific row if a self-vote was zeroed out
                        if num_effective_candidates > 1:
                            row_sum_after_self_vote_fix = prob_matrix[elector_idx, :].sum()
                            if row_sum_after_self_vote_fix > 1e-9:
                                prob_matrix[elector_idx, :] /= row_sum_after_self_vote_fix
                            else: # All other candidates also had zero prob, revert to uniform among N-1
                                prob_matrix[elector_idx, :] = 1.0 / (num_effective_candidates -1)
                                prob_matrix[elector_idx, cand_idx_in_effective] = 0 # re-set self to 0
                        elif num_effective_candidates == 1: # Only self as candidate, prob must be 0
                             prob_matrix[elector_idx, cand_idx_in_effective] = 0

        # Final validation of row sums
        final_sums = prob_matrix.sum(axis=1)
        # Allow sums to be 0 if there are no effective candidates an elector can vote for (e.g. N=1 and self-vote=0)
        valid_sums = np.isclose(final_sums, 1.0, rtol=1e-5, atol=1e-7) | \
                     (np.isclose(final_sums, 0.0, atol=1e-7) & (num_effective_candidates <=1 )) # Or if only 1 candidate and it's self
        
        if not np.all(valid_sums):
            invalid_row_indices = np.where(~valid_sums)[0]
            problematic_elector_ids_actual = self.elector_data_full.index[invalid_row_indices].tolist()
            log.error(f"TransitionModel: Probability matrix normalization failed for electors: {problematic_elector_ids_actual} (Indices: {invalid_row_indices})")
            log.error(f"Problematic sums: {final_sums[invalid_row_indices]}")
            log.error(f"Problematic prob_matrix rows:\n{prob_matrix[invalid_row_indices]}")
            # raise ValueError("Transition probability matrix rows do not sum correctly to 1.0 or 0.0.")
            # For now, log error but don't raise, to allow inspection if this is hit. Strict check later.

        return prob_matrix, effective_candidate_names


# Example usage (simplified):
# elector_data_example = pd.DataFrame({
#     'ideology_score': [0.1, 0.3, 0.5, 0.7, 0.9],
#     'region': ['North', 'North', 'South', 'South', 'North'],
#     'is_papabile': [True, False, True, False, False]
# }, index=['elector1', 'elector2', 'elector3', 'elector4', 'elector5'])
# model = TransitionModel(elector_data_example)
# prob_matrix, effective_candidates = model.calculate_transition_probabilities(elector_data_example)
