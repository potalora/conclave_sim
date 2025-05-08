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
            - 'papabile_weight_factor': Multiplicative weight for papabile candidate.
    """

    def __init__(self, elector_data: pd.DataFrame,
                 initial_beta_weight: float = 1.0,
                 enable_dynamic_beta: bool = False,
                 beta_growth_rate: float = 1.05, # Rate per round, e.g., 1.05 for 5% growth
                 stickiness_factor: float = 0.5,
                 bandwagon_strength: float = 0.0,
                 regional_affinity_bonus: float = 0.1,
                 papabile_weight_factor: float = 1.5,
                 enable_candidate_fatigue: bool = False,
                 fatigue_penalty_factor: float = 0.5, # Multiplicative penalty
                 fatigue_min_vote_share_threshold: float = 0.01,
                 fatigue_min_rounds_threshold: int = 3,
                 enable_stop_candidate: bool = False,
                 stop_candidate_boost_factor: float = 1.5, # Multiplicative boost
                 stop_candidate_threshold_unacceptable_distance: float = 2.0,
                 stop_candidate_threat_min_vote_share: float = 0.15
                 ):

        if not isinstance(elector_data, pd.DataFrame):
            raise ValueError("elector_data must be a pandas DataFrame.")
        if elector_data.empty:
            raise ValueError("elector_data must be a non-empty DataFrame.")

        self.elector_data = elector_data.copy()

        # Validate numeric parameters
        if initial_beta_weight < 0:
            raise ValueError("initial_beta_weight must be non-negative.")
        if not (0 <= stickiness_factor <= 1):
            raise ValueError("stickiness_factor must be between 0 and 1.")
        if bandwagon_strength < 0:
            raise ValueError("bandwagon_strength must be non-negative.")
        if regional_affinity_bonus < 0:
            raise ValueError("regional_affinity_bonus must be non-negative.")
        if papabile_weight_factor < 0:
            raise ValueError("papabile_weight_factor must be non-negative.")
        if beta_growth_rate <= 0:
             raise ValueError("beta_growth_rate must be positive.")
        if fatigue_penalty_factor < 0 or fatigue_penalty_factor > 1:
            raise ValueError("fatigue_penalty_factor must be between 0 and 1.")
        if fatigue_min_vote_share_threshold < 0 or fatigue_min_vote_share_threshold > 1:
            raise ValueError("fatigue_min_vote_share_threshold must be between 0 and 1.")
        if fatigue_min_rounds_threshold < 0:
            raise ValueError("fatigue_min_rounds_threshold must be non-negative.")
        if stop_candidate_boost_factor < 1:
            raise ValueError("stop_candidate_boost_factor must be >= 1.")
        if stop_candidate_threshold_unacceptable_distance < 0:
            raise ValueError("stop_candidate_threshold_unacceptable_distance must be non-negative.")
        if stop_candidate_threat_min_vote_share < 0 or stop_candidate_threat_min_vote_share > 1:
            raise ValueError("stop_candidate_threat_min_vote_share must be between 0 and 1.")

        self.initial_beta_weight = initial_beta_weight
        self.enable_dynamic_beta = enable_dynamic_beta
        self.beta_growth_rate = beta_growth_rate
        self.stickiness_factor = stickiness_factor
        self.bandwagon_strength = bandwagon_strength
        self.regional_affinity_bonus = regional_affinity_bonus
        self.papabile_weight_factor = papabile_weight_factor
        self.enable_candidate_fatigue = enable_candidate_fatigue
        self.fatigue_penalty_factor = fatigue_penalty_factor
        self.fatigue_min_vote_share_threshold = fatigue_min_vote_share_threshold
        self.fatigue_min_rounds_threshold = fatigue_min_rounds_threshold
        self.enable_stop_candidate = enable_stop_candidate
        self.stop_candidate_boost_factor = stop_candidate_boost_factor
        self.stop_candidate_threshold_unacceptable_distance = stop_candidate_threshold_unacceptable_distance
        self.stop_candidate_threat_min_vote_share = stop_candidate_threat_min_vote_share

        self.effective_beta_weight = initial_beta_weight

        required_cols = ['ideology_score', 'region', 'is_papabile']
        if not all(col in self.elector_data.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in self.elector_data.columns]
            raise ValueError(f"elector_data must contain '{', '.join(missing_cols)}' column(s).")

        # Standardize elector_id as index
        if 'elector_id' in self.elector_data.columns:
            if self.elector_data['elector_id'].duplicated().any():
                raise ValueError("Duplicate values found in 'elector_id' column.")
            self.elector_data = self.elector_data.set_index('elector_id', drop=True)
        elif self.elector_data.index.name != 'elector_id':
            log.warning("Elector data index is not named 'elector_id' and 'elector_id' column not found. Model might not function as expected if index is not unique elector identifiers.")
            # Proceeding with caution, assuming index is intended to be elector_id

        self.num_electors = len(self.elector_data)
        self.num_candidates = len(self.elector_data)
        self.candidate_data = self.elector_data.copy()
        self.candidate_ids = self.candidate_data.index.tolist()

        # Pre-calculate attributes for performance
        self.elector_regions = self.elector_data['region'].values
        self.candidate_regions = self.candidate_data['region'].values
        self.elector_ideologies = self.elector_data['ideology_score'].values
        self.candidate_ideologies = self.candidate_data['ideology_score'].values
        self.candidate_is_papabile = self.candidate_data['is_papabile'].astype(bool).values

    def calculate_transition_probabilities(self, previous_round_votes: pd.Series = None, 
                                         current_round_num: int = 1,
                                         fatigued_candidate_indices: set = None,
                                         candidate_vote_shares_current_round: np.ndarray = None):
        if self.num_electors == 0 or self.num_candidates == 0:
            log.warning("No electors or candidates to calculate transition probabilities for.")
            return np.array([]).reshape(self.num_electors, 0), []

        # A. Dynamic Beta Calculation
        if self.enable_dynamic_beta:
            current_dynamic_beta = self.initial_beta_weight * (self.beta_growth_rate ** (current_round_num - 1))
        else:
            current_dynamic_beta = self.initial_beta_weight
        # log.debug(f"Round {current_round_num}, Dynamic Beta: {current_dynamic_beta:.2f}")

        # 1. Ideological Preference Scores
        ideological_distances = np.abs(self.elector_ideologies[:, np.newaxis] - self.candidate_ideologies)
        current_pref_scores = np.exp(-current_dynamic_beta * ideological_distances)

        # 2. Apply Multiplicative Papabile Weight
        papabile_candidates = np.where(self.candidate_is_papabile)[0]
        if papabile_candidates.size > 0:
            current_pref_scores[:, papabile_candidates] *= self.papabile_weight_factor

        # 3. Apply Additive Regional Bonus
        # Create a boolean matrix for shared regions
        same_region_matrix = (self.elector_regions[:, np.newaxis] == self.candidate_regions)
        current_pref_scores[same_region_matrix] += self.regional_affinity_bonus
        
        # 4. Apply Additive Bandwagon Effect
        if self.bandwagon_strength > 0 and previous_round_votes is not None:
            # Ensure previous_round_votes is a Series for consistent handling
            if isinstance(previous_round_votes, dict):
                previous_round_votes_series = pd.Series(previous_round_votes, dtype=object)
            elif isinstance(previous_round_votes, pd.Series):
                previous_round_votes_series = previous_round_votes
            else:
                # If not a dict or Series, log a warning and treat as empty to avoid errors
                logging.warning(f"Unsupported type for previous_round_votes in bandwagon: {type(previous_round_votes)}. Skipping effect.")
                previous_round_votes_series = pd.Series(dtype=object) # Empty series

            if not previous_round_votes_series.empty:
                valid_previous_votes = previous_round_votes_series[previous_round_votes_series.isin(self.candidate_ids)]

                if not valid_previous_votes.empty:
                    vote_counts = valid_previous_votes.value_counts()
                    max_votes = vote_counts.max() if not vote_counts.empty else 1
                    if max_votes == 0: max_votes = 1

                    bandwagon_bonuses_aligned = pd.Series(0.0, index=self.candidate_ids)
                    for candidate_id, count in vote_counts.items():
                        if candidate_id in self.candidate_ids:
                            bonus = (count / max_votes) * self.bandwagon_strength
                            bandwagon_bonuses_aligned[candidate_id] = bonus
                    
                    current_pref_scores += bandwagon_bonuses_aligned.values
    
        # 5. Apply Multiplicative Stickiness Factor
        if previous_round_votes is not None and self.stickiness_factor > 0:
            # Identify electors who voted in the previous round and their chosen candidate
            # Assuming previous_round_votes is a Series with elector_id as index and candidate_id as value
            for elector_id_prev_vote, chosen_candidate_id_prev_vote in previous_round_votes.items():
                if elector_id_prev_vote in self.elector_data.index and chosen_candidate_id_prev_vote in self.candidate_ids:
                    elector_idx = self.elector_data.index.get_loc(elector_id_prev_vote)
                    try:
                        # Ensure chosen_candidate_id_prev_vote is compatible with candidate_ids type (e.g. both str or int)
                        chosen_candidate_idx = self.candidate_ids.index(chosen_candidate_id_prev_vote)
                        current_pref_scores[elector_idx, chosen_candidate_idx] *= (1 + self.stickiness_factor)
                    except ValueError:
                        log.warning(f"Stickiness: Previously chosen candidate ID '{chosen_candidate_id_prev_vote}' not found in current candidate_ids list.")
                        pass 

        # Ensure scores are not negative after additive bonuses before fatigue/stop candidate logic
        current_pref_scores = np.maximum(current_pref_scores, 0)

        # C. Candidate Fatigue Logic
        if self.enable_candidate_fatigue and fatigued_candidate_indices:
            fatigued_indices_list = list(fatigued_candidate_indices) 
            if fatigued_indices_list: 
                # log.debug(f"Applying fatigue penalty to candidates: {fatigued_indices_list}")
                current_pref_scores[:, fatigued_indices_list] *= self.fatigue_penalty_factor

        base_pref_scores_after_fatigue = current_pref_scores.copy()

        # D. Stop Candidate Logic
        # Requires candidate_vote_shares_current_round and elector/candidate ideologies
        if self.enable_stop_candidate and candidate_vote_shares_current_round is not None:
            # log.debug(f"Applying Stop Candidate Logic. Threshold dist: {self.stop_candidate_threshold_unacceptable_distance}, threat share: {self.stop_candidate_threat_min_vote_share}")
            for e_idx in range(self.num_electors):
                elector_ideology = self.elector_ideologies[e_idx]
                unacceptable_k_indices = np.where(
                    np.abs(elector_ideology - self.candidate_ideologies) > self.stop_candidate_threshold_unacceptable_distance
                )[0]

                if unacceptable_k_indices.size > 0:
                    threatening_unacceptable_indices = [
                        k_idx for k_idx in unacceptable_k_indices 
                        if candidate_vote_shares_current_round[k_idx] > self.stop_candidate_threat_min_vote_share
                    ]

                    if threatening_unacceptable_indices: 
                        # log.debug(f"Elector {self.elector_data.index[e_idx]} finds candidates {threatening_unacceptable_indices} threatening.")
                        potential_blockers_indices = np.array(
                            [m_idx for m_idx in range(self.num_candidates) if m_idx not in unacceptable_k_indices]
                        )
                        
                        if potential_blockers_indices.size > 0:
                            # Apply boost to all potential blockers
                            base_pref_scores_after_fatigue[e_idx, potential_blockers_indices] *= self.stop_candidate_boost_factor
            current_pref_scores = base_pref_scores_after_fatigue 

        # E. Set diagonal to zero (no self-votes)
        np.fill_diagonal(current_pref_scores, 0) # AI: Ensure no self-votes

        # Handle cases where all preference scores for an elector might be zero
        # If all scores for an elector are zero (e.g., due to extreme fatigue penalties or no viable candidates),
        # assign a tiny uniform probability to prevent NaN in normalization and allow random choice.
        for i in range(self.num_electors):
            if np.sum(current_pref_scores[i, :]) == 0:
                # log.debug(f"All preference scores are zero for elector {self.elector_data.index[i]}. Assigning uniform probability.")
                current_pref_scores[i, :] = 1e-9 

        # F. Normalization to Probabilities
        row_sums = current_pref_scores.sum(axis=1, keepdims=True)
        probabilities = current_pref_scores / row_sums

        details = []
        for e_idx in range(self.num_electors):
            candidate_details_for_elector = []
            for c_idx in range(self.num_candidates):
                candidate_details_for_elector.append({
                    'candidate_id': self.candidate_ids[c_idx],
                    'final_utility_before_softmax': current_pref_scores[e_idx, c_idx]
                })
            
            detail_entry = {
                'round': current_round_num,
                'elector_id': self.elector_data.index[e_idx],
                'effective_beta_weight': current_dynamic_beta,
                'candidate_details': candidate_details_for_elector,
                'probabilities': probabilities[e_idx, :].tolist(),
                'fatigued_candidates_in_round': list(fatigued_candidate_indices) if fatigued_candidate_indices else []
            }
            details.append(detail_entry)

        return probabilities, details

    def get_candidate_ids(self):
        return self.candidate_ids

    def get_elector_data(self):
        return self.elector_data

# Example usage (simplified):
# elector_data_example = pd.DataFrame({
#     'ideology_score': [0.1, 0.3, 0.5, 0.7, 0.9],
#     'region': ['North', 'North', 'South', 'South', 'North'],
#     'is_papabile': [True, False, True, False, False]
# }, index=['elector1', 'elector2', 'elector3', 'elector4', 'elector5'])
# model = TransitionModel(elector_data_example)
# prob_matrix, effective_candidates = model.calculate_transition_probabilities(elector_data_example)
