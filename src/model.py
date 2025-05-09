import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
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

    log = logging.getLogger(__name__ + ".TransitionModel") 

    def _validate_elector_data(self, elector_data: pd.DataFrame) -> None:
        """Validates elector_data structure, specifically index name and 'elector_id' column."""
        if elector_data.empty: # No further validation needed for empty data anent index/columns
            return

        if elector_data.index.name != 'elector_id':
            if 'elector_id' not in elector_data.columns:
                self.log.warning(
                    "Elector data index is not named 'elector_id' and 'elector_id' column not found. "
                    "Model might not function as expected if index is not unique elector identifiers."
                )
            else:
                # Index not 'elector_id', but 'elector_id' column exists. This might be okay if user intends to use the column.
                self.log.info(
                    "Elector data index is not named 'elector_id', but an 'elector_id' column was found. "
                    "Ensure the 'elector_id' column contains unique elector identifiers if the index is not used for this purpose."
                )
        elif 'elector_id' in elector_data.columns:
            # Index IS 'elector_id', but there's ALSO an 'elector_id' column. Could be confusing.
            self.log.info(
                "Elector data index is named 'elector_id', and an 'elector_id' column also exists. "
                "Ensure the index correctly represents unique elector identifiers."
            )

    def __init__(self, elector_data: pd.DataFrame,
                 initial_beta_weight: float = 1.0, 
                 enable_dynamic_beta: bool = False,
                 beta_growth_rate: float = 1.05, # Rate per round, e.g., 1.05 for 5% growth
                 beta_increment_amount: Optional[float] = None, # Specific amount for stepped increment
                 beta_increment_interval_rounds: int = 1, # Rounds between stepped increments
                 stickiness_factor: float = 0.0, # Adherence to previous vote
                 bandwagon_strength: float = 0.0, # Tendency to follow popular candidates
                 regional_affinity_bonus: float = 0.0, # Bonus for shared region
                 papabile_weight_factor: float = 1.0, # Multiplicative factor for papabile candidates
                 enable_candidate_fatigue: bool = False,
                 fatigue_threshold_rounds: int = 3, # Rounds of low support to trigger fatigue
                 fatigue_vote_share_threshold: float = 0.05, # Vote share below which a candidate is considered for fatigue
                 fatigue_penalty_factor: float = 0.5, # Multiplicative penalty for fatigued candidates
                 fatigue_top_n_immune: int = 0, # Top N candidates immune from fatigue
                 enable_stop_candidate: bool = False,
                 stop_candidate_threshold_unacceptable_distance: float = 1.0, # Ideological distance beyond which a candidate is 'unacceptable'
                 stop_candidate_threat_min_vote_share: float = 0.15, # Min vote share for an 'unacceptable' candidate to be a 'threat'
                 stop_candidate_boost_factor: float = 1.5 # Multiplicative boost for 'blocker' candidates
                 ):
        self._validate_elector_data(elector_data) # Call validation at the beginning

        if elector_data.empty:
            log.warning("TransitionModel initialized with empty elector_data. Most operations will result in empty outputs.")
        
        self.elector_data = elector_data
        self.num_electors = len(elector_data)
        self.elector_ideologies = elector_data['ideology_score'].values if 'ideology_score' in elector_data else np.array([])
        self.elector_regions = elector_data['region'].values if 'region' in elector_data else np.array([])

        # Assuming candidate_data is derived from elector_data (all electors are potential candidates)
        if not elector_data.empty and 'ideology_score' in elector_data.columns and 'region' in elector_data.columns:
            self.candidate_ids = list(elector_data.index) # Master list of all possible candidate IDs in original order
            self.candidate_ids_set = set(self.candidate_ids) # For quick lookups
            self.num_total_candidates = len(self.candidate_ids) # Total number of candidates in master list
            
            # These are master attribute arrays, aligned with self.candidate_ids
            self.master_candidate_ideologies = self.elector_ideologies 
            self.master_candidate_regions = self.elector_regions
            self.master_candidate_is_papabile = np.full(self.num_total_candidates, True, dtype=bool)
            if 'is_papabile' in elector_data:
                self.master_candidate_is_papabile = elector_data['is_papabile'].astype(bool).values
            else:
                log.info("'is_papabile' column not found in elector_data. Assuming all candidates are papabile.")
        else:
            self.candidate_ids = []
            self.candidate_ids_set = set()
            self.num_total_candidates = 0 # Ensure this is also set to 0
            self.master_candidate_ideologies = np.array([])
            self.master_candidate_regions = np.array([])
            self.master_candidate_is_papabile = np.array([], dtype=bool)

        self.initial_beta_weight = initial_beta_weight
        self.effective_beta_weight = float(initial_beta_weight) if initial_beta_weight is not None else 0.0
        self.enable_dynamic_beta = enable_dynamic_beta
        self.beta_growth_rate = beta_growth_rate
        self.beta_increment_amount = beta_increment_amount
        self.beta_increment_interval_rounds = beta_increment_interval_rounds

        self.stickiness_factor = stickiness_factor
        self.bandwagon_strength = bandwagon_strength
        self.regional_affinity_bonus = regional_affinity_bonus
        self.papabile_weight_factor = papabile_weight_factor

        self.enable_candidate_fatigue = enable_candidate_fatigue
        self.fatigue_threshold_rounds = fatigue_threshold_rounds
        self.fatigue_vote_share_threshold = fatigue_vote_share_threshold
        self.fatigue_penalty_factor = fatigue_penalty_factor
        self.fatigue_top_n_immune = fatigue_top_n_immune

        self.enable_stop_candidate = enable_stop_candidate
        self.stop_candidate_threshold_unacceptable_distance = stop_candidate_threshold_unacceptable_distance
        self.stop_candidate_threat_min_vote_share = stop_candidate_threat_min_vote_share
        self.stop_candidate_boost_factor = stop_candidate_boost_factor

        if self.enable_dynamic_beta:
            if self.beta_increment_amount is not None and self.beta_growth_rate != 1.0 and self.beta_growth_rate != 1.05: # 1.05 was a default placeholder sometimes
                 log.warning("Both beta_increment_amount and a custom beta_growth_rate are set. Stepped increment (beta_increment_amount) will be prioritized.")
            elif self.beta_increment_amount is None and self.beta_growth_rate <= 1.0:
                 log.warning(f"Dynamic beta enabled, beta_increment_amount is None, and beta_growth_rate ({self.beta_growth_rate}) is not > 1.0. Beta will not change multiplicatively.")
            # No warning if only one is appropriately set or if growth rate is > 1.0 for multiplicative, or amount is set for stepped.
        elif not self.enable_dynamic_beta and self.initial_beta_weight is None:
            log.warning("Initial_beta_weight is None and dynamic_beta is not enabled. Effective_beta_weight defaults to 0.0. This might lead to uniform probabilities if not intended.")

    # AI: Helper to map original candidate indices to actual candidate indices
    def _get_actual_candidate_map_and_indices(self, actual_candidate_ids_list: List[Any]) -> Tuple[Dict[int, int], List[int]]:
        map_original_idx_to_actual_idx: Dict[int, int] = {}
        original_indices_for_actual_candidates: List[int] = []
        for actual_idx, actual_cand_id in enumerate(actual_candidate_ids_list):
            try:
                original_idx = self.candidate_ids.index(actual_cand_id)
                map_original_idx_to_actual_idx[original_idx] = actual_idx
                original_indices_for_actual_candidates.append(original_idx)
            except ValueError:
                log.error(f"Consistency error: Candidate ID {actual_cand_id} from actual list not in master candidate_ids.")
        return map_original_idx_to_actual_idx, original_indices_for_actual_candidates

    def calculate_transition_probabilities(self, previous_round_votes: pd.Series = None, 
                                         current_round_num: int = 1,
                                         fatigued_candidate_indices: Optional[set] = None, # Original indices
                                         candidate_vote_shares_current_round: Optional[np.ndarray] = None, # Aligned with self.candidate_ids
                                         effective_candidate_ids: Optional[List[Any]] = None):
        
        # Determine the actual set of candidates for this round's calculation
        actual_candidate_ids_list: List[Any]
        original_indices_for_actual: List[int]
        map_original_idx_to_actual_idx: Dict[int, int] # AI: Ensure this map is available

        if effective_candidate_ids is not None and len(effective_candidate_ids) > 0:
            # Filter provided effective_candidate_ids against master list to ensure validity and order
            actual_candidate_ids_list = [cid for cid in effective_candidate_ids if cid in self.candidate_ids_set]
            if not actual_candidate_ids_list:
                log.warning("Provided effective_candidate_ids resulted in an empty list after validation. Falling back to all candidates.")
                actual_candidate_ids_list = list(self.candidate_ids) # shallow copy
                original_indices_for_actual = list(range(self.num_total_candidates))
                # Create a simple map for all candidates if falling back
                map_original_idx_to_actual_idx = {i: i for i in range(self.num_total_candidates)}
            else:
                map_original_idx_to_actual_idx, original_indices_for_actual = self._get_actual_candidate_map_and_indices(actual_candidate_ids_list)
        else:
            actual_candidate_ids_list = list(self.candidate_ids) # shallow copy
            original_indices_for_actual = list(range(self.num_total_candidates))
            map_original_idx_to_actual_idx = {i: i for i in range(self.num_total_candidates)} # Map for all candidates

        num_actual_candidates = len(actual_candidate_ids_list)

        if self.num_electors == 0 or num_actual_candidates == 0:
            log.warning("No electors or actual candidates to calculate transition probabilities for.")
            # Return shape consistent with num_actual_candidates
            return np.array([]).reshape(self.num_electors, num_actual_candidates), [] 

        self.update_beta_weight(current_round_num)
        current_dynamic_beta = self.effective_beta_weight

        # Filter master candidate attributes to match actual_candidate_ids_list
        actual_candidate_ideologies = self.master_candidate_ideologies[original_indices_for_actual]
        actual_candidate_regions = self.master_candidate_regions[original_indices_for_actual]
        actual_candidate_is_papabile = self.master_candidate_is_papabile[original_indices_for_actual]

        # 1. Ideological Preference Scores (Dimensions: num_electors x num_actual_candidates)
        ideological_distances = np.abs(self.elector_ideologies[:, np.newaxis] - actual_candidate_ideologies)
        current_pref_scores = np.exp(-current_dynamic_beta * ideological_distances)

        # 2. Apply Multiplicative Papabile Weight (to actual candidates)
        # np.where on actual_candidate_is_papabile gives indices relative to the 'actual' set
        papabile_actual_indices = np.where(actual_candidate_is_papabile)[0]
        if papabile_actual_indices.size > 0:
            current_pref_scores[:, papabile_actual_indices] *= self.papabile_weight_factor

        # 3. Apply Additive Regional Bonus (for actual candidates)
        same_region_matrix_actual = (self.elector_regions[:, np.newaxis] == actual_candidate_regions)
        current_pref_scores[same_region_matrix_actual] += self.regional_affinity_bonus
        
        # 4. Apply Additive Bandwagon Effect (adapted for actual candidates)
        if self.bandwagon_strength > 0 and previous_round_votes is not None:
            if isinstance(previous_round_votes, dict):
                previous_round_votes_series = pd.Series(previous_round_votes, dtype=object)
            elif isinstance(previous_round_votes, pd.Series):
                previous_round_votes_series = previous_round_votes
            else:
                logging.warning(f"Unsupported type for previous_round_votes in bandwagon: {type(previous_round_votes)}. Skipping effect.")
                previous_round_votes_series = pd.Series(dtype=object)

            if not previous_round_votes_series.empty:
                # Filter votes for candidates who are in the current actual_candidate_ids_list
                valid_previous_votes_for_actual = previous_round_votes_series[previous_round_votes_series.isin(actual_candidate_ids_list)]

                if not valid_previous_votes_for_actual.empty:
                    vote_counts_actual = valid_previous_votes_for_actual.value_counts()
                    max_votes_actual = vote_counts_actual.max() if not vote_counts_actual.empty else 1
                    if max_votes_actual == 0: max_votes_actual = 1

                    bandwagon_bonuses_actual = np.zeros(num_actual_candidates) # Aligned with actual_candidate_ids_list
                    for actual_cand_id, count in vote_counts_actual.items():
                        # actual_cand_id is already confirmed to be in actual_candidate_ids_list by isin filter
                        try:
                            actual_idx = actual_candidate_ids_list.index(actual_cand_id)
                            bonus = (count / max_votes_actual) * self.bandwagon_strength
                            bandwagon_bonuses_actual[actual_idx] = bonus
                        except ValueError:
                             # Should not happen if actual_cand_id is from vote_counts_actual keys
                            log.error(f"Bandwagon: Candidate ID {actual_cand_id} from vote_counts_actual not found in actual_candidate_ids_list.")
                    current_pref_scores += bandwagon_bonuses_actual # Broadcasting to each elector row

        # 5. Apply Multiplicative Stickiness Factor (adapted for actual candidates)
        if previous_round_votes is not None and self.stickiness_factor > 0:
            for elector_id_prev_vote, chosen_candidate_id_prev_vote in previous_round_votes.items():
                if elector_id_prev_vote in self.elector_data.index and chosen_candidate_id_prev_vote in actual_candidate_ids_list:
                    elector_idx = self.elector_data.index.get_loc(elector_id_prev_vote)
                    try:
                        actual_chosen_candidate_idx = actual_candidate_ids_list.index(chosen_candidate_id_prev_vote)
                        current_pref_scores[elector_idx, actual_chosen_candidate_idx] *= (1 + self.stickiness_factor)
                    except ValueError:
                        # Should not happen if chosen_candidate_id_prev_vote is in actual_candidate_ids_list
                        log.warning(f"Stickiness: Previously chosen candidate ID '{chosen_candidate_id_prev_vote}' (actual) not found by index.")
                        pass 

        current_pref_scores = np.maximum(current_pref_scores, 0)

        # C. Candidate Fatigue Logic (adapted for actual candidates)
        if self.enable_candidate_fatigue and fatigued_candidate_indices: # fatigued_candidate_indices are ORIGINAL indices
            actual_fatigued_indices_to_penalize = []
            for original_fatigued_idx in fatigued_candidate_indices:
                if original_fatigued_idx in map_original_idx_to_actual_idx:
                    actual_fatigued_indices_to_penalize.append(map_original_idx_to_actual_idx[original_fatigued_idx])
            
            if actual_fatigued_indices_to_penalize:
                current_pref_scores[:, actual_fatigued_indices_to_penalize] *= self.fatigue_penalty_factor

        base_pref_scores_after_fatigue = current_pref_scores.copy()

        # D. Stop Candidate Logic (adapted for actual candidates)
        if self.enable_stop_candidate and candidate_vote_shares_current_round is not None: # candidate_vote_shares are ORIGINAL order
            for e_idx in range(self.num_electors):
                elector_ideology = self.elector_ideologies[e_idx]
                
                # Find unacceptable candidates based on master list ideologies (original indices)
                unacceptable_original_indices = np.where(
                    np.abs(elector_ideology - self.master_candidate_ideologies) > self.stop_candidate_threshold_unacceptable_distance
                )[0]

                if unacceptable_original_indices.size > 0:
                    # Filter these by vote share (still original indices)
                    threatening_unacceptable_original_indices = [
                        k_orig_idx for k_orig_idx in unacceptable_original_indices 
                        if candidate_vote_shares_current_round[k_orig_idx] > self.stop_candidate_threat_min_vote_share
                    ]

                    if threatening_unacceptable_original_indices:
                        # Potential blockers are those NOT in unacceptable_original_indices (original indices)
                        potential_blockers_original_indices = np.array(
                            [m_orig_idx for m_orig_idx in range(self.num_total_candidates) if m_orig_idx not in unacceptable_original_indices]
                        )
                        
                        # Map these original blocker indices to actual_indices for boosting
                        actual_blocker_indices_to_boost = []
                        if potential_blockers_original_indices.size > 0:
                            for original_blocker_idx in potential_blockers_original_indices:
                                if original_blocker_idx in map_original_idx_to_actual_idx:
                                    actual_blocker_indices_to_boost.append(map_original_idx_to_actual_idx[original_blocker_idx])
                            
                            if actual_blocker_indices_to_boost:
                                base_pref_scores_after_fatigue[e_idx, actual_blocker_indices_to_boost] *= self.stop_candidate_boost_factor
            current_pref_scores = base_pref_scores_after_fatigue

        # E. Set diagonal to zero (no self-votes) for actual candidates
        for e_idx, elector_id in enumerate(self.elector_data.index):
            if elector_id in actual_candidate_ids_list:
                try:
                    actual_cand_idx_for_self_vote = actual_candidate_ids_list.index(elector_id)
                    current_pref_scores[e_idx, actual_cand_idx_for_self_vote] = 0
                except ValueError:
                    pass # Elector is not in the current actual_candidate_ids_list as a candidate

        # F. Normalization to Probabilities (adapted for actual_candidates)
        current_pref_scores = np.maximum(current_pref_scores, 0) # Ensure no negative scores from additive effects if any slipped through

        # AI: Add debug logging for NaNs/Infs before normalization
        if np.any(np.isnan(current_pref_scores)):
            nan_rows = np.where(np.isnan(current_pref_scores).any(axis=1))[0]
            log.debug(f"TransitionModel: Found NaNs in current_pref_scores for elector indices {nan_rows} (Round {current_round_num}) BEFORE normalization.")
            # Example: Log details for up to 3 such electors if needed later for more detail
            # for i_nan in nan_rows[:min(3, len(nan_rows))]:
            #     elector_id_log_nan = self.elector_data.index[i_nan] if i_nan < len(self.elector_data.index) else f"index {i_nan} (ID not found)"
            #     log.debug(f"  Elector {elector_id_log_nan} (idx {i_nan}) pre-norm scores with NaN: {current_pref_scores[i_nan, :]}")

        if np.any(np.isinf(current_pref_scores)):
            inf_rows = np.where(np.isinf(current_pref_scores).any(axis=1))[0]
            log.debug(f"TransitionModel: Found Infs in current_pref_scores for elector indices {inf_rows} (Round {current_round_num}) BEFORE normalization.")
            # Example: Log details for up to 3 such electors if needed later for more detail
            # for i_inf in inf_rows[:min(3, len(inf_rows))]:
            #     elector_id_log_inf = self.elector_data.index[i_inf] if i_inf < len(self.elector_data.index) else f"index {i_inf} (ID not found)"
            #     log.debug(f"  Elector {elector_id_log_inf} (idx {i_inf}) pre-norm scores with Inf: {current_pref_scores[i_inf, :]}")
        # AI: End of debug logging

        for i in range(self.num_electors):
            if np.sum(current_pref_scores[i, :]) == 0:
                current_pref_scores[i, :] = 1e-9 
        
        row_sums = current_pref_scores.sum(axis=1, keepdims=True)

        if np.any(row_sums == 0):
            log.warning(
                f"TransitionModel: Zero row_sum detected for some electors BEFORE normalization (num_actual_candidates: {num_actual_candidates}). Re-attempting fix."
            )
            zero_sum_indices = np.where(row_sums.flatten() == 0)[0]
            for i in zero_sum_indices:
                elector_id_log = self.elector_data.index[i] if i < len(self.elector_data.index) else f"index {i} (ID not found)"
                log.debug(f"TransitionModel: Forcing uniform tiny probabilities for elector index {i} (ID: {elector_id_log}) due to zero sum.")
                current_pref_scores[i, :] = 1e-9
                row_sums[i] = np.sum(current_pref_scores[i, :])
            
            if np.any(row_sums == 0):
                log.error(f"TransitionModel: CRITICAL - Zero row_sum persists after 1e-9 fix (num_actual_candidates: {num_actual_candidates}). Probable NaN/inf or all actual candidates have zero score.")
                problematic_indices = np.where(row_sums.flatten() == 0)[0]
                for p_idx in problematic_indices:
                    current_pref_scores[p_idx, :] = 1.0 / num_actual_candidates if num_actual_candidates > 0 else 0 # Avoid division by zero if num_actual_candidates is 0
                    row_sums[p_idx] = 1.0 if num_actual_candidates > 0 else 0
        
        # Handle case where row_sums might still be zero if num_actual_candidates was zero (though earlier checks should prevent this)
        # or if all scores became zero and num_actual_candidates is zero.
        # This check ensures we don't divide by zero in the final step.
        # Create a mask for rows where row_sums is zero.
        zero_row_sum_mask = (row_sums == 0).flatten() 
        probabilities = np.zeros_like(current_pref_scores)

        if num_actual_candidates > 0:
            # Calculate probabilities for rows with non-zero sums.
            non_zero_row_sum_mask = ~zero_row_sum_mask
            probabilities[non_zero_row_sum_mask] = current_pref_scores[non_zero_row_sum_mask] / row_sums[non_zero_row_sum_mask]
            
            # For rows that had zero sums (and num_actual_candidates > 0), assign uniform probability.
            # This happens if the 1e-9 logic somehow resulted in a sum of 0 again, or if the critical error fallback was hit.
            if np.any(zero_row_sum_mask):
                probabilities[zero_row_sum_mask, :] = 1.0 / num_actual_candidates
        else: # num_actual_candidates is 0
            # All probabilities should be zero as there are no candidates to choose from.
            # The shape of probabilities is already (num_electors, 0) due to earlier return.
            pass # probabilities is already an empty array with correct shape from np.zeros_like or earlier return

        # AI: Add detailed temporary diagnostic logging for a specific elector and round
        if self.num_electors > 93 and current_round_num == 99: # Check if elector 93 exists
            elector_idx_to_debug = 93
            elector_id_to_debug = self.elector_data.index[elector_idx_to_debug] if elector_idx_to_debug < len(self.elector_data.index) else f"index {elector_idx_to_debug}"
            self.log.critical(f"--- START DETAILED DIAGNOSTIC FOR ELECTOR {elector_id_to_debug} (idx {elector_idx_to_debug}), ROUND {current_round_num} ---")
            self.log.critical(f"NumActualCandidates: {num_actual_candidates}")
            if elector_idx_to_debug < current_pref_scores.shape[0] and num_actual_candidates > 0:
                scores_slice = current_pref_scores[elector_idx_to_debug, :num_actual_candidates]
                self.log.critical(f"current_pref_scores[{elector_idx_to_debug}, :num_actual_candidates]: {scores_slice}")
                actual_row_sum = row_sums[elector_idx_to_debug]
                self.log.critical(f"row_sums[{elector_idx_to_debug}]: {actual_row_sum}")
                if actual_row_sum != 0 and np.isfinite(actual_row_sum):
                    normalized_slice = scores_slice / actual_row_sum
                    self.log.critical(f"scores_slice / actual_row_sum: {normalized_slice}")
                    self.log.critical(f"np.sum(scores_slice / actual_row_sum): {np.sum(normalized_slice)}")
                else:
                    self.log.critical(f"Skipping division for elector {elector_idx_to_debug} due to zero or non-finite row_sum: {actual_row_sum}")
                # Also log the probabilities slice as it is in the probabilities array at this point
                if elector_idx_to_debug < probabilities.shape[0] and probabilities.shape[1] > 0:
                     self.log.critical(f"probabilities[{elector_idx_to_debug}, :num_actual_candidates] after normalization attempt: {probabilities[elector_idx_to_debug, :num_actual_candidates]}")
                     self.log.critical(f"np.sum(probabilities[{elector_idx_to_debug}, :num_actual_candidates]): {np.sum(probabilities[elector_idx_to_debug, :num_actual_candidates])}")
            self.log.critical(f"--- END DETAILED DIAGNOSTIC FOR ELECTOR {elector_id_to_debug} (idx {elector_idx_to_debug}) ---")
        # AI: End of detailed temporary diagnostic logging

        # Immediate post-normalization check
        for i in range(self.num_electors):
            if num_actual_candidates > 0: # Only check sum if there are candidates
                # AI: BUG FIX - Only sum probabilities for ACTUAL candidates (0 to num_actual_candidates)
                # Instead of summing the entire row (which includes zeros beyond actual candidates)
                current_sum = np.sum(probabilities[i, :num_actual_candidates])
                
                # AI: Add detailed debug log for internal check
                model_internal_check_isclose_result = np.isclose(current_sum, 1.0)
                elector_id_for_debug = self.elector_data.index[i] if i < len(self.elector_data.index) else f"index {i} (ID not found)"
                self.log.warning(
                    f"TransitionModel Internal Check: Elector idx {i} (ID: {elector_id_for_debug}), Round {current_round_num}, "
                    f"NumActualCandidates: {num_actual_candidates}, CalculatedSum: {current_sum:.17f}, "
                    f"np.isclose(sum, 1.0) result: {model_internal_check_isclose_result}"
                )
                # AI: End of detailed debug log

                if not model_internal_check_isclose_result: # Original condition
                    elector_id_log = self.elector_data.index[i] if i < len(self.elector_data.index) else f"index {i} (ID not found)"
                    self.log.error(
                        f"INTERNAL MODEL ERROR (calc_trans_probs): Probabilities for elector index {i} "
                        f"(ID: {elector_id_log}) sum to {current_sum:.17f} with {num_actual_candidates} actual candidates. "
                        f"This should NOT happen. Investigate preference scores or normalization. "
                        f"Elector's pre-norm scores (current_pref_scores[i, :num_actual_candidates]): {current_pref_scores[i, :num_actual_candidates]}. "
                        f"Row sum used for division: {row_sums[i] if i < len(row_sums) else 'N/A'}."
                    )
                    # AI: BUG FIX - Only sum preference scores for actual candidates, not the entire row
                    faulty_row_sum_for_log = np.sum(current_pref_scores[i, :num_actual_candidates])
                    if faulty_row_sum_for_log != 0 and np.isfinite(faulty_row_sum_for_log):
                        self.log.warning(f"Force re-normalizing problematic row for elector {elector_id_log}. Original sum was {faulty_row_sum_for_log}")
                        # Only normalize the actual candidates slice
                        probabilities[i, :num_actual_candidates] = current_pref_scores[i, :num_actual_candidates] / faulty_row_sum_for_log
                        # Clear any values beyond num_actual_candidates
                        if probabilities.shape[1] > num_actual_candidates:
                            probabilities[i, num_actual_candidates:] = 0.0
                    else:
                        self.log.warning(f"Cannot force re-normalize for elector {elector_id_log} as sum is {faulty_row_sum_for_log}. Assigning uniform.")
                        # Clear the entire row first
                        probabilities[i, :] = 0.0
                        # Set only the actual candidates slice to uniform probability
                        probabilities[i, :num_actual_candidates] = 1.0 / num_actual_candidates
            elif probabilities.shape[1] > 0: # num_actual_candidates is 0, but probabilities matrix has columns (should not happen)
                self.log.error(f"INTERNAL MODEL ERROR: num_actual_candidates is 0, but probability matrix for elector index {i} has shape {probabilities[i,:].shape}")

        return probabilities, [] # Details list is empty for now

    def update_beta_weight(self, current_round_num: int):
        """Updates the effective beta weight if dynamic beta is enabled."""
        if not self.enable_dynamic_beta or current_round_num <= 0:
            return
            
        old_beta = self.effective_beta_weight

        if self.beta_increment_amount is not None:
            # Stepped increment takes precedence
            if self.beta_increment_interval_rounds > 0 and current_round_num % self.beta_increment_interval_rounds == 0:
                self.effective_beta_weight += self.beta_increment_amount
                self.log.debug(f"Round {current_round_num}: Dynamic beta (stepped) applied. Old: {old_beta:.4f}, New: {self.effective_beta_weight:.4f}, Increment: {self.beta_increment_amount:.4f}")
            # else: No increment this round for stepped if interval not met
        elif self.beta_growth_rate > 1.0:
            # Multiplicative growth if no stepped amount is set and growth rate is applicable
            self.effective_beta_weight *= self.beta_growth_rate
            self.log.debug(f"Round {current_round_num}: Dynamic beta (multiplicative) applied. Old: {old_beta:.4f}, New: {self.effective_beta_weight:.4f}, Rate: {self.beta_growth_rate:.4f}")
        # If neither condition is met (e.g., increment_amount is None and growth_rate <= 1.0), beta doesn't change.
        # This case should ideally be caught by a warning in __init__.

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
