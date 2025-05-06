import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union


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
            elector_data: DataFrame containing elector profiles. Must have 'elector_id'
                          and 'ideology_score' columns.
            current_votes: Optional dictionary mapping elector_id (voter) to the
                           elector_id (candidate) they voted for in the *previous* round.
                           If None (e.g., first round), only ideology is considered.

        Returns:
            A numpy array (matrix) where rows represent electors (voters) and columns
            represent candidates (all electors), containing the probability of each
            elector voting for each candidate in the next round.

        Raises:
            ValueError: If elector_data is empty or missing required columns, or if
                        current_votes contains IDs not in elector_data.
            KeyError: If required parameters are missing (should be caught in __init__).
        """

        if elector_data.empty:
            raise ValueError("Cannot calculate probabilities with zero electors.")
        required_cols = ['elector_id', 'ideology_score']
        for col in required_cols:
            if col not in elector_data.columns:
                raise ValueError(f"Elector data must contain '{col}'.")

        beta = self.parameters['beta_weight']
        stickiness = self.parameters['stickiness_factor']
        num_electors = len(elector_data)
        num_candidates = num_electors # Assume all electors are potential candidates

        # Create mapping from elector ID to matrix index for quick lookup
        # Ensure index aligns with 0 to num_electors-1 for matrix operations
        elector_ids = elector_data['elector_id'].tolist()
        id_to_index = {e_id: i for i, e_id in enumerate(elector_ids)}

        # --- 1. Calculate Base Probabilities (Ideology) ---
        ideology_scores = elector_data['ideology_score'].values
        ideology_voters = ideology_scores[:, np.newaxis]
        ideology_candidates = ideology_scores[np.newaxis, :]
        diff_sq_matrix = (ideology_voters - ideology_candidates) ** 2
        unnormalized_probs = np.exp(-beta * diff_sq_matrix)

        # --- 2. Apply Vote Stickiness (if not first round) ---
        if current_votes is not None and stickiness > 0:
            print(f"Applying stickiness factor: {stickiness}")
            # Iterate through previous votes
            for voter_id, candidate_id in current_votes.items():
                if voter_id not in id_to_index:
                    print(f"Warning: Voter ID {voter_id} from current_votes not found in elector_data. Skipping.")
                    continue
                if candidate_id not in id_to_index:
                    print(f"Warning: Candidate ID {candidate_id} voted for by {voter_id} not found in elector_data. Skipping stickiness boost.")
                    continue

                voter_idx = id_to_index[voter_id]
                candidate_idx = id_to_index[candidate_id]

                # Boost the probability for the previously chosen candidate
                boost_multiplier = (1.0 + stickiness)
                #print(f" Boosting voter {voter_id} (idx {voter_idx}) for cand {candidate_id} (idx {candidate_idx}): {unnormalized_probs[voter_idx, candidate_idx]:.4f} *= {boost_multiplier:.2f}")
                unnormalized_probs[voter_idx, candidate_idx] *= boost_multiplier
                #print(f"  -> New value: {unnormalized_probs[voter_idx, candidate_idx]:.4f}")

        # --- 3. Normalize Probabilities ---
        row_sums = unnormalized_probs.sum(axis=1, keepdims=True)

        # Avoid division by zero if an elector has zero probability for all candidates
        zero_mask = (row_sums == 0)
        if np.any(zero_mask):
            print("Warning: Zero probability sum encountered for some electors. Assigning uniform probability to them.")
            # Assign uniform probability (1/N) where sum is zero
            uniform_prob = 1.0 / num_candidates
            unnormalized_probs[zero_mask.flatten(), :] = uniform_prob
            # Recalculate row sums only where needed (should now be 1.0)
            row_sums[zero_mask] = 1.0

        probabilities = unnormalized_probs / row_sums

        # Optional: Verify row sums are close to 1
        # if not np.allclose(probabilities.sum(axis=1), 1.0):
        #     print("Warning: Probabilities do not sum to 1 for all electors after normalization.")
        #     # Consider logging problematic rows/sums here for debugging

        return probabilities
