import numpy as np
import pandas as pd
from typing import Dict, Any, Optional


class TransitionModel:
    """Models the transition probabilities of elector votes between rounds.

    Attributes:
        parameters: A dictionary containing model parameters (e.g., weights,
                    influence factors).
    """

    def __init__(self, parameters: Dict[str, Any]):
        """Initializes the TransitionModel.

        Args:
            parameters: Dictionary of model parameters.
        """
        self.parameters = parameters
        print(f"TransitionModel initialized with parameters: {self.parameters}")
        # Validate necessary parameters
        if 'beta_weight' not in self.parameters:
            raise ValueError("Model parameters must include 'beta_weight'.")
        # TODO: Add more parameter validation as model complexity increases

    def calculate_transition_probabilities(
        self, current_votes: pd.DataFrame, elector_data: pd.DataFrame
    ) -> np.ndarray:
        """Calculates the probability of each elector voting for each candidate.

        This version uses ideological distance weighted by beta_weight.

        Args:
            current_votes: DataFrame summarizing votes from the previous round (unused in this version).
            elector_data: DataFrame containing elector profiles. Must have 'elector_id'
                          and 'ideology_score' columns.

        Returns:
            A numpy array (matrix) where rows represent electors (voters) and columns
            represent candidates (all electors in this version), containing the
            probability of each elector voting for each candidate based on ideology.

        Raises:
            ValueError: If elector_data is empty or missing required columns.
            KeyError: If 'beta_weight' is missing from parameters.
        """
        print("Calculating ideology-based transition probabilities...")

        if elector_data.empty:
            raise ValueError("Cannot calculate probabilities with zero electors.")
        if 'elector_id' not in elector_data.columns or 'ideology_score' not in elector_data.columns:
            raise ValueError("Elector data must contain 'elector_id' and 'ideology_score'.")
        if 'beta_weight' not in self.parameters:
             # Should be caught in __init__, but double-check
             raise KeyError("Model parameter 'beta_weight' is missing.")

        num_electors = len(elector_data)
        # Assume all electors are potential candidates in this version
        num_candidates = num_electors
        beta = self.parameters['beta_weight']

        # Extract ideology scores
        # Ensure index aligns with 0 to num_electors-1 for matrix operations
        ideology_scores = elector_data['ideology_score'].values

        # Calculate pairwise squared differences in ideology
        # Elector i's ideology: ideology_scores[i]
        # Candidate j's ideology: ideology_scores[j]
        # We need a matrix where diff_matrix[i, j] = (ideology_scores[i] - ideology_scores[j])**2
        # Use numpy broadcasting:
        ideology_voters = ideology_scores[:, np.newaxis] # Column vector (num_electors x 1)
        ideology_candidates = ideology_scores[np.newaxis, :] # Row vector (1 x num_candidates)
        diff_sq_matrix = (ideology_voters - ideology_candidates) ** 2

        # Calculate unnormalized probabilities: exp(-beta * diff^2)
        # Higher beta -> stronger preference for closeness
        # Lower beta -> more uniform probabilities
        unnormalized_probs = np.exp(-beta * diff_sq_matrix)

        # Normalize probabilities row-wise (each elector's probabilities must sum to 1)
        row_sums = unnormalized_probs.sum(axis=1, keepdims=True)

        # Avoid division by zero if an elector has zero probability for all candidates
        # (unlikely with exp function unless beta is infinite or diffs are infinite)
        # Replace zero sums with 1 to avoid NaN; the resulting row will be all zeros,
        # which needs careful handling in the simulation step (or assign uniform prob).
        # For now, let's assume row_sums are never zero.
        if np.any(row_sums == 0):
             print("Warning: Zero probability sum encountered for an elector. Assigning uniform probability.")
             # Find rows with zero sum
             zero_sum_rows = np.where(row_sums == 0)[0]
             # Assign uniform probability for these rows
             uniform_prob = 1.0 / num_candidates
             unnormalized_probs[zero_sum_rows, :] = uniform_prob
             # Recalculate row sums for these specific rows (which will now be 1.0)
             row_sums[zero_sum_rows] = 1.0

        probabilities = unnormalized_probs / row_sums

        # Verify row sums are close to 1 (within floating point tolerance)
        # assert np.allclose(probabilities.sum(axis=1), 1.0),
        #       "Probabilities do not sum to 1 for all electors."

        return probabilities
