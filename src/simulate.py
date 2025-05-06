import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
import time

# AI: Import necessary components from other modules
from .model import TransitionModel

# Constants
MAX_ROUNDS = 100  # Maximum rounds per simulation before declaring no winner
REQUIRED_MAJORITY_FRACTION = 2 / 3

def run_monte_carlo_simulation(
    num_simulations: int,
    elector_data: pd.DataFrame,
    model_parameters: Dict[str, Any],
    verbose: bool = False
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Runs the Monte Carlo simulation for the conclave.

    Args:
        num_simulations: The number of simulation iterations to run.
        elector_data: DataFrame containing elector profiles (e.g., id, ideology).
                      Must include a unique 'elector_id' column.
        model_parameters: Dictionary of parameters for the TransitionModel.
        verbose: If True, print detailed round-by-round progress.

    Returns:
        A tuple containing:
            - results_df: DataFrame summarizing the outcome of each simulation
                          (e.g., winner_id, rounds_taken).
            - aggregate_stats: Dictionary containing aggregated statistics across
                               all simulations (e.g., win frequencies, avg rounds).

    Raises:
        ValueError: If elector_data is empty or lacks 'elector_id'.
    """
    print(f"Starting Monte Carlo simulation with {num_simulations} runs...")
    start_time = time.time()

    if elector_data.empty:
        raise ValueError("Elector data cannot be empty.")
    if 'elector_id' not in elector_data.columns:
        raise ValueError("Elector data must contain an 'elector_id' column.")

    num_electors = len(elector_data)
    required_votes = int(np.ceil(num_electors * REQUIRED_MAJORITY_FRACTION))
    # Ensure elector_ids are readily available for mapping
    elector_ids_list = elector_data['elector_id'].tolist()
    if len(set(elector_ids_list)) != num_electors:
        raise ValueError("Duplicate elector_ids found in elector_data.")

    if verbose:
        print(f"Simulating with {num_electors} electors. Required votes for majority: {required_votes}")

    # Initialize the transition model
    transition_model = TransitionModel(parameters=model_parameters)

    # Prepare results storage
    simulation_results: List[Dict[str, Any]] = []

    # --- Main Simulation Loop ---
    for sim_num in range(1, num_simulations + 1):
        if verbose:
            print(f"\n--- Simulation Run {sim_num} / {num_simulations} ---")
        round_num = 1
        winner_found = False
        winner_id = None
        # Tracks {voter_id: candidate_id} from the *previous* round
        previous_votes_dict: Optional[Dict[Any, Any]] = None

        # Assume all electors are candidates for now
        candidate_ids = elector_ids_list # Use the list directly

        while round_num <= MAX_ROUNDS and not winner_found:
            if verbose:
                print(f"  Round {round_num}...")

            # 1. Calculate Transition Probabilities (Elector -> Candidate)
            # Pass the dictionary of votes from the previous round for stickiness
            probabilities = transition_model.calculate_transition_probabilities(
                elector_data=elector_data,
                current_votes=previous_votes_dict
            )

            # 2. Simulate Voting
            votes_cast = [] # Stores the candidate ID each elector voted for in *this* round
            for elector_idx in range(num_electors):
                # Get the actual ID of the voter
                voter_id = elector_ids_list[elector_idx]
                # Choose a candidate based on the probability distribution for this elector
                prob_dist = probabilities[elector_idx]
                # Normalize probabilities due to potential floating point inaccuracies
                prob_dist_sum = prob_dist.sum()
                if prob_dist_sum > 0:
                    prob_dist /= prob_dist_sum
                else:
                    # Handle case where all probabilities are zero (should be rare)
                    print(f"Warning: Zero probability sum for elector {voter_id} (idx {elector_idx}). Assigning uniform vote.")
                    prob_dist = np.ones(len(candidate_ids)) / len(candidate_ids)

                # Use p=prob_dist which MUST sum to 1
                voted_for_candidate_id = np.random.choice(candidate_ids, p=prob_dist)
                votes_cast.append(voted_for_candidate_id)

            # Create dictionary mapping elector_id -> voted_candidate_id for this round
            current_round_votes_dict = {
                elector_ids_list[i]: votes_cast[i] for i in range(num_electors)
            }

            # 3. Tally Votes
            vote_counts = pd.Series(votes_cast).value_counts()

            # 4. Check for Winner
            if not vote_counts.empty:
                top_candidate_id = vote_counts.index[0]
                top_votes = vote_counts.iloc[0]
                if top_votes >= required_votes:
                    winner_id = top_candidate_id
                    winner_found = True
                    if verbose:
                        print(f"    Winner found! Candidate {winner_id} received {top_votes} votes.")
                else:
                    if verbose:
                        print(f"    No winner yet. Top candidate {top_candidate_id} has {top_votes} votes (need {required_votes}).")
            else:
                 if verbose:
                    print(f"    No votes cast in round {round_num}.")

            # 5. Update State (prepare for next round if no winner)
            if not winner_found:
                # Update the dictionary for the next round's stickiness calculation
                previous_votes_dict = current_round_votes_dict
                round_num += 1
            # If winner found, loop will terminate

        # Record simulation outcome
        if winner_found:
             result = {'simulation_id': sim_num, 'winner_id': winner_id, 'rounds_taken': round_num, 'status': 'Success'}
             if verbose:
                print(f"--- Simulation Run {sim_num} finished: Winner={winner_id}, Rounds={round_num} ---")
        else:
             result = {'simulation_id': sim_num, 'winner_id': None, 'rounds_taken': MAX_ROUNDS, 'status': 'Max Rounds Reached'}
             if verbose:
                print(f"--- Simulation Run {sim_num} finished: No winner after {MAX_ROUNDS} rounds ---")

        simulation_results.append(result)

    print(f"\n--- Simulation Complete ({num_simulations} runs) ---")
    end_time = time.time()
    print(f"Total simulation time: {end_time - start_time:.2f} seconds")

    # Convert results list to DataFrame
    results_df = pd.DataFrame(simulation_results)
    print("Results DataFrame generated:")
    # print(results_df) # Can be large

    # --- Aggregate Results --- Placeholder, needs refinement
    print("\nCalculating aggregate statistics...")
    aggregate_stats = {}
    if not results_df.empty:
        successful_sims = results_df[results_df['status'] == 'Success']
        if not successful_sims.empty:
            aggregate_stats['avg_rounds_success'] = successful_sims['rounds_taken'].mean()
            aggregate_stats['median_rounds_success'] = successful_sims['rounds_taken'].median()
            # Calculate win frequency
            win_counts = successful_sims['winner_id'].value_counts(normalize=True)
            aggregate_stats['win_frequency'] = win_counts.to_dict()
            aggregate_stats['most_frequent_winner'] = win_counts.index[0] if not win_counts.empty else None
        else:
            aggregate_stats['avg_rounds_success'] = None
            aggregate_stats['median_rounds_success'] = None
            aggregate_stats['win_frequency'] = {}
            aggregate_stats['most_frequent_winner'] = None

        aggregate_stats['num_max_rounds_reached'] = len(results_df[results_df['status'] == 'Max Rounds Reached'])
        aggregate_stats['success_rate'] = len(successful_sims) / num_simulations
    else:
        print("No simulation results to aggregate.")

    return results_df, aggregate_stats
