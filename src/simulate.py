import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
import time

# AI: Import necessary components from other modules
from .model import TransitionModel
# from .ingest import load_elector_data # Keep commented until elector_data is used differently

# Constants
MAX_ROUNDS = 100  # Maximum rounds per simulation before declaring no winner
REQUIRED_MAJORITY_FRACTION = 2 / 3

def run_monte_carlo_simulation(
    num_simulations: int,
    elector_data: pd.DataFrame,
    model_parameters: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Runs the Monte Carlo simulation for the conclave.

    Args:
        num_simulations: The number of simulation iterations to run.
        elector_data: DataFrame containing elector profiles (e.g., id, ideology).
                      Must include a unique 'elector_id' column.
        model_parameters: Dictionary of parameters for the TransitionModel.

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
    print(f"Simulating with {num_electors} electors. Required votes for majority: {required_votes}")

    # Initialize the transition model
    transition_model = TransitionModel(parameters=model_parameters)
    # print(f"TransitionModel initialized with parameters: {model_parameters}") # Removed redundant print

    # Prepare results storage
    simulation_results: List[Dict[str, Any]] = [] # More robust than appending to df

    # --- Main Simulation Loop ---
    for sim_num in range(1, num_simulations + 1):
        print(f"\n--- Simulation Run {sim_num} / {num_simulations} ---")
        round_num = 1
        winner_found = False
        winner_id = None
        current_votes_summary = pd.DataFrame() # Placeholder for potential future model input
        # In this basic model, candidates are all electors
        # Use elector_id as the candidate identifier
        candidate_ids = elector_data['elector_id'].tolist()

        while round_num <= MAX_ROUNDS and not winner_found:
            print(f"  Round {round_num}...")

            # 1. Calculate Transition Probabilities (Elector -> Candidate)
            # Returns a matrix (num_electors x num_candidates)
            # In this basic model, candidates are all electors
            probabilities = transition_model.calculate_transition_probabilities(
                current_votes=current_votes_summary, elector_data=elector_data
            )
            # Ensure probabilities sum to 1 for each elector (row-wise)
            # probabilities /= probabilities.sum(axis=1, keepdims=True)

            # 2. Simulate Voting
            votes_cast = []
            for elector_idx in range(num_electors):
                # Choose a candidate based on the probability distribution for this elector
                # np.random.choice requires probabilities to sum to 1
                prob_dist = probabilities[elector_idx]
                # Normalize probabilities due to potential floating point inaccuracies
                prob_dist /= prob_dist.sum()
                voted_for = np.random.choice(candidate_ids, p=prob_dist)
                votes_cast.append(voted_for)

            # 3. Tally Votes
            vote_counts = pd.Series(votes_cast).value_counts()
            # print(f"    Vote Counts: {vote_counts.to_dict()}") # Can be verbose

            # 4. Check for Winner
            if not vote_counts.empty:
                top_candidate = vote_counts.index[0]
                top_votes = vote_counts.iloc[0]
                if top_votes >= required_votes:
                    winner_id = top_candidate
                    winner_found = True
                    print(f"    Winner found! Candidate {winner_id} received {top_votes} votes.")
                else:
                    print(f"    No winner yet. Top candidate {top_candidate} has {top_votes} votes (need {required_votes}).")
            else:
                 print(f"    No votes cast in round {round_num}.") # Should not happen with current logic

            # 5. Update State (prepare for next round if no winner)
            if not winner_found:
                # Placeholder for updating current_votes_summary if model needs it
                round_num += 1
            # If winner found, loop will terminate

        # Record simulation outcome
        if winner_found:
             result = {'simulation_id': sim_num, 'winner_id': winner_id, 'rounds_taken': round_num, 'status': 'Success'}
             print(f"--- Simulation Run {sim_num} finished: Winner={winner_id}, Rounds={round_num} ---")
        else:
             result = {'simulation_id': sim_num, 'winner_id': None, 'rounds_taken': MAX_ROUNDS, 'status': 'Max Rounds Reached'}
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

    # raise NotImplementedError("Core simulation logic/aggregation not yet implemented.")

    return results_df, aggregate_stats
