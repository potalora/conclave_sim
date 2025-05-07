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
        elector_data: DataFrame containing elector profiles.
                      Must have 'elector_id' set as the index and include
                      an 'ideology_score' column.
        model_parameters: Dictionary of parameters for the TransitionModel.
        verbose: If True, print detailed round-by-round progress.

    Returns:
        A tuple containing:
            - results_df: DataFrame summarizing the outcome of each simulation
                          (e.g., winner_id, rounds_taken).
            - aggregate_stats: Dictionary containing aggregated statistics across
                               all simulations (e.g., win frequencies, avg rounds).

    Raises:
        ValueError: If elector_data is empty or lacks the 'ideology_score' column.
    """
    print(f"Starting Monte Carlo simulation with {num_simulations} runs...")
    start_time = time.time()

    if elector_data.empty:
        raise ValueError("Elector data cannot be empty.")
    required_cols = ['ideology_score'] # Ensure ideology score is present for the model
    for col in required_cols:
        if col not in elector_data.columns:
            raise ValueError(f"Elector data must contain an '{col}' column.")

    num_electors = len(elector_data)
    required_votes = int(np.ceil(num_electors * REQUIRED_MAJORITY_FRACTION))
    # AI: Get elector IDs from the index now
    elector_ids_list = elector_data.index.tolist() # Use index directly

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
        # AI: Get candidate IDs from index as well
        candidate_ids = elector_data.index.tolist() # Use index directly

        while round_num <= MAX_ROUNDS and not winner_found:
            if verbose:
                print(f"  Round {round_num}...")

            # 1. Calculate Transition Probabilities (Elector -> Candidate)
            # Pass the DataFrame with elector_id as index
            probabilities = transition_model.calculate_transition_probabilities(
                elector_data=elector_data, # Now has elector_id as index
                current_votes=previous_votes_dict
            )
            # Ensure probabilities matrix dimensions match N x N
            if probabilities.shape != (num_electors, num_electors):
                 raise RuntimeError(
                     f"Probability matrix shape mismatch. Expected ({num_electors}, {num_electors}), "
                     f"got {probabilities.shape}"
                 )

            # 2. Simulate Voting
            votes_cast = [] # Stores the candidate ID each elector voted for in *this* round
            # AI: Iterate using index positions (0 to N-1) which map to rows/cols of prob matrix
            for elector_idx in range(num_electors):
                # Get the probability distribution for this elector (row index)
                prob_dist = probabilities[elector_idx, :]
                # Normalize probabilities due to potential floating point inaccuracies
                prob_dist_sum = prob_dist.sum()
                if prob_dist_sum > 0:
                    if not np.isclose(prob_dist_sum, 1.0):
                        # Re-normalize if necessary (should be close to 1 after model step)
                        # print(f"Warning: Normalizing prob dist for elector idx {elector_idx}. Sum was {prob_dist_sum}")
                        prob_dist /= prob_dist_sum
                else:
                    # Handle case where all probabilities are zero (should be rare after model fixes)
                    voter_id = elector_ids_list[elector_idx]
                    print(f"Warning: Zero probability sum for elector {voter_id} (idx {elector_idx}). Assigning uniform vote.")
                    prob_dist = np.ones(len(candidate_ids)) / len(candidate_ids)

                # Use p=prob_dist which MUST sum to 1
                # np.random.choice selects based on index, so map back to candidate ID
                chosen_candidate_idx = np.random.choice(len(candidate_ids), p=prob_dist)
                voted_for_candidate_id = candidate_ids[chosen_candidate_idx] # Map index back to ID
                votes_cast.append(voted_for_candidate_id)

            # Create dictionary mapping elector_id -> voted_candidate_id for this round
            # AI: Map using elector IDs from the index list
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
    print("Results DataFrame generated.")
    # print(results_df) # Can be large

    # --- Aggregate Results ---
    print("\nCalculating aggregate statistics...")
    aggregate_stats = {}
    if not results_df.empty:
        successful_sims = results_df[results_df['status'] == 'Success']
        aggregate_stats['num_simulations_run'] = num_simulations
        aggregate_stats['num_successful'] = len(successful_sims)
        aggregate_stats['success_rate'] = aggregate_stats['num_successful'] / num_simulations
        aggregate_stats['num_max_rounds_reached'] = len(results_df[results_df['status'] == 'Max Rounds Reached'])

        if not successful_sims.empty:
            aggregate_stats['avg_rounds_success'] = successful_sims['rounds_taken'].mean()
            aggregate_stats['median_rounds_success'] = successful_sims['rounds_taken'].median()
            aggregate_stats['min_rounds_success'] = successful_sims['rounds_taken'].min()
            aggregate_stats['max_rounds_success'] = successful_sims['rounds_taken'].max()
            # Calculate win frequency
            win_counts = successful_sims['winner_id'].value_counts()
            win_freq = (win_counts / aggregate_stats['num_successful']).round(4) # Normalize by successful sims
            aggregate_stats['win_counts'] = win_counts.to_dict()
            aggregate_stats['win_frequency'] = win_freq.to_dict()
            aggregate_stats['most_frequent_winner_id'] = win_counts.index[0] if not win_counts.empty else None
            aggregate_stats['most_frequent_winner_count'] = int(win_counts.iloc[0]) if not win_counts.empty else 0
            aggregate_stats['most_frequent_winner_freq'] = win_freq.iloc[0] if not win_freq.empty else 0.0
        else:
            aggregate_stats['avg_rounds_success'] = None
            aggregate_stats['median_rounds_success'] = None
            aggregate_stats['min_rounds_success'] = None
            aggregate_stats['max_rounds_success'] = None
            aggregate_stats['win_counts'] = {}
            aggregate_stats['win_frequency'] = {}
            aggregate_stats['most_frequent_winner_id'] = None
            aggregate_stats['most_frequent_winner_count'] = 0
            aggregate_stats['most_frequent_winner_freq'] = 0.0

    else:
        print("No simulation results to aggregate.")

    return results_df, aggregate_stats


# Example Usage (Illustrative - Needs actual data loading and model params)
if __name__ == '__main__':
    import argparse
    import json
    from pathlib import Path
    import sys # For exit
    import os # For checking GOOGLE_API_KEY if model needs it

    # Ensure correct imports when run as module
    from .utils import setup_logging # Assuming setup_logging is useful here too

    # --- Setup Argument Parser ---
    parser = argparse.ArgumentParser(description="Run Conclave Monte Carlo Simulation.")
    parser.add_argument(
        "-n", "--num-simulations",
        type=int,
        default=1000,
        help="Number of simulation iterations to run."
    )
    parser.add_argument(
        "--input-data",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "merged_electors.csv",
        help="Path to the merged elector data CSV file."
    )
    parser.add_argument(
        "--output-results",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "simulation_results.csv",
        help="Path to save the detailed simulation results CSV."
    )
    parser.add_argument(
        "--output-stats",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "simulation_stats.json",
        help="Path to save the aggregate simulation statistics JSON."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed round-by-round output."
    )
    # TODO: Add arguments for model parameters if desired

    args = parser.parse_args()

    # --- Setup Logging ---
    log_level = 'DEBUG' if args.verbose else 'INFO'
    log = setup_logging(level=log_level)

    # --- Load Data ---
    try:
        log.info(f"Loading elector data from: {args.input_data}")
        if not args.input_data.exists():
            raise FileNotFoundError(f"Input file not found: {args.input_data}")
        electors_df = pd.read_csv(args.input_data)

        if 'elector_id' not in electors_df.columns:
            raise ValueError(f"'elector_id' column not found in {args.input_data}.")

        # Convert elector_id to string before setting index
        electors_df['elector_id'] = electors_df['elector_id'].astype(str)
        electors_df.set_index('elector_id', inplace=True)
        log.info(f"Successfully loaded {len(electors_df)} electors.")

        # Check for ideology score (required by run_monte_carlo_simulation)
        if 'ideology_score' not in electors_df.columns:
            log.error("'ideology_score' column not found in input data. Please re-run the ingestion script.")
            raise ValueError("'ideology_score' column missing from input data.")

        # Ensure the ideology score column is numeric and handle potential NaNs
        try:
            electors_df['ideology_score'] = pd.to_numeric(electors_df['ideology_score'])
            if electors_df['ideology_score'].isnull().any():
                log.warning("NaN values found in 'ideology_score'. Imputing with median.")
                median_score = electors_df['ideology_score'].median()
                if pd.isna(median_score):
                    log.warning("Median ideology score is NaN. Imputing with 0.5 (midpoint of expected 0-1 range).")
                    median_score = 0.5
                electors_df['ideology_score'].fillna(median_score, inplace=True)
            log.info("Validated 'ideology_score' column (numeric, imputed NaNs).")
        except ValueError as e:
            raise ValueError(f"Could not convert 'ideology_score' column to numeric: {e}")

    except FileNotFoundError as e:
        log.error(f"Error: {e}")
        sys.exit(1)
    except ValueError as e:
        log.error(f"Error loading or processing data: {e}")
        sys.exit(1)
    except Exception as e:
        log.error(f"An unexpected error occurred loading data: {e}", exc_info=True)
        sys.exit(1)

    # --- Define Model Parameters (Placeholder) ---
    model_params = {
        'beta_weight': 0.3,       # Sensitivity to ideology difference (required by model)
        'stickiness_factor': 0.5  # Tendency to repeat previous vote (required by model)
        # TODO: Add other parameters if the model is extended (e.g., regional influence, randomness)
    }
    log.info(f"Using model parameters: {model_params}")

    # --- Run Simulation ---
    try:
        results_df, aggregate_stats = run_monte_carlo_simulation(
            num_simulations=args.num_simulations,
            elector_data=electors_df,
            model_parameters=model_params,
            verbose=args.verbose
        )
    except Exception as e:
        log.error(f"An error occurred during the simulation: {e}", exc_info=True)
        sys.exit(1)

    # --- Save Results ---
    try:
        # Ensure output directories exist
        args.output_results.parent.mkdir(parents=True, exist_ok=True)
        args.output_stats.parent.mkdir(parents=True, exist_ok=True)

        log.info(f"Saving simulation results ({len(results_df)} rows) to: {args.output_results}")
        results_df.to_csv(args.output_results, index=False)

        log.info(f"Saving aggregate stats to: {args.output_stats}")

        # Convert numpy types in stats to standard types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(i) for i in obj]
            # Handle potential None or other non-serializable types gracefully
            if pd.isna(obj):
                 return None
            return obj

        serializable_stats = convert_numpy_types(aggregate_stats)

        with open(args.output_stats, 'w', encoding='utf-8') as f:
            json.dump(serializable_stats, f, indent=4)

        log.info("\n--- Simulation Finished ---")
        # Print some key stats from the serializable dictionary
        success_rate = serializable_stats.get('success_rate', None)
        avg_rounds = serializable_stats.get('avg_rounds_success', None)
        winner_id = serializable_stats.get('most_frequent_winner_id', 'N/A')
        winner_freq = serializable_stats.get('most_frequent_winner_freq', 0)

        print(f"Success Rate: {success_rate:.2%}" if success_rate is not None else "Success Rate: N/A")
        print(f"Average Rounds (Successful): {avg_rounds:.2f}" if avg_rounds is not None else "Average Rounds (Successful): N/A")
        print(f"Most Frequent Winner: Elector {winner_id} ({winner_freq:.2%})")

    except Exception as e:
        log.error(f"An error occurred saving results: {e}", exc_info=True)
        sys.exit(1)
