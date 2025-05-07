import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
import time
from src.model import TransitionModel #, PreferenceModel # PreferenceModel not directly used here now
from src.utils import setup_logging # Assuming setup_logging is useful here too
from src.viz import plot_ideology_distribution
import logging
import argparse
import json
from pathlib import Path
import sys
import os
import multiprocessing
from multiprocessing import cpu_count
from collections import Counter

# Constants
MAX_ROUNDS_DEFAULT = 100 # Default max rounds if not specified by run-off
WINNER_THRESHOLD = 2/3

# AI: Define default bonus values centrally if they might be used in multiple places
# For now, direct use in argparse is fine.
DEFAULT_REGIONAL_AFFINITY_BONUS = 0.1
DEFAULT_PAPABILE_CANDIDATE_BONUS = 0.1

class SimulationIdCounter:
    def __init__(self, start_id: int = 0):
        self.current_id = start_id

    def __iter__(self):
        return self

    def __next__(self) -> int:
        res = self.current_id
        self.current_id += 1
        return res

def _check_winner(votes: Dict[str, int], total_electors: int, threshold: float) -> Tuple[Optional[str], float]:
    """Check if there's a winner based on the votes."""
    if not votes:
        return None, 0.0
    
    winner = max(votes, key=votes.get)
    winner_votes = votes[winner]
    winner_fraction = winner_votes / total_electors
    
    if winner_fraction >= threshold:
        return winner, winner_fraction
    else:
        return None, 0.0


def _run_single_simulation_worker(
    elector_data: pd.DataFrame,
    model_params: Dict[str, Any],
    max_rounds: int,
    sim_id: int,
    winner_threshold: float,
    runoff_threshold_rounds: int,
    runoff_candidate_count: int,
    verbose_logging: bool = False
) -> Tuple[Optional[str], int, int, Dict[str, Any]]:
    """Worker function to run a single simulation instance."""
    log = logging.getLogger(f"src.simulate.worker.{sim_id}")
    worker_log_level = logging.DEBUG if verbose_logging else logging.INFO
    log.setLevel(worker_log_level)
    if not log.handlers:
        ch = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        log.addHandler(ch)
        log.propagate = False 

    log.debug(f"[Sim {sim_id}] Initializing TransitionModel with params: {model_params}")
    try:
        model = TransitionModel(elector_data.copy(), **model_params)
    except ValueError as e:
        log.error(f"[Sim {sim_id}] Error initializing TransitionModel: {e}")
        return None, 0, sim_id, {'error': str(e)}

    total_electors = len(elector_data)
    current_round = 0
    votes_history: List[Dict[str, Any]] = []
    winner: Optional[str] = None
    final_standings: Dict[str, float] = {}
    election_status = "Ongoing"
    current_active_candidates: Optional[List[str]] = None # Initially all candidates are active

    log.debug(f"[Sim {sim_id}] Starting simulation loop.")
    while current_round < max_rounds:
        current_round += 1
        log.debug(f"[Sim {sim_id}] Round {current_round}")

        # Get current votes from the last round for stickiness, if available
        current_votes_dict = votes_history[-1]['votes'] if votes_history else None

        # AI: Unpack tuple from calculate_transition_probabilities
        prob_matrix_tuple_result = model.calculate_transition_probabilities(
            elector_data, 
            current_votes_dict, 
            active_candidates=current_active_candidates
        )
        # Ensure it's a tuple and unpack
        if not isinstance(prob_matrix_tuple_result, tuple) or len(prob_matrix_tuple_result) != 2:
            log.error(f"[Sim {sim_id}] calculate_transition_probabilities did not return a tuple of (matrix, names). Got: {type(prob_matrix_tuple_result)}")
            # Fallback or raise error - for now, let's assume it might be old version for a moment, or error out
            # This path should ideally not be hit if model.py is updated.
            election_status = "Error: Model output format unexpected"
            break 
        
        prob_matrix, current_effective_candidates = prob_matrix_tuple_result

        if not isinstance(prob_matrix, np.ndarray):
            log.error(f"[Sim {sim_id}] prob_matrix is not a numpy array (Round {current_round}). Type: {type(prob_matrix)}")
            election_status = "Error: Invalid prob_matrix type"
            break
        
        if not current_effective_candidates:
            log.error(f"[Sim {sim_id}] No effective candidates returned from model (Round {current_round}).")
            election_status = "Error: No effective candidates"
            break

        # Determine votes for this round
        # Each elector votes based on the probability matrix
        # The prob_matrix columns correspond to current_effective_candidates
        round_votes: Dict[str, int] = Counter()
        num_effective_candidates_for_choice = prob_matrix.shape[1]

        if num_effective_candidates_for_choice == 0:
            log.warning(f"[Sim {sim_id}] No candidates to choose from in round {current_round} based on prob_matrix shape. Skipping vote.")
            # This might happen if all candidates were filtered out, leading to an empty prob_matrix column-wise
            # We should record this situation and potentially end the simulation for this run.
            # For now, let's assume votes are empty and see how winner check handles it.

        else:
            for i in range(model.num_electors):
                try:
                    # Ensure probabilities sum to 1 for np.random.choice
                    p_row = prob_matrix[i, :]
                    if not np.isclose(np.sum(p_row), 1.0):
                        # Attempt to re-normalize if not close to 1, could be due to floating point issues or earlier problem
                        log.warning(f"[Sim {sim_id}] Elector {i} probabilities sum to {np.sum(p_row)} (Round {current_round}). Re-normalizing.")
                        p_row = p_row / (np.sum(p_row) + 1e-9) # Add epsilon to avoid division by zero
                    
                    chosen_candidate_idx = np.random.choice(num_effective_candidates_for_choice, p=p_row)
                    # AI: Use current_effective_candidates to get the name
                    candidate_name = current_effective_candidates[chosen_candidate_idx]
                    round_votes[candidate_name] += 1
                except ValueError as e:
                    log.error(f"[Sim {sim_id}] Error during voting for elector {i}, round {current_round}: {e}. Probabilities: {prob_matrix[i, :]}. Sum: {np.sum(prob_matrix[i, :])}")
                    # Decide how to handle: skip vote, assign random, or halt simulation for this run
                    # For now, this elector effectively abstains this round if choice fails
                    continue 

        # Record votes for this round
        current_total_votes = sum(round_votes.values())
        votes_history.append({'votes': round_votes, 'total_votes': current_total_votes})

        winner, _ = _check_winner(round_votes, total_electors, winner_threshold)

        if winner:
            final_standings[winner] = 1.0 # Mark winner distinctly
            election_status = "Winner"
            log.info(f"[Sim {sim_id}] Winner found in round {current_round}: {winner}")
            break

        # Check for run-off condition if no winner yet and run-off not yet active
        if not winner and not current_active_candidates and current_round >= runoff_threshold_rounds:
            if runoff_threshold_rounds > 0 and runoff_candidate_count >= 2: # Ensure run-off is meaningful
                # Sort candidates by votes in the current round
                sorted_candidates_by_vote = sorted(round_votes.items(), key=lambda item: item[1], reverse=True)
                # AI: Extract only the candidate IDs (strings) for current_active_candidates
                current_active_candidates = [c[0] for c in sorted_candidates_by_vote[:runoff_candidate_count]]
                
                log.info(f"[Sim {sim_id}] Round {current_round}: No winner, initiating run-off with top {runoff_candidate_count} candidates.")
                
                if len(current_active_candidates) < runoff_candidate_count and len(model.candidate_names) > len(current_active_candidates):
                    log.warning(f"[Sim {sim_id}] Run-off initiated with {len(current_active_candidates)} candidates, less than requested {runoff_candidate_count}.")
                
                log.info(f"[Sim {sim_id}] Run-off candidates: {current_active_candidates}")

    else: # Loop finished without a winner (max_rounds reached)
        log.info(f"[Sim {sim_id}] Max rounds ({max_rounds}) reached. No winner decided through threshold.")
        # TODO: Consider handling this case differently, e.g., by declaring a winner based on final standings or by extending the simulation

    return winner, current_round, sim_id, votes_history


def run_monte_carlo_simulation(
    num_simulations: int,
    elector_data_full: pd.DataFrame,
    model_params: Dict[str, Any],
    max_rounds: int = MAX_ROUNDS_DEFAULT,
    num_cores: Optional[int] = None,
    winner_threshold: float = WINNER_THRESHOLD,
    report_interval: int = 10,
    runoff_threshold_rounds: int = 0,
    runoff_candidate_count: int = 0,
    verbose_logging: bool = False
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Runs a Monte Carlo simulation of the conclave voting process using multiprocessing."""
    log = logging.getLogger(__name__)
    log.info(f"Starting Monte Carlo simulation with {num_simulations} runs.")
    log.info(f"Max rounds per simulation: {max_rounds}, Winner threshold: {winner_threshold:.2f}")
    log.info(f"Run-off settings: Threshold Rounds={runoff_threshold_rounds}, Candidate Count={runoff_candidate_count}")
    log.info(f"Regional Affinity Bonus: {model_params.get('regional_affinity_bonus', DEFAULT_REGIONAL_AFFINITY_BONUS)}, Papabile Candidate Bonus: {model_params.get('papabile_candidate_bonus', DEFAULT_PAPABILE_CANDIDATE_BONUS)}") # New log

    # Prepare arguments for each worker
    # Ensure elector_data has 'elector_id' as index if not already
    if elector_data_full.index.name != 'elector_id':
        if 'elector_id' in elector_data_full.columns:
            log.info("Setting 'elector_id' as index for elector_data_full.")
            elector_data_full = elector_data_full.set_index('elector_id', drop=False) # Keep column if needed elsewhere
        else:
            log.error("'elector_id' not found as index or column in elector_data_full. Critical for model.")
            raise ValueError("'elector_id' must be present and set as index for the simulation.")

    # Check for required columns for new features in the DataFrame
    required_df_cols_for_model = ['ideology_score', 'region', 'is_papabile'] 
    for col in required_df_cols_for_model:
        if col not in elector_data_full.columns:
            log.error(f"Required column '{col}' not found in elector_data_full. This column is needed for the model features.")
            raise ValueError(f"Missing required DataFrame column for model: {col}")

    num_cpus_to_use = num_cores if num_cores and num_cores > 0 else multiprocessing.cpu_count() // 2
    if num_cpus_to_use == 0: num_cpus_to_use = 1
    log.info(f"Using {num_cpus_to_use} cores for simulation.")
    sim_id_counter = SimulationIdCounter()

    worker_args = [
        (elector_data_full.copy(), model_params, max_rounds, next(sim_id_counter), winner_threshold, runoff_threshold_rounds, runoff_candidate_count, verbose_logging)
        for _ in range(num_simulations)
    ]

    results_list: List[Tuple[Optional[str], int, int, Dict[str, Any]]] = []
    start_time = time.time()

    if num_cpus_to_use > 1:
        with multiprocessing.Pool(processes=num_cpus_to_use) as pool:
            results_list = pool.starmap(_run_single_simulation_worker, worker_args)
    else:
        for args in worker_args:
            results_list.append(_run_single_simulation_worker(*args))

    end_time = time.time()
    log.info(f"Simulation time: {end_time - start_time:.2f} seconds")

    results_df = pd.DataFrame(results_list, columns=['winner', 'rounds', 'sim_id', 'history'])
    
    # Calculate aggregate statistics
    aggregate_stats = {}
    aggregate_stats['total_simulations'] = num_simulations
    
    # Ensure 'winner' column is appropriate for checking non-null (e.g., strings, not objects that might evaluate to True)
    # Pandas isna() is good for this, as it handles None, np.nan, etc.
    successful_sims_df = results_df[results_df['winner'].notna() & (results_df['winner'] != '')]
    
    aggregate_stats['successful_simulations'] = len(successful_sims_df)
    
    if num_simulations > 0:
        aggregate_stats['success_rate'] = aggregate_stats['successful_simulations'] / num_simulations
    else:
        aggregate_stats['success_rate'] = 0.0
        
    if aggregate_stats['successful_simulations'] > 0:
        aggregate_stats['average_rounds_successful'] = successful_sims_df['rounds'].mean()
        aggregate_stats['min_rounds_successful'] = successful_sims_df['rounds'].min()
        aggregate_stats['max_rounds_successful'] = successful_sims_df['rounds'].max()
        aggregate_stats['median_rounds_successful'] = successful_sims_df['rounds'].median()
        
        winner_counts = successful_sims_df['winner'].value_counts()
        aggregate_stats['winner_counts'] = winner_counts.to_dict()
        
        if not winner_counts.empty:
            most_frequent_winner_id = winner_counts.index[0]
            most_frequent_winner_wins = winner_counts.iloc[0]
            aggregate_stats['most_frequent_winner'] = str(most_frequent_winner_id) # Ensure it's string for JSON
            aggregate_stats['most_frequent_winner_count'] = int(most_frequent_winner_wins)
            if num_simulations > 0:
                aggregate_stats['most_frequent_winner_percentage'] = most_frequent_winner_wins / num_simulations
            else:
                aggregate_stats['most_frequent_winner_percentage'] = 0.0
        else:
            aggregate_stats['most_frequent_winner'] = None
            aggregate_stats['most_frequent_winner_count'] = 0
            aggregate_stats['most_frequent_winner_percentage'] = 0.0
    else:
        aggregate_stats['average_rounds_successful'] = None
        aggregate_stats['min_rounds_successful'] = None
        aggregate_stats['max_rounds_successful'] = None
        aggregate_stats['median_rounds_successful'] = None
        aggregate_stats['winner_counts'] = {}
        aggregate_stats['most_frequent_winner'] = None
        aggregate_stats['most_frequent_winner_count'] = 0
        aggregate_stats['most_frequent_winner_percentage'] = 0.0
        
    aggregate_stats['average_rounds_all'] = results_df['rounds'].mean() if not results_df.empty else None

    log.info(f"Calculated aggregate stats: {aggregate_stats}")
    return results_df, aggregate_stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Conclave Monte Carlo Simulation.")
    parser.add_argument(
        "-n", "--num-simulations",
        type=int,
        default=1000,
        help="Number of simulation iterations to run."
    )
    parser.add_argument(
        "--electors-file",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "merged_electors.csv",
        help="Path to the CSV file containing elector data.",
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
        help="Enable DEBUG level logging. If not set, INFO level is used."
    )
    parser.add_argument(
        "--beta-weight",
        type=float,
        default=0.3,
        help="Beta weight for the TransitionModel (sensitivity to ideology difference)."
    )
    parser.add_argument(
        "--stickiness-factor",
        type=float,
        default=0.5,
        help="Stickiness factor for the TransitionModel (tendency to repeat previous vote)."
    )
    parser.add_argument(
        "--bandwagon-strength",
        type=float,
        default=0.0,
        help="Strength of the bandwagon effect (>=0). Multiplies vote probabilities by (1 + strength * prev_vote_share)."
    )
    parser.add_argument(
        "--runoff-threshold-rounds",
        type=int,
        default=5,
        help="Number of initial rounds before a run-off can be triggered if no winner (must be > 0). Set to 0 to disable."
    )
    parser.add_argument(
        "--runoff-candidate-count",
        type=int,
        default=2,
        help="Number of top candidates to advance to the run-off stage (must be >= 2). Set to 0 to disable."
    )
    parser.add_argument(
        "--plot-ideology-dist",
        action="store_true",
        help="If set, plots and saves the ideology score distribution to data/plots/."
    )
    parser.add_argument(
        "--num-cores",
        type=int,
        default=None,
        help="Number of CPU cores to use for parallel simulation. Defaults to half available cores."
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=MAX_ROUNDS_DEFAULT,
        help=f'Maximum number of voting rounds per simulation. Default {MAX_ROUNDS_DEFAULT}.'
    )
    parser.add_argument(
        "--report-interval",
        type=int,
        default=10,
        help="Interval for reporting simulation progress during verbose runs. Default 10."
    )
    # AI: Add new arguments for bonuses
    parser.add_argument("--regional-bonus", type=float, default=DEFAULT_REGIONAL_AFFINITY_BONUS, 
                        help=f"Additive bonus for regional affinity (default: {DEFAULT_REGIONAL_AFFINITY_BONUS})")
    parser.add_argument("--papabile-bonus", type=float, default=DEFAULT_PAPABILE_CANDIDATE_BONUS, 
                        help=f"Additive bonus for papabile candidates (default: {DEFAULT_PAPABILE_CANDIDATE_BONUS})")

    args = parser.parse_args()

    main_log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(main_log_level) 

    log = logging.getLogger(__name__) 

    if args.verbose:
        log.debug("Verbose logging enabled for main process.") 

    try:
        log.info(f"Loading elector data from: {args.electors_file}")
        if not args.electors_file.exists():
            raise FileNotFoundError(f"Input file not found: {args.electors_file}")
        electors_df = pd.read_csv(args.electors_file)

        if 'elector_id' not in electors_df.columns:
            raise ValueError(f"'elector_id' column not found in {args.electors_file}.")

        electors_df['elector_id'] = electors_df['elector_id'].astype(str)
        electors_df.set_index('elector_id', inplace=True)
        log.info(f"Successfully loaded {len(electors_df)} electors.")

        if 'ideology_score' not in electors_df.columns:
            log.error("'ideology_score' column not found in input data. Please re-run the ingestion script.")
            raise ValueError("'ideology_score' column missing from input data.")

        electors_df['ideology_score'] = pd.to_numeric(electors_df['ideology_score'])
        if electors_df['ideology_score'].isnull().any():
            log.warning("NaN values found in 'ideology_score'. Imputing with median.")
            median_score = electors_df['ideology_score'].median()
            if pd.isna(median_score):
                log.warning("Median ideology score is NaN. Imputing with 0.5 (midpoint of expected 0-1 range).")
                median_score = 0.5
            electors_df['ideology_score'].fillna(median_score, inplace=True)
        log.info("Validated 'ideology_score' column (numeric, imputed NaNs).")

        if args.plot_ideology_dist:
            plot_output_path = Path(__file__).resolve().parent.parent / "data" / "plots" / "ideology_distribution.png"
            plot_ideology_distribution(electors_df['ideology_score'], plot_output_path)

    except FileNotFoundError as e:
        log.error(f"Error: {e}")
        sys.exit(1)
    except ValueError as e:
        log.error(f"Error loading or processing data: {e}")
        sys.exit(1)
    except Exception as e:
        log.error(f"An unexpected error occurred loading data: {e}", exc_info=True)
        sys.exit(1)

    model_params = {
        'beta_weight': args.beta_weight,
        'stickiness_factor': args.stickiness_factor,
        'bandwagon_strength': args.bandwagon_strength,
        'regional_affinity_bonus': args.regional_bonus,
        'papabile_candidate_bonus': args.papabile_bonus
    }
    log.info(f"Using model parameters: {model_params}")

    try:
        results_df, aggregate_stats = run_monte_carlo_simulation(
            num_simulations=args.num_simulations,
            elector_data_full=electors_df,
            model_params=model_params,
            max_rounds=args.max_rounds,
            num_cores=args.num_cores,
            winner_threshold=WINNER_THRESHOLD,
            report_interval=args.report_interval,
            runoff_threshold_rounds=args.runoff_threshold_rounds,
            runoff_candidate_count=args.runoff_candidate_count,
            verbose_logging=args.verbose
        )

        args.output_results.parent.mkdir(parents=True, exist_ok=True)
        args.output_stats.parent.mkdir(parents=True, exist_ok=True)

        log.info(f"Saving simulation results ({len(results_df)} rows) to: {args.output_results}")
        results_df.to_csv(args.output_results, index=False)

        log.info(f"Saving aggregate stats to: {args.output_stats}")

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
            if pd.isna(obj):
                 return None
            return obj

        serializable_stats = convert_numpy_types(aggregate_stats)

        with open(args.output_stats, 'w', encoding='utf-8') as f:
            json.dump(serializable_stats, f, indent=4)

        log.info("\n--- Simulation Finished ---")
        success_rate = serializable_stats.get('success_rate', None)
        avg_rounds = serializable_stats.get('average_rounds_successful', None)
        winner_id = serializable_stats.get('most_frequent_winner', 'N/A')
        winner_freq_percentage = serializable_stats.get('most_frequent_winner_percentage', 0)

        print(f"Success Rate: {success_rate:.2%}" if success_rate is not None else "Success Rate: N/A")
        print(f"Average Rounds (Successful): {avg_rounds:.2f}" if avg_rounds is not None else "Average Rounds (Successful): N/A")
        print(f"Most Frequent Winner: Elector {winner_id} ({winner_freq_percentage:.2%})")

    except Exception as e:
        log.error(f"An error occurred saving results: {e}", exc_info=True)
        sys.exit(1)
