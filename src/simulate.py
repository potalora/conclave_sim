import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import Counter # Added missing import
import argparse
import json
import logging
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from src.model import TransitionModel
from src.ingest import ElectorDataIngester

# Configure logging
logging.basicConfig(level=logging.INFO, # Changed back from logging.DEBUG
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)
worker_log = logging.getLogger(f"{__name__}.worker") # Use a more specific name for worker logs

# Constants
DEFAULT_NUM_SIMULATIONS = 100
DEFAULT_MAX_ROUNDS = 100
DEFAULT_SUPERMAJORITY_THRESHOLD = 2/3
DEFAULT_RUNOFF_THRESHOLD_ROUNDS = 5 # If no winner after these rounds, consider runoff
DEFAULT_RUNOFF_CANDIDATE_COUNT = 2 # Number of top candidates to include in a runoff


def _run_single_simulation_worker(sim_id: int, elector_data_full: pd.DataFrame, model_params: Dict[str, Any],
                                max_rounds: int, supermajority_threshold: float,
                                runoff_threshold_rounds: int, runoff_candidate_count: int,
                                verbose: bool, 
                                # New params for advanced features
                                enable_candidate_fatigue: bool,
                                fatigue_threshold_rounds: int,
                                fatigue_vote_share_threshold: float,
                                # fatigue_penalty_factor is in model_params
                                fatigue_top_n_immune: int,
                                enable_stop_candidate: bool
                                ) -> Dict[str, Any]:
    if verbose:
        worker_log.setLevel(logging.DEBUG)
    else:
        worker_log.setLevel(logging.INFO)

    worker_log.debug(f"[Sim {sim_id}] Initializing TransitionModel with params: {model_params}")
    model = TransitionModel(elector_data=elector_data_full.copy(), **model_params)
    candidate_ids = model.get_candidate_ids()
    num_candidates = len(candidate_ids)
    elector_ids = model.elector_data.index.tolist()
    num_electors = len(elector_ids)

    # History tracking for fatigue
    candidate_vote_counts_history: List[Dict[str, int]] = [] # Stores vote counts (dict) for each round
    # Store history of vote *shares* as well, for easier fatigue calculation based on shares
    candidate_vote_shares_history: List[np.ndarray] = [] # List of numpy arrays, each array is shares for a round

    current_round_num = 0
    winner = None
    reason_for_ending = "Max rounds reached"
    final_vote_counts = pd.Series(dtype=int)
    effective_candidates_this_round = list(candidate_ids) # Start with all candidates

    worker_log.debug(f"[Sim {sim_id}] Starting simulation loop.")
    for round_num_actual in range(1, max_rounds + 1):
        current_round_num = round_num_actual
        worker_log.debug(f"[Sim {sim_id}] Round {current_round_num}")

        # --- Prepare parameters for calculate_transition_probabilities ---
        fatigued_candidate_indices_set: Set[int] = set()
        
        previous_round_vote_shares_for_model = np.zeros(num_candidates)
        if candidate_vote_shares_history: # Use shares history for stop-cand for now
            previous_round_vote_shares_for_model = candidate_vote_shares_history[-1]
        elif candidate_vote_counts_history: # Fallback to calculating from counts history
            last_round_counts_dict = candidate_vote_counts_history[-1]
            total_votes_last_round = sum(last_round_counts_dict.values())
            if total_votes_last_round > 0:
                for i, cid in enumerate(candidate_ids):
                    previous_round_vote_shares_for_model[i] = last_round_counts_dict.get(str(cid), 0) / total_votes_last_round
        
        # === CANDIDATE FATIGUE LOGIC ===
        if enable_candidate_fatigue and current_round_num > fatigue_threshold_rounds:
            worker_log.debug(f"[Sim {sim_id}] Candidate fatigue check active for round {current_round_num}.")
            # Determine candidates immune due to being in top N
            immune_candidate_indices: Set[int] = set()
            if fatigue_top_n_immune > 0 and candidate_vote_shares_history:
                # Consider average share over last `fatigue_threshold_rounds` or just last round for immunity ranking
                # Using average share from history up to `fatigue_threshold_rounds` deep
                num_history_rounds_for_immunity = min(len(candidate_vote_shares_history), fatigue_threshold_rounds)
                if num_history_rounds_for_immunity > 0:
                    avg_shares_for_immunity = np.mean(candidate_vote_shares_history[-num_history_rounds_for_immunity:], axis=0)
                    top_n_indices_by_avg_share = np.argsort(avg_shares_for_immunity)[-fatigue_top_n_immune:]
                    immune_candidate_indices.update(top_n_indices_by_avg_share)
                    worker_log.debug(f"[Sim {sim_id}] Top {fatigue_top_n_immune} immune candidates (by avg share over last {num_history_rounds_for_immunity} rounds): {immune_candidate_indices}")

            # Check fatigue for non-immune candidates
            # Need at least `fatigue_threshold_rounds` of history to check
            if len(candidate_vote_shares_history) >= fatigue_threshold_rounds:
                relevant_shares_history = candidate_vote_shares_history[-fatigue_threshold_rounds:]
                for cand_idx in range(num_candidates):
                    if cand_idx in immune_candidate_indices:
                        continue # Skip immune candidates

                    # Check if candidate consistently has low vote share
                    is_fatigued = True
                    for round_shares_idx in range(fatigue_threshold_rounds):
                        # relevant_shares_history is list of arrays, round_shares_idx iterates through these arrays (rounds)
                        # cand_idx accesses the specific candidate's share in that round's array
                        if relevant_shares_history[round_shares_idx][cand_idx] >= fatigue_vote_share_threshold:
                            is_fatigued = False
                            break
                    
                    if is_fatigued:
                        fatigued_candidate_indices_set.add(cand_idx)
            if fatigued_candidate_indices_set:
                worker_log.debug(f"[Sim {sim_id}] Candidates fatigued this round: {fatigued_candidate_indices_set}")

        # === Calculate probabilities ===
        probabilities, _ = model.calculate_transition_probabilities(
            previous_round_votes=final_vote_counts, # This is from the *previous* round
            current_round_num=current_round_num,
            fatigued_candidate_indices=fatigued_candidate_indices_set,
            candidate_vote_shares_current_round=previous_round_vote_shares_for_model
        )

        if probabilities.size == 0:
            worker_log.warning(f"[Sim {sim_id}] No probabilities returned. Ending simulation.")
            reason_for_ending = "No probabilities generated"
            break

        # Electors cast their votes
        votes_for_candidates = np.zeros(num_candidates, dtype=int)
        current_round_vote_details = []
        
        # Map candidate IDs to their current indices in the probability matrix
        # This is crucial if effective_candidates_this_round changes (e.g. due to runoff)
        current_candidate_to_prob_idx_map = {cand_id: i for i, cand_id in enumerate(effective_candidates_this_round)}

        for elector_idx in range(num_electors):
            # AI: FIX - Ensure probability row corresponds to the number of *currently effective* candidates
            # and verify that it sums to 1.0
            num_effective = len(effective_candidates_this_round)
            prob_row = probabilities[elector_idx, :num_effective]
            
            # Double-check that probabilities sum to 1.0 (or very close to it) using np.isclose
            if not np.isclose(np.sum(prob_row), 1.0):
                 worker_log.warning(f"[Sim {sim_id}] Probabilities for elector {elector_ids[elector_idx]} do not sum to 1: {np.sum(prob_row)}. Re-normalizing.")
                 # Explicitly re-normalize the probability row before using it
                 prob_row = prob_row / np.sum(prob_row) if np.sum(prob_row) > 0 else np.ones_like(prob_row) / num_effective

            chosen_candidate_effective_idx = np.random.choice(len(effective_candidates_this_round), p=prob_row)
            chosen_candidate_id = effective_candidates_this_round[chosen_candidate_effective_idx]
            
            # Find the original index of the chosen candidate in the full candidate_ids list
            original_candidate_idx = candidate_ids.index(chosen_candidate_id) 
            votes_for_candidates[original_candidate_idx] += 1
            current_round_vote_details.append({'elector_id': elector_ids[elector_idx], 'voted_for': chosen_candidate_id})

        final_vote_counts = pd.Series(votes_for_candidates, index=candidate_ids)
        current_vote_counts_dict_for_history = {str(cid): count for cid, count in final_vote_counts.items()}
        candidate_vote_counts_history.append(current_vote_counts_dict_for_history)

        # Store shares for next round's fatigue calculation
        current_total_votes_cast = final_vote_counts.sum()
        if current_total_votes_cast > 0:
            current_round_shares = final_vote_counts.values / current_total_votes_cast
        else:
            current_round_shares = np.zeros(num_candidates)
        candidate_vote_shares_history.append(current_round_shares)

        # Check for winner
        max_votes = final_vote_counts.max()
        if max_votes >= supermajority_threshold * num_electors:
            winner_id = final_vote_counts.idxmax()
            winner = str(winner_id) # Ensure winner ID is string
            reason_for_ending = f"Winner found by supermajority ({supermajority_threshold*100:.0f}% threshold)"
            worker_log.info(f"[Sim {sim_id}] {reason_for_ending} in round {current_round_num}: {winner}")
            break

        # Check for runoff condition (if no winner yet)
        if current_round_num >= runoff_threshold_rounds and runoff_candidate_count > 0:
            worker_log.info(f"[Sim {sim_id}] Round {current_round_num}: No winner, initiating run-off with top {runoff_candidate_count} candidates.")
            top_candidates = final_vote_counts.nlargest(runoff_candidate_count).index.tolist()
            effective_candidates_this_round = top_candidates
            worker_log.info(f"[Sim {sim_id}] Run-off candidates: {effective_candidates_this_round}")
            # Re-initialize model with subset of candidates or handle subsetting within model
            # For now, we assume the model can handle varying candidate sets if probabilities are indexed correctly
            # This implies the model needs to be aware of effective_candidates_this_round or a mapping.
            # The current TransitionModel is initialized once. For a true runoff with a reduced candidate set,
            # we might need to re-initialize or adapt the model. For now, the probability matrix will be subsetted.
            # Re-mapping candidate_ids to new indices is crucial if model.calculate_transition_probabilities
            # returns a matrix for *all* original candidates.
            # Let's adjust the model to always return based on full set, and subset here.
            # The model's current structure assumes a fixed candidate set defined at init.
            # THIS PART NEEDS CAREFUL REVIEW WITH THE MODEL'S ASSUMPTIONS FOR RUNOFFS
            # For now, we continue with the full probability matrix and just select from effective candidates.
            # This means electors *can* still vote for non-runoff candidates if their prob is non-zero, which is not a true runoff.
            # To implement a true runoff, the model.calculate_transition_probabilities would need to be called
            # with only the runoff candidates, or its output filtered and re-normalized for runoff candidates.
            # This simplified runoff logic (electors vote only from top N based on prob) is a temporary measure.
            # A more robust runoff would restrict model.calculate_transition_probabilities to only consider runoff candidates.
            # For this iteration, the model's probabilities are for all candidates, but choices are restricted.
            # This is not ideal; a proper runoff would re-calculate probabilities among ONLY runoff candidates.
            pass # Current model continues with all candidates, choice logic above handles effective candidates.

    if winner is None and current_round_num == max_rounds:
        worker_log.info(f"[Sim {sim_id}] Max rounds ({max_rounds}) reached. No winner decided through threshold.")

    return {
        'sim_id': sim_id,
        'winner': winner,
        'rounds': current_round_num,
        'reason': reason_for_ending,
        'final_votes': final_vote_counts.to_dict() if final_vote_counts is not None else {},
        'run_details': [] # Placeholder for more detailed round-by-round logging if needed later
    }

def parse_args():
    parser = argparse.ArgumentParser(description="Run Conclave simulations.")
    parser.add_argument("-n", "--num-simulations", type=int, default=DEFAULT_NUM_SIMULATIONS, help="Number of simulations to run.")
    parser.add_argument("-r", "--max-rounds", type=int, default=DEFAULT_MAX_ROUNDS, help="Maximum number of voting rounds per simulation.")
    parser.add_argument("-t", "--supermajority-threshold", type=float, default=DEFAULT_SUPERMAJORITY_THRESHOLD, help="Supermajority threshold for winning.")
    parser.add_argument("--runoff-rounds", type=int, default=DEFAULT_RUNOFF_THRESHOLD_ROUNDS, help="Rounds after which to trigger a runoff if no winner.")
    parser.add_argument("--runoff-candidates", type=int, default=DEFAULT_RUNOFF_CANDIDATE_COUNT, help="Number of top candidates in a runoff.")
    parser.add_argument("-f", "--elector-data-file", type=str, default="merged_electors.csv", help="Path to the elector data CSV file.")
    parser.add_argument("-o", "--output-results", type=str, default="data/simulation_results.csv", help="Path to save detailed simulation results.")
    parser.add_argument("-s", "--output-stats", type=str, default="data/simulation_stats.json", help="Path to save aggregate simulation statistics.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging for simulation workers.")
    parser.add_argument("-w", "--workers", type=int, default=None, help="Number of worker processes (default: number of CPU cores).")

    # Model specific parameters
    model_group = parser.add_argument_group('Transition Model Parameters')
    model_group.add_argument("--initial-beta-weight", type=float, default=1.0, help="Initial weight for ideological distance sensitivity.") # Renamed
    model_group.add_argument("--beta-increment-amount", type=float, default=0.05, help="Increment to beta weight per N scaled rounds (for Dynamic Beta).")
    model_group.add_argument("--beta-increment-interval-rounds", type=float, default=10.0, help="Number of rounds over which beta increments fully (for Dynamic Beta).")
    model_group.add_argument("--enable-dynamic-beta", action='store_true', help="Enable dynamic beta adjustment.")
    model_group.add_argument("--stickiness-factor", type=float, default=0.5, help="Factor for elector stickiness to previous vote.")
    model_group.add_argument("--bandwagon-strength", type=float, default=0.0, help="Strength of the bandwagon effect.")
    model_group.add_argument("--regional-bonus", type=float, default=0.1, help="Bonus for candidates from the same region.")
    model_group.add_argument("--papabile-weight-factor", type=float, default=1.5, help="Multiplicative weight for papabile candidates.")
    
    # Candidate Fatigue Parameters
    fatigue_group = parser.add_argument_group('Candidate Fatigue Parameters')
    fatigue_group.add_argument("--enable-candidate-fatigue", action='store_true', help="Enable candidate fatigue mechanism.")
    fatigue_group.add_argument("--fatigue-threshold-rounds", type=int, default=3, help="Number of recent rounds to consider for fatigue.")
    fatigue_group.add_argument("--fatigue-vote-share-threshold", type=float, default=0.05, help="Vote share below which a candidate is considered for fatigue.")
    fatigue_group.add_argument("--fatigue-penalty-factor", type=float, default=0.5, help="Factor to penalize preference scores of fatigued candidates.")
    fatigue_group.add_argument("--fatigue-top-n-immune", type=int, default=3, help="Top N candidates (by recent vote share) immune to fatigue.")

    # Stop Candidate Parameters
    stop_cand_group = parser.add_argument_group('Stop Candidate Parameters')
    stop_cand_group.add_argument("--enable-stop-candidate", action='store_true', help="Enable stop candidate behavior.")
    stop_cand_group.add_argument("--stop-candidate-threshold-unacceptable-distance", type=float, default=0.5, help="Ideological distance beyond which a candidate is 'unacceptable'.")
    stop_cand_group.add_argument("--stop-candidate-threat-min-vote-share", type=float, default=0.15, help="Vote share a candidate needs to be a 'threat'.")
    stop_cand_group.add_argument("--stop-candidate-boost-factor", type=float, default=1.5, help="Factor to boost preference for a strategic 'blocker' candidate.")

    return parser.parse_args()

def model_params_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    """Extracts model parameters from command line arguments."""
    params = {
        "initial_beta_weight": args.initial_beta_weight,
        "enable_dynamic_beta": args.enable_dynamic_beta,
        "beta_increment_amount": args.beta_increment_amount, # Pass directly
        "beta_increment_interval_rounds": int(args.beta_increment_interval_rounds), # Pass directly, ensure int
        # beta_growth_rate uses TransitionModel's default if not overridden and dynamic beta (growth type) is on
        "stickiness_factor": args.stickiness_factor,
        "bandwagon_strength": args.bandwagon_strength,
        "regional_affinity_bonus": args.regional_bonus,
        "papabile_weight_factor": args.papabile_weight_factor,
        "fatigue_penalty_factor": args.fatigue_penalty_factor, # Passed to model init
        "stop_candidate_threshold_unacceptable_distance": args.stop_candidate_threshold_unacceptable_distance, # Passed to model init
        "stop_candidate_boost_factor": args.stop_candidate_boost_factor, # Passed to model init
        "stop_candidate_threat_min_vote_share": args.stop_candidate_threat_min_vote_share # Added for model init
    }

    if args.enable_dynamic_beta:
        if args.beta_increment_amount is not None and args.beta_increment_amount != 0:
            log.info(f"Dynamic Beta Enabled (stepped): Increment Amount={args.beta_increment_amount}, Interval Rounds={int(args.beta_increment_interval_rounds)}.")
        else:
            # This case implies beta_growth_rate would be used by the model if > 1.0
            log.info(f"Dynamic Beta Enabled (likely multiplicative growth): beta_increment_amount is {args.beta_increment_amount}. TransitionModel's default beta_growth_rate will apply if > 1.0.")
    else:
        log.info("Dynamic Beta Disabled.")

    if args.enable_candidate_fatigue:
        log.info(f"Candidate Fatigue Enabled: Threshold Rounds={args.fatigue_threshold_rounds}, Vote Share Threshold={args.fatigue_vote_share_threshold*100:.0f}%, Penalty={args.fatigue_penalty_factor}, Top N Immune={args.fatigue_top_n_immune}")
    if args.enable_stop_candidate:
        log.info(f"Stop Candidate Enabled: Unacceptable Distance={args.stop_candidate_threshold_unacceptable_distance}, "
                 f"Threat Share={args.stop_candidate_threat_min_vote_share*100:.0f}%, "
                 f"Boost Factor={args.stop_candidate_boost_factor}")

    return params

def main():
    args = parse_args()

    if args.verbose:
        log.setLevel(logging.DEBUG)
    else:
        log.setLevel(logging.INFO)

    log.info(f"Starting Conclave simulation with {args.num_simulations} simulations.")
    log.info(f"Parameters: Max Rounds={args.max_rounds}, Supermajority={args.supermajority_threshold*100:.0f}%" 
             f", Runoff Rounds={args.runoff_rounds}, Runoff Candidates={args.runoff_candidates}")
    log.info(f"Model Params: Initial Beta={args.initial_beta_weight}, Stickiness={args.stickiness_factor}, Bandwagon={args.bandwagon_strength}, "
             f"Regional Bonus={args.regional_bonus}, Papabile Factor={args.papabile_weight_factor}")
    if args.enable_candidate_fatigue:
        log.info(f"Candidate Fatigue Enabled: Threshold Rounds={args.fatigue_threshold_rounds}, Vote Share Threshold={args.fatigue_vote_share_threshold*100:.0f}%, Penalty={args.fatigue_penalty_factor}, Top N Immune={args.fatigue_top_n_immune}")
    if args.enable_stop_candidate:
        log.info(f"Stop Candidate Enabled: Unacceptable Distance={args.stop_candidate_threshold_unacceptable_distance}, "
                 f"Threat Share={args.stop_candidate_threat_min_vote_share*100:.0f}%, "
                 f"Boost Factor={args.stop_candidate_boost_factor}")
    if args.beta_increment_amount > 0:
        log.info(f"Dynamic Beta Enabled: Increment={args.beta_increment_amount} per {args.beta_increment_interval_rounds} rounds.")

    ingester = ElectorDataIngester(args.elector_data_file)
    elector_data_full = ingester.load_and_prepare_data()
    if elector_data_full.empty:
        log.error("Failed to load or prepare elector data. Exiting.")
        return

    model_params = model_params_from_args(args)

    start_time = time.time()
    results = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                _run_single_simulation_worker, 
                i, elector_data_full, model_params,
                args.max_rounds, args.supermajority_threshold,
                args.runoff_rounds, args.runoff_candidates, args.verbose,
                # Pass new feature flags and specific params to worker
                args.enable_candidate_fatigue,
                args.fatigue_threshold_rounds,
                args.fatigue_vote_share_threshold,
                # fatigue_penalty_factor not passed separately, it's in model_params
                args.fatigue_top_n_immune,
                args.enable_stop_candidate
            ) for i in range(args.num_simulations)
        ]
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                log.error(f"Error in simulation worker: {e}", exc_info=True)

    simulation_time = time.time() - start_time
    log.info(f"Simulation time: {simulation_time:.2f} seconds")

    # Process results
    successful_simulations = sum(1 for r in results if r['winner'] is not None)
    success_rate = successful_simulations / args.num_simulations if args.num_simulations > 0 else 0
    
    rounds_successful = [r['rounds'] for r in results if r['winner'] is not None]
    avg_rounds_successful = np.mean(rounds_successful) if rounds_successful else None
    min_rounds_successful = np.min(rounds_successful) if rounds_successful else None
    max_rounds_successful = np.max(rounds_successful) if rounds_successful else None
    median_rounds_successful = np.median(rounds_successful) if rounds_successful else None

    all_winner_ids = [r['winner'] for r in results if r['winner'] is not None]
    winner_counts = Counter(all_winner_ids)
    most_frequent_winner = winner_counts.most_common(1)[0][0] if winner_counts else None
    most_frequent_winner_count = winner_counts.most_common(1)[0][1] if winner_counts else 0
    mfs_percentage = most_frequent_winner_count / args.num_simulations if args.num_simulations > 0 else 0
    
    avg_rounds_all = np.mean([r['rounds'] for r in results]) if results else 0

    aggregate_stats = {
        'total_simulations': args.num_simulations,
        'successful_simulations': successful_simulations,
        'success_rate': success_rate,
        'average_rounds_successful': avg_rounds_successful,
        'min_rounds_successful': min_rounds_successful,
        'max_rounds_successful': max_rounds_successful,
        'median_rounds_successful': median_rounds_successful,
        'winner_counts': dict(winner_counts),
        'most_frequent_winner': most_frequent_winner,
        'most_frequent_winner_count': most_frequent_winner_count,
        'most_frequent_winner_percentage': mfs_percentage,
        'average_rounds_all': avg_rounds_all
    }

    log.info(f"Calculated aggregate stats: {aggregate_stats}")

    # Save detailed results to CSV
    # Expanding final_votes dict into separate columns might be too wide.
    # Storing as a dictionary string in the CSV or one row per final vote.
    # For now, just basic info.
    results_df_data = []
    for r in results:
        row = {
            'sim_id': r['sim_id'], 
            'winner': r['winner'], 
            'rounds': r['rounds'], 
            'reason': r['reason'],
            # 'final_votes': json.dumps(r['final_votes']) # Store dict as JSON string
        }
        # Add individual final votes for easier analysis if not too many candidates
        # This might be better handled by a separate detailed votes log file
        # for c_id, c_votes in r['final_votes'].items():
        #     row[f"votes_{c_id}"] = c_votes
        results_df_data.append(row)

    results_df = pd.DataFrame(results_df_data)
    try:
        results_df.to_csv(args.output_results, index=False)
        log.info(f"Saving simulation results ({len(results_df)} rows) to: {args.output_results}")
    except Exception as e:
        log.error(f"Failed to save results CSV: {e}")

    # Save aggregate stats to JSON
    output_stats_path = Path(args.output_stats)
    output_stats_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(output_stats_path, "w") as f:
            json.dump(aggregate_stats, f, indent=4, default=lambda x: int(x) if isinstance(x, np.integer) else None if x is None or (isinstance(x, float) and np.isnan(x)) else x)
        log.info(f"Saving aggregate stats to: {output_stats_path}")
    except Exception as e:
        log.error(f"Failed to save stats JSON: {e}")

    log.info("\n--- Simulation Finished ---")
    log.info(f"Success Rate: {success_rate*100:.2f}%")
    log.info(f"Average Rounds (Successful): {avg_rounds_successful if avg_rounds_successful is not None else 'N/A'}")
    log.info(f"Most Frequent Winner: Elector {most_frequent_winner} ({most_frequent_winner_count} times, {mfs_percentage*100:.2f}%)")

if __name__ == "__main__":
    main()
