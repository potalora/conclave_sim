# # AI: Main execution script for the Conclave Simulation

import argparse
import sys
import os
import pandas as pd
import numpy as np
import traceback
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# AI: Assuming src is in PYTHONPATH or running with python -m src
# Remove ingest import, load directly
# from .ingest import load_elector_data
from .simulate import run_monte_carlo_simulation

def main():
    """Main entry point for running the simulation from the command line."""
    parser = argparse.ArgumentParser(description="Run a Monte Carlo Papal Conclave Simulation.")
    parser.add_argument(
        "elector_file",
        type=str,
        help="Path to the CSV file containing elector data."
    )
    parser.add_argument(
        "-n", "--num-simulations",
        type=int,
        default=100,
        help="Number of Monte Carlo simulations to run (default: 100)."
    )
    parser.add_argument(
        "-b", "--beta-weight",
        type=float,
        default=0.5,
        help="Beta weight for ideology influence in the transition model (default: 0.5)."
    )
    # AI: Add argument for stickiness factor
    parser.add_argument(
        "-s", "--stickiness-factor",
        type=float,
        default=0.2,
        help="Stickiness factor for vote transition model (0-1, default: 0.2)."
    )
    # AI: Add verbosity flag
    parser.add_argument(
        "-v", "--verbose",
        action="store_true", # Set to True if flag is present
        help="Print detailed round-by-round simulation progress."
    )
    # TODO: Add more arguments for other model parameters as needed

    args = parser.parse_args()

    print(f"Starting Conclave Simulation...")
    print(f" - Elector Data: {args.elector_file}")
    print(f" - Simulations: {args.num_simulations}")
    print(f" - Beta Weight: {args.beta_weight}")
    print(f" - Stickiness Factor: {args.stickiness_factor}") # AI: Print new param

    # --- 1. Load Data ---
    try:
        # AI: Load data directly using pandas
        print(f"Loading elector data from: {args.elector_file}")
        if not os.path.exists(args.elector_file):
            raise FileNotFoundError(f"Elector file not found: {args.elector_file}")
        elector_df = pd.read_csv(args.elector_file)
        if elector_df.empty:
            raise ValueError("Elector file is empty.")
        # Basic check for required column (others checked in simulation func)
        if 'elector_id' not in elector_df.columns:
             raise ValueError("Elector file must contain an 'elector_id' column.")
        # AI: Add check for the conclavoscope score column
        if 'conclavoscope_score' not in elector_df.columns:
            raise ValueError("Elector file must contain a 'conclavoscope_score' column. Run ingest script first.")

        print(f"Loaded {len(elector_df)} electors successfully.")
    except (FileNotFoundError, ValueError, pd.errors.EmptyDataError, Exception) as e:
        print(f"Error loading elector data: {e}", file=sys.stderr)
        sys.exit(1)

    # --- 1b. Assign Ideology Score --- (AI: Modified Step)
    try:
        # AI: Assign ideology_score from the pre-calculated conclavoscope_score
        log.info("Assigning 'ideology_score' from 'conclavoscope_score' column.")
        initial_nan_count = elector_df['conclavoscope_score'].isna().sum()
        if initial_nan_count > 0:
            log.warning(f"Found {initial_nan_count} missing values in 'conclavoscope_score'. Filling with 0.0.")
            elector_df['ideology_score'] = elector_df['conclavoscope_score'].fillna(0.0)
        else:
            elector_df['ideology_score'] = elector_df['conclavoscope_score']

        # Ensure the final score is within the expected -1 to 1 range (optional, but good practice)
        elector_df['ideology_score'] = np.clip(elector_df['ideology_score'], -1.0, 1.0)
        log.info("Successfully assigned 'ideology_score'.")

    except KeyError as e:
        # This check is now less likely due to the earlier check, but kept for safety
        print(f"Error assigning ideology score: Missing required column - {e}. Ensure 'conclavoscope_score' exists.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during ideology score assignment: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    # --- 2. Define Model Parameters ---
    model_parameters = {
        'beta_weight': args.beta_weight,
        # AI: Add stickiness factor to parameters
        'stickiness_factor': args.stickiness_factor
        # Add other parameters here if the model evolves
    }

    # --- 3. Run Simulation ---
    try:
        results_df, aggregate_stats = run_monte_carlo_simulation(
            num_simulations=args.num_simulations,
            elector_data=elector_df,
            model_parameters=model_parameters,
            verbose=args.verbose # AI: Pass verbose flag
        )
        print("\n--- Simulation Complete ---")

        # --- 4. Print Results ---
        print("Aggregate Statistics:")
        for key, value in aggregate_stats.items():
            # Format percentages nicely
            if 'rate' in key and isinstance(value, float):
                print(f"  {key.replace('_', ' ').title()}: {value:.2%}")
            else:
                 print(f"  {key.replace('_', ' ').title()}: {value}")

        # Optionally print or save the detailed results_df
        # print("\nDetailed Results:")
        # print(results_df.head())
        # results_df.to_csv("simulation_results.csv", index=False)

    except ValueError as e:
        print(f"Error during simulation: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        # Catch unexpected errors during simulation
        print(f"An unexpected error occurred during simulation: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
