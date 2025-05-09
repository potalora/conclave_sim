#!/usr/bin/env python

import logging
import pandas as pd
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

from src.viz import (
    plot_rounds_distribution,
    plot_winner_distribution,
    plot_convergence_timeline,
    plot_voting_heatmap,
    create_simulation_dashboard,
    generate_all_visualizations
)

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Input files for visualizations
    results_csv_path = Path("data/simulation_results_1000_fast.csv")
    detailed_json_path = Path("data/simulation_stats_1000_fast.json")
    
    # Output directory for visualizations
    output_dir = Path("data/plots/1000_fast")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data files
    if results_csv_path.exists():
        try:
            results_df = pd.read_csv(results_csv_path)
            logger.info(f"Loaded results data from {results_csv_path} with {len(results_df)} rows")
        except Exception as e:
            logger.error(f"Error loading results CSV: {e}")
            results_df = pd.DataFrame()  # Empty DataFrame as fallback
    else:
        logger.warning(f"Results CSV not found at {results_csv_path}")
        results_df = pd.DataFrame()  # Empty DataFrame as fallback
    
    # Generate individual visualizations if we have data
    if not results_df.empty:
        logger.info("Generating rounds distribution visualization...")
        plot_rounds_distribution(
            results_df=results_df, 
            output_path=output_dir / "rounds_distribution.png"
        )
        
        logger.info("Generating winner distribution visualization...")
        plot_winner_distribution(
            results_df=results_df,
            output_path=output_dir / "winner_distribution.png",
            top_n=10
        )
        
        logger.info("Generating convergence timeline visualization...")
        plot_convergence_timeline(
            results_df=results_df,
            output_path=output_dir / "convergence_timeline.png",
            include_failures=True
        )
    else:
        logger.warning("Cannot generate visualizations because results DataFrame is empty")
    
    # If we have detailed results with round-by-round data
    if detailed_json_path.exists():
        logger.info("Generating voting heatmap for a sample simulation...")
        plot_voting_heatmap(
            results_file=detailed_json_path,
            output_path=output_dir / "voting_heatmap_sim0.png",
            simulation_id=0,
            max_rounds=10,
            top_candidates=10
        )
    
    # Create comprehensive dashboard
    if not results_df.empty:
        logger.info("Creating complete simulation dashboard...")
        create_simulation_dashboard(
            results_csv=results_csv_path,
            output_dir=output_dir / "dashboard",
            detailed_results_json=detailed_json_path if detailed_json_path.exists() else None
        )
    else:
        logger.warning("Cannot create dashboard because results DataFrame is empty")
    
    # Compare parameter variations if multiple parameter files exist
    parameter_files = {
        "Baseline": Path("data/simulation_stats_baseline.json"),
        "Fast Converge": Path("data/simulation_stats_fastconverge.json"),
        "Beta 1.0": Path("data/simulation_stats_beta1.0_full.json"),
        "Adjusted Beta+Bandwagon": Path("data/simulation_stats_adj_beta_bandwagon.json"),
        "Optimized 1000": Path("data/simulation_stats_1000_fast.json")
    }
    
    # Filter to only include existing files
    parameter_comparisons = {k: v for k, v in parameter_files.items() if v.exists()}
    
    if parameter_comparisons:
        logger.info(f"Found {len(parameter_comparisons)} parameter variations for comparison: {list(parameter_comparisons.keys())}")
        # Select a baseline path to compare against
        baseline_path = parameter_comparisons.get("Baseline", next(iter(parameter_comparisons.values())))
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        from src.viz import plot_parameter_impact_matrix
        
        # Check that we can read the parameter files
        valid_params = True
        for name, path in parameter_comparisons.items():
            try:
                with open(path, 'r') as f:
                    json.load(f)  # Just test if it can be loaded
                logger.info(f"Successfully validated parameter file: {name}")
            except Exception as e:
                logger.error(f"Error reading parameter file {name} at {path}: {e}")
                valid_params = False
        
        if valid_params:
            logger.info("Generating parameter impact visualizations...")
            plot_parameter_impact_matrix(
                base_results_path=baseline_path,
                param_var_results_paths=parameter_comparisons,
                output_path=output_dir / "parameter_impact.png",
                metrics=['success_rate', 'avg_rounds']
            )
        else:
            logger.warning("Skipping parameter impact visualization due to errors in parameter files")
    else:
        logger.warning("No parameter comparison files found")
    
    logger.info("Visualization generation complete!")

