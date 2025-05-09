import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple, Any, Union
import logging

# Set up logging
log = logging.getLogger(__name__)

def plot_ideology_distribution(
    ideology_scores: pd.Series,
    output_path: Path,
    bins: int = 20
) -> None:
    """Plots and saves a histogram of the ideology scores.

    Args:
        ideology_scores: A pandas Series containing the ideology scores.
        output_path: The path (including filename) to save the plot image.
        bins: The number of bins for the histogram.
    """
    if ideology_scores.empty:
        log.warning("Ideology scores series is empty. Skipping plot.")
        return

    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    plt.hist(ideology_scores.dropna(), bins=bins, edgecolor='black', alpha=0.7)
    plt.title(f'Distribution of Elector Ideology Scores (N={len(ideology_scores)})',
              fontsize=16)
    plt.xlabel('Ideology Score', fontsize=14)
    plt.ylabel('Number of Electors', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', alpha=0.75)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        log.info(f"Ideology distribution plot saved to: {output_path}")
    except Exception as e:
        log.error(f"Error saving plot: {e}")
    plt.close() # Close the plot to free memory


def plot_rounds_distribution(
    results_df: pd.DataFrame,
    output_path: Path,
    max_rounds: int = None,
    kde: bool = True
) -> None:
    """Plots and saves a histogram of rounds to convergence.

    Args:
        results_df: DataFrame containing simulation results with a 'rounds' column.
        output_path: The path (including filename) to save the plot image.
        max_rounds: Optional maximum rounds to include in the visualization.
        kde: Whether to include a kernel density estimate curve.
    """
    if results_df.empty:
        log.warning("Results DataFrame is empty. Skipping rounds distribution plot.")
        return

    # Filter to only include simulations that reached a winner
    successful_df = results_df.dropna(subset=['winner'])
    
    if successful_df.empty:
        log.warning("No successful simulations found. Skipping rounds distribution plot.")
        return

    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    sns.set_palette("viridis")
    
    # Create the histogram with density curve
    ax = sns.histplot(data=successful_df, x='rounds', bins=range(1, int(successful_df['rounds'].max()) + 2), 
                      kde=kde, color='#2c7bb6', alpha=0.7, stat='density', discrete=True)
    
    # Add vertical line for mean
    mean_rounds = successful_df['rounds'].mean()
    plt.axvline(x=mean_rounds, color='#d7301f', linestyle='--', 
                linewidth=2, label=f'Mean: {mean_rounds:.2f} rounds')
    
    # Add vertical line for median
    median_rounds = successful_df['rounds'].median()
    plt.axvline(x=median_rounds, color='#fdae61', linestyle='-.', 
                linewidth=2, label=f'Median: {median_rounds:.1f} rounds')
    
    # Add count annotations above each bar
    for p in ax.patches:
        count = int(p.get_height() * len(successful_df))
        if count > 0:  # Only annotate non-zero bars
            ax.annotate(f'{count}', 
                        (p.get_x() + p.get_width()/2., p.get_height()), 
                        ha='center', va='bottom', fontsize=10)
    
    plt.title(f'Distribution of Rounds to Convergence (N={len(successful_df)})', fontsize=16)
    plt.xlabel('Number of Rounds', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    
    # Set x-axis ticks to integers
    plt.xticks(range(1, int(successful_df['rounds'].max()) + 1), fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.legend(fontsize=12)
    plt.grid(axis='y', alpha=0.3)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        log.info(f"Rounds distribution plot saved to: {output_path}")
    except Exception as e:
        log.error(f"Error saving plot: {e}")
    plt.close() # Close the plot to free memory


def plot_winner_distribution(
    results_df: pd.DataFrame, 
    output_path: Path,
    top_n: int = 10
) -> None:
    """Creates a bar chart showing the distribution of winning electors.

    Args:
        results_df: DataFrame containing simulation results with a 'winner' column.
        output_path: The path (including filename) to save the plot image.
        top_n: Number of top winners to display.
    """
    if results_df.empty:
        log.warning("Results DataFrame is empty. Skipping winner distribution plot.")
        return

    # Count winner frequencies, excluding non-winners (NaN values)
    winner_counts = results_df['winner'].value_counts().reset_index()
    winner_counts.columns = ['winner', 'count']
    
    # Calculate percentage of total simulations
    total_simulations = len(results_df)
    winner_counts['percentage'] = (winner_counts['count'] / total_simulations) * 100
    
    # Take top N winners
    if len(winner_counts) > top_n:
        top_winners = winner_counts.iloc[:top_n]
        remaining_count = winner_counts.iloc[top_n:]['count'].sum()
        remaining_pct = winner_counts.iloc[top_n:]['percentage'].sum()
        
        # Add an "Others" category for remaining winners
        if remaining_count > 0:
            others = pd.DataFrame({'winner': ['Others'], 
                                 'count': [remaining_count],
                                 'percentage': [remaining_pct]})
            top_winners = pd.concat([top_winners, others])
    else:
        top_winners = winner_counts
        
    # Add a category for simulations with no winner
    no_winner_count = results_df['winner'].isna().sum()
    if no_winner_count > 0:
        no_winner_pct = (no_winner_count / total_simulations) * 100
        no_winner = pd.DataFrame({'winner': ['No Winner'], 
                                'count': [no_winner_count],
                                'percentage': [no_winner_pct]})
        top_winners = pd.concat([top_winners, no_winner])
    
    # Plot the winner distribution
    plt.figure(figsize=(14, 8))
    sns.set_style("whitegrid")
    
    # Create custom colormap with distinct colors
    palette = sns.color_palette("viridis", len(top_winners))
    # Make the "No Winner" category red if it exists
    if 'No Winner' in top_winners['winner'].values:
        no_winner_idx = top_winners[top_winners['winner'] == 'No Winner'].index[0]
        palette_list = list(palette)
        palette_list[no_winner_idx] = '#d7301f'  # Red color for No Winner
        palette = palette_list
    
    # Create the bar plot
    ax = sns.barplot(x='winner', y='count', data=top_winners, palette=palette)
    
    # Add value labels on top of each bar
    for i, p in enumerate(ax.patches):
        pct = top_winners.iloc[i]['percentage']
        ax.annotate(f'{p.get_height()} ({pct:.1f}%)', 
                    (p.get_x() + p.get_width()/2., p.get_height()), 
                    ha='center', va='bottom', fontsize=11)
    
    plt.title(f'Distribution of Winners Across {total_simulations} Simulations', fontsize=16)
    plt.xlabel('Elector ID', fontsize=14)
    plt.ylabel('Number of Wins', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        log.info(f"Winner distribution plot saved to: {output_path}")
    except Exception as e:
        log.error(f"Error saving plot: {e}")
    plt.close() # Close the plot to free memory


def plot_convergence_timeline(
    results_df: pd.DataFrame,
    output_path: Path,
    include_failures: bool = True,
    max_rounds: int = None
) -> None:
    """Creates a visualization showing when simulations converged to a winner.

    Args:
        results_df: DataFrame containing simulation results.
        output_path: The path to save the visualization.
        include_failures: Whether to include simulations that didn't reach a winner.
        max_rounds: Optional maximum rounds to include in the visualization.
    """
    if results_df.empty:
        log.warning("Results DataFrame is empty. Skipping convergence timeline plot.")
        return

    # Create a copy to avoid modifying the original
    plot_df = results_df.copy()
    
    # Add a simulation status column
    plot_df['status'] = plot_df['winner'].apply(lambda x: 'Converged' if pd.notna(x) else 'No Winner')
    
    # Filter out non-converging simulations if requested
    if not include_failures:
        plot_df = plot_df[plot_df['status'] == 'Converged']
    
    if plot_df.empty:
        log.warning("No data to plot after filtering. Skipping convergence timeline plot.")
        return
    
    # Set max rounds for visualization if not provided
    if max_rounds is None:
        max_rounds = plot_df['rounds'].max()
    
    # Cap rounds at max_rounds for better visualization
    plot_df['rounds_capped'] = plot_df['rounds'].apply(lambda x: min(x, max_rounds))
    
    # Sort by rounds for better visualization
    plot_df = plot_df.sort_values(['status', 'rounds_capped'])
    
    # Create a column for the y-axis position (simulation ID)
    plot_df['simulation'] = range(len(plot_df))
    
    # Prepare the plot
    plt.figure(figsize=(12, 10))
    sns.set_style("whitegrid")
    
    # Create a custom colormap
    colors = {'Converged': '#4daf4a', 'No Winner': '#e41a1c'}
    
    # Create the scatter plot
    ax = sns.scatterplot(data=plot_df, x='rounds_capped', y='simulation', 
                      hue='status', palette=colors, s=100, alpha=0.7)
    
    # Add a reference line for median convergence
    median_rounds = plot_df[plot_df['status'] == 'Converged']['rounds'].median()
    if not pd.isna(median_rounds):
        plt.axvline(x=median_rounds, color='#984ea3', linestyle='--', 
                    linewidth=2, label=f'Median Convergence: {median_rounds} rounds')
    
    # Add counts to legend
    for status in plot_df['status'].unique():
        count = (plot_df['status'] == status).sum()
        plt.plot([], [], ' ', label=f'{status}: {count} simulations')
    
    # Set plot labels and title
    plt.title('Convergence Timeline Across Simulations', fontsize=16)
    plt.xlabel('Number of Rounds', fontsize=14)
    plt.ylabel('Simulation Index', fontsize=14)
    
    # Set x-axis ticks to integers
    plt.xticks(range(1, max_rounds + 1), fontsize=12)
    plt.xlim(0.5, max_rounds + 0.5)  # Add padding around the data
    
    # Remove y-axis ticks since they're just indices
    plt.yticks([])
    
    # Customize the legend
    plt.legend(title='Status', fontsize=12, title_fontsize=14)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        log.info(f"Convergence timeline plot saved to: {output_path}")
    except Exception as e:
        log.error(f"Error saving plot: {e}")
    plt.close()


def plot_voting_heatmap(
    results_file: Path,
    output_path: Path,
    simulation_id: int = 0,
    max_rounds: int = 10,
    top_candidates: int = 10
) -> None:
    """Creates a heatmap showing vote distribution across rounds for a specific simulation.

    Args:
        results_file: Path to the detailed simulation results file (with round-by-round votes).
        output_path: The path to save the visualization.
        simulation_id: The specific simulation ID to visualize.
        max_rounds: Maximum rounds to include in the visualization.
        top_candidates: Number of top candidates to include.
    """
    try:
        # Load the detailed results
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Check if the requested simulation exists
        if simulation_id >= len(results):
            log.warning(f"Simulation ID {simulation_id} not found in results. Skipping voting heatmap.")
            return
        
        sim_data = results[simulation_id]
        rounds_data = sim_data.get('round_data', [])
        
        if not rounds_data:
            log.warning(f"No round data found for simulation {simulation_id}. Skipping voting heatmap.")
            return
        
        # Limit to max_rounds
        rounds_data = rounds_data[:min(max_rounds, len(rounds_data))]
        
        # Extract vote counts by round
        vote_data = {}
        for round_idx, round_data in enumerate(rounds_data):
            round_num = round_idx + 1
            votes = round_data.get('votes', {})
            
            # Handle different possible formats of the vote data
            if isinstance(votes, dict):
                vote_counts = votes
            elif isinstance(votes, list):
                vote_counts = votes[0] if votes else {}
            else:
                vote_counts = {}
            
            # Store vote counts by candidate for this round
            for candidate, count in vote_counts.items():
                if candidate not in vote_data:
                    vote_data[candidate] = [0] * len(rounds_data)
                vote_data[candidate][round_idx] = count
        
        # Find top candidates based on maximum votes across all rounds
        candidate_max_votes = {c: max(votes) for c, votes in vote_data.items()}
        top_candidate_ids = sorted(candidate_max_votes.keys(), 
                                   key=lambda c: candidate_max_votes[c], 
                                   reverse=True)[:top_candidates]
        
        # Create a DataFrame for the heatmap
        heatmap_data = []
        for candidate in top_candidate_ids:
            for round_idx, vote_count in enumerate(vote_data[candidate]):
                heatmap_data.append({
                    'Candidate': f"Elector {candidate}",
                    'Round': round_idx + 1,
                    'Votes': vote_count
                })
        
        heatmap_df = pd.DataFrame(heatmap_data)
        
        # Pivot the data for the heatmap
        pivot_df = heatmap_df.pivot(index='Candidate', columns='Round', values='Votes')
        
        # Create the heatmap
        plt.figure(figsize=(14, 10))
        sns.set_style("whitegrid")
        
        # Use a sequential colormap that's effective for vote counts
        ax = sns.heatmap(pivot_df, annot=True, fmt="d", cmap="YlGnBu", 
                      linewidths=.5, cbar_kws={'label': 'Vote Count'})
        
        # Set plot labels and title
        plt.title(f'Vote Distribution by Round for Simulation {simulation_id}', fontsize=16)
        plt.xlabel('Round', fontsize=14)
        plt.ylabel('Candidate ID', fontsize=14)
        
        # Adjust x-axis ticks
        plt.xticks(rotation=0, fontsize=12)
        plt.yticks(fontsize=12)
        
        # Add a note about the outcome
        winner = sim_data.get('winner')
        rounds = sim_data.get('rounds')
        if winner:
            outcome_text = f"Winner: Elector {winner} (Round {rounds})"
        else:
            outcome_text = "No winner reached"
        
        plt.figtext(0.5, 0.01, outcome_text, ha="center", fontsize=14, 
                    bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            log.info(f"Voting heatmap saved to: {output_path}")
        except Exception as e:
            log.error(f"Error saving plot: {e}")
        plt.close()
    
    except Exception as e:
        log.error(f"Error creating voting heatmap: {e}")


def create_simulation_dashboard(
    results_csv: Path,
    output_dir: Path,
    detailed_results_json: Optional[Path] = None
) -> None:
    """Creates a comprehensive dashboard of visualizations for simulation results.

    Args:
        results_csv: Path to the simulation results CSV file.
        output_dir: Directory to save all visualization outputs.
        detailed_results_json: Optional path to detailed results JSON with round-by-round data.
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load simulation results
        results_df = pd.read_csv(results_csv)
        
        if results_df.empty:
            log.warning("Results DataFrame is empty. Cannot create dashboard.")
            return
        
        # 1. Create rounds distribution plot
        rounds_plot_path = output_dir / "rounds_distribution.png"
        plot_rounds_distribution(results_df, rounds_plot_path)
        
        # 2. Create winner distribution plot
        winner_plot_path = output_dir / "winner_distribution.png"
        plot_winner_distribution(results_df, winner_plot_path)
        
        # 3. Create convergence timeline plot
        timeline_plot_path = output_dir / "convergence_timeline.png"
        plot_convergence_timeline(results_df, timeline_plot_path)
        
        # 4. Create voting heatmap if detailed results are available
        if detailed_results_json and detailed_results_json.exists():
            # Get a sample of successful and unsuccessful simulations
            successful_sims = results_df.dropna(subset=['winner'])
            failed_sims = results_df[results_df['winner'].isna()]
            
            # Create heatmaps for up to 3 successful simulations
            for i, (_, sim_row) in enumerate(successful_sims.head(3).iterrows()):
                sim_id = sim_row['sim_id']
                heatmap_path = output_dir / f"voting_heatmap_sim_{sim_id}.png"
                plot_voting_heatmap(detailed_results_json, heatmap_path, 
                                   simulation_id=sim_id, max_rounds=10)
            
            # Create a heatmap for one unsuccessful simulation if available
            if not failed_sims.empty:
                sim_id = failed_sims.iloc[0]['sim_id']
                heatmap_path = output_dir / f"voting_heatmap_failed_sim_{sim_id}.png"
                plot_voting_heatmap(detailed_results_json, heatmap_path, 
                                   simulation_id=sim_id, max_rounds=10)
        
        log.info(f"Simulation dashboard created in {output_dir}")
        
    except Exception as e:
        log.error(f"Error creating simulation dashboard: {e}")


# Function to generate a summary of simulation parameters and their impact
def plot_parameter_impact_matrix(
    base_results_path: Path,
    param_var_results_paths: Dict[str, Path],
    output_path: Path,
    metrics: List[str] = ['success_rate', 'avg_rounds']
) -> None:
    """Creates a matrix visualization comparing parameter impacts across simulations.

    Args:
        base_results_path: Path to the baseline simulation results stats JSON.
        param_var_results_paths: Dictionary mapping parameter names to their results stats JSON paths.
        output_path: Path to save the visualization.
        metrics: List of metrics to compare (e.g., 'success_rate', 'avg_rounds').
    """
    try:
        # Load baseline results
        with open(base_results_path, 'r') as f:
            base_stats = json.load(f)
        
        # Prepare data for the visualization
        data = []
        
        # Add baseline
        baseline_row = {'Parameter': 'Baseline'}
        for metric in metrics:
            # Map metrics to their actual keys in the stats JSON
            if metric == 'success_rate':
                baseline_row[metric] = base_stats.get('success_rate', 0) * 100  # Convert to percentage
            elif metric == 'avg_rounds':
                baseline_row[metric] = base_stats.get('average_rounds_successful', 0)
            elif metric == 'avg_rounds_all':
                baseline_row[metric] = base_stats.get('average_rounds_all', 0)
            else:
                baseline_row[metric] = base_stats.get(metric, 0)
        data.append(baseline_row)
        
        # Add parameter variations
        for param_name, result_path in param_var_results_paths.items():
            with open(result_path, 'r') as f:
                param_stats = json.load(f)
            
            param_row = {'Parameter': param_name}
            for metric in metrics:
                # Map metrics to their actual keys in the stats JSON
                if metric == 'success_rate':
                    param_row[metric] = param_stats.get('success_rate', 0) * 100  # Convert to percentage
                elif metric == 'avg_rounds':
                    param_row[metric] = param_stats.get('average_rounds_successful', 0)
                elif metric == 'avg_rounds_all':
                    param_row[metric] = param_stats.get('average_rounds_all', 0)
                else:
                    param_row[metric] = param_stats.get(metric, 0)
            data.append(param_row)
        
        # Create DataFrame for visualization
        df = pd.DataFrame(data)
        
        # Set up the figure
        plt.figure(figsize=(12, len(df) * 1.2))
        sns.set_style("whitegrid")
        
        # Create the heatmap
        cmap = sns.diverging_palette(240, 10, as_cmap=True)
        
        # Prepare the data for the heatmap with normalized values
        heatmap_data = df.set_index('Parameter')
        
        # Normalize by column (0-1 scale for each metric)
        normalized_data = heatmap_data.copy()
        for col in normalized_data.columns:
            col_min = normalized_data[col].min()
            col_max = normalized_data[col].max()
            if col_max > col_min:  # Avoid division by zero
                normalized_data[col] = (normalized_data[col] - col_min) / (col_max - col_min)
        
        # Plot heatmap with the original values as annotations
        ax = sns.heatmap(normalized_data, annot=heatmap_data, fmt='.2f', 
                       cmap=cmap, cbar=False, linewidths=.5)
        
        # Set plot labels and title
        plt.title('Parameter Impact Comparison', fontsize=16)
        plt.xlabel('Metric', fontsize=14)
        plt.ylabel('Parameter', fontsize=14)
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            log.info(f"Parameter impact matrix saved to: {output_path}")
        except Exception as e:
            log.error(f"Error saving plot: {e}")
        plt.close()
        
    except Exception as e:
        log.error(f"Error creating parameter impact matrix: {e}")


# Main function to run all visualizations
def generate_all_visualizations(
    results_csv: Path,
    output_dir: Path,
    detailed_results_json: Optional[Path] = None,
    parameter_comparisons: Optional[Dict[str, Path]] = None
) -> None:
    """Generates all available visualizations for the simulation results.

    Args:
        results_csv: Path to the simulation results CSV file.
        output_dir: Directory to save all visualization outputs.
        detailed_results_json: Optional path to detailed results JSON with round-by-round data.
        parameter_comparisons: Optional dictionary of parameter variation results for comparison.
    """
    # Create the dashboard of standard visualizations
    create_simulation_dashboard(results_csv, output_dir, detailed_results_json)
    
    # Create parameter impact matrix if comparison data is provided
    if parameter_comparisons and len(parameter_comparisons) > 0:
        param_impact_path = output_dir / "parameter_impact_matrix.png"
        # Get the first path as the baseline
        baseline_path = next(iter(parameter_comparisons.values()))
        plot_parameter_impact_matrix(baseline_path, parameter_comparisons, param_impact_path)
    
    log.info(f"All visualizations generated in {output_dir}")
