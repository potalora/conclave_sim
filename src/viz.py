import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

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
        print("Ideology scores series is empty. Skipping plot.")
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
        print(f"Ideology distribution plot saved to: {output_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    plt.close() # Close the plot to free memory
