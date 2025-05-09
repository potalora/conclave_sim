# Windsurf Conclave Simulation

A Monte Carlo simulation engine for modeling papal conclaves and forecasting outcomes based on cardinal elector dynamics, voting rules, and voting behavior models.

## Background

This project simulates the process of a papal conclave, where 135 cardinal electors gather to elect a new Pope through a series of voting rounds. The model incorporates ideological preferences, regional affinity, papabile status, bandwagon effects, and vote stickiness to create realistic voting dynamics.

## Installation

```bash
# Clone the repository
git clone https://github.com/potalora/conclave_sim.git
cd conclave_sim

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running Simulations

To run a simulation with default parameters:

```bash
python -m src.simulate
```

To run a simulation with optimized parameters for faster convergence:

```bash
python -m src.simulate \
--num-simulations 100 \
--max-rounds 20 \
--enable-dynamic-beta \
--beta-increment-amount 0.3 \
--beta-increment-interval-rounds 1 \
--initial-beta-weight 1.5 \
--bandwagon-strength 0.7 \
--papabile-weight-factor 2.0 \
--stickiness-factor 0.8 \
--supermajority-threshold 0.56
```

### Generating Visualizations

After running simulations, you can generate visualizations using:

```bash
python generate_visualizations.py
```

Visualization outputs will be saved to the `data/plots` directory.

## Recent Simulation Results

### Optimized Parameters (May 2025)

Our most recent simulation run with 1000 iterations achieved significantly improved convergence rates using the following parameters:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `initial_beta_weight` | 1.5 | Controls sensitivity to ideological distance |
| `beta_increment_amount` | 0.3 | Increases beta weight per round to promote convergence |
| `beta_increment_interval_rounds` | 1 | How often beta increments |
| `bandwagon_strength` | 0.7 | Strength of tendency to vote for front-runners |
| `papabile_weight_factor` | 2.0 | Multiplicative advantage for "papabile" cardinals |
| `stickiness_factor` | 0.8 | Tendency of electors to maintain previous votes |
| `supermajority_threshold` | 0.56 | Required threshold to elect a pope (56%) |

#### Key Results

- **Success Rate**: 89% of simulations reached a decisive outcome
- **Average Rounds**: 10.5 rounds for successful simulations
- **Convergence Range**: 6-20 rounds

Visualizations of these results can be found in the `data/plots/1000_fast` directory after running the visualization script.

## Project Structure

```bash
/ (root)
├─ data/                  # Raw and processed elector CSVs, simulation results
│  ├─ plots/              # Generated visualizations
├─ src/                   # Core simulation modules
│  ├─ ingest.py           # Elector roster loader
│  ├─ model.py            # Preference & transition models
│  ├─ simulate.py         # Monte Carlo engine
│  └─ viz.py              # Plotting and reporting
├─ tests/                 # Pytest suites and test data fixtures
├─ docs/                  # Markdown documentation
└─ README.md              # Project overview
```

## Model Parameters

The simulation model supports various configurable parameters:

| Parameter | Description |
|-----------|-------------|
| `initial_beta_weight` | Controls sensitivity to ideological distance |
| `beta_increment_amount` | Increases beta weight per round to promote convergence |
| `bandwagon_strength` | Strength of tendency to vote for front-runners |
| `papabile_weight_factor` | Multiplicative advantage for "papabile" cardinals |
| `stickiness_factor` | Tendency of electors to maintain previous votes |
| `regional_bonus` | Bonus for candidates from the same region |
| `supermajority_threshold` | Required threshold to elect a pope |

## Features

- Multi-round papal election simulation with configurable parameters
- Ideological distance-based preference modeling
- Regional affinity effects
- Papabile status weighting
- Bandwagon effects and vote stickiness
- Comprehensive visualization tools
- Parallelized simulation for performance

## Visualizations

The visualization module (`src/viz.py`) provides several visualizations for simulation results:

- **Rounds Distribution**: Histogram showing how many rounds it takes for simulations to converge
- **Winner Distribution**: Bar chart showing which electors won and how frequently
- **Convergence Timeline**: Visual mapping of when each simulation reached consensus
- **Parameter Impact Matrix**: Heatmap comparing parameter sets and their effects

## Contributing

Contributions to the Conclave Simulation project are welcome. Please follow these steps to contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/my-feature`)
5. Create a new Pull Request

Please ensure your code follows the project's coding standards outlined in the repository documentation.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
