#!/usr/bin/env python
"""
Focused test script to debug transition probability calculation issues.
This isolates the TransitionModel.calculate_transition_probabilities method to diagnose
why probability sums are not equaling 1.0.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from src.model import TransitionModel
from src.ingest import ElectorDataIngester

# Configure logging to see critical/warnings/errors
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("test_probability_calc")

def main():
    log.info("Starting targeted test of probability calculations")
    
    # Load a small subset of elector data to avoid memory issues
    ingester = ElectorDataIngester()
    elector_data_path = Path("data/electors_2013.csv")  # Adjust if needed
    
    if not elector_data_path.exists():
        log.error(f"Elector data file not found: {elector_data_path}")
        return
        
    # Load elector data
    log.info(f"Loading elector data from {elector_data_path}")
    elector_data = ingester.load_elector_data(elector_data_path)
    
    # Take only a small subset to avoid memory issues
    if len(elector_data) > 20:
        log.info(f"Using only first 20 electors out of {len(elector_data)} for testing")
        elector_data = elector_data.iloc[:20]
    
    log.info(f"Loaded {len(elector_data)} electors for testing")
    
    # Initialize model with default parameters
    log.info("Initializing TransitionModel")
    model = TransitionModel(
        elector_data,
        initial_beta_weight=1.0,
        enable_dynamic_beta=True,
        beta_growth_rate=1.1,
        stickiness_factor=0.3,
        bandwagon_strength=0.1,
        regional_affinity_bonus=0.1,
        papabile_weight_factor=1.5
    )
    
    # Create dummy previous votes (everyone votes for themselves initially)
    previous_votes = pd.Series(
        index=elector_data.index,
        data=elector_data.index
    )
    
    # Run the calculation that's causing issues
    log.info("Calculating transition probabilities")
    probabilities, details = model.calculate_transition_probabilities(
        previous_round_votes=previous_votes,
        current_round_num=99  # Match the round number in our diagnostic code
    )
    
    # Check the probabilities directly
    log.info("Checking probability sums for each elector:")
    for i in range(len(elector_data)):
        elector_id = elector_data.index[i]
        prob_sum = np.sum(probabilities[i, :])
        log.info(f"Elector {elector_id} (idx {i}): Probability sum = {prob_sum:.17f}")
        if not np.isclose(prob_sum, 1.0):
            log.warning(f"!! Elector {elector_id} (idx {i}): Probability sum {prob_sum:.17f} is not close to 1.0")
    
    log.info("Test complete")

if __name__ == "__main__":
    main()
