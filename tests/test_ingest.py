# # AI: Unit tests for the ingest module

import pytest
import pandas as pd
import os

# Assuming the test is run from the root directory or pytest handles paths
# Adjust the import path if necessary based on your test runner setup
from src.ingest import load_elector_data

# Define the path to the sample data relative to the test file or project root
# This assumes tests are run from the project root directory
SAMPLE_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
SAMPLE_CSV = os.path.join(SAMPLE_DATA_DIR, 'electors_sample.csv')
NON_EXISTENT_CSV = os.path.join(SAMPLE_DATA_DIR, 'non_existent.csv')
EMPTY_CSV = os.path.join(SAMPLE_DATA_DIR, 'empty_sample.csv') # Need to create this
NOT_CSV = os.path.join(SAMPLE_DATA_DIR, 'not_a_csv.txt') # Need to create this

# --- Test Fixtures (Optional but good practice) ---
@pytest.fixture(scope='module')
def sample_elector_file():
    """Provides the path to the valid sample elector CSV file."""
    # Create dummy empty and non-csv files for testing errors
    with open(EMPTY_CSV, 'w') as f:
        f.write('elector_id,name,region,ideology_score\n') # Header only
    with open(NOT_CSV, 'w') as f:
        f.write('This is not a CSV file.')

    yield SAMPLE_CSV # Provide the valid file path to tests

    # Teardown: Clean up created dummy files
    os.remove(EMPTY_CSV)
    os.remove(NOT_CSV)

# --- Test Cases ---

def test_load_elector_data_success(sample_elector_file):
    """Tests successful loading of elector data from a valid CSV."""
    elector_df = load_elector_data(sample_elector_file)
    assert isinstance(elector_df, pd.DataFrame)
    assert not elector_df.empty
    # Check for expected columns (adjust if schema changes)
    expected_columns = ['elector_id', 'name', 'region', 'ideology_score']
    assert all(col in elector_df.columns for col in expected_columns)
    # Check number of records (based on electors_sample.csv)
    assert len(elector_df) == 7

def test_load_elector_data_file_not_found():
    """Tests FileNotFoundError when the CSV file does not exist."""
    with pytest.raises(FileNotFoundError):
        load_elector_data(NON_EXISTENT_CSV)

def test_load_elector_data_not_csv(sample_elector_file): # Fixture needed for cleanup
    """Tests ValueError when the file is not a CSV."""
    # AI: Updated match pattern to align with actual error message
    with pytest.raises(ValueError, match="Error: File path does not point to a CSV file:"): # Check error message
        load_elector_data(NOT_CSV)

def test_load_elector_data_empty_csv(sample_elector_file): # Fixture needed for cleanup
    """Tests ValueError when the CSV file is empty (only header)."""
    # AI: Updated match pattern to align with actual error message
    with pytest.raises(ValueError, match="Error: CSV file .* contains no data"): # Check error message
        load_elector_data(EMPTY_CSV)

# TODO: Add tests for schema validation once implemented
