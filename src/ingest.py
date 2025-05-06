import pandas as pd
from typing import Optional, Dict
import os # Import os for path checking

def load_elector_data(file_path: str, schema: Optional[Dict[str, type]] = None) -> pd.DataFrame:
    """Loads elector roster data from a specified CSV file.

    Supports loading data from CSV and optionally validating it
    against a provided schema.

    Args:
        file_path: The path to the data file (e.g., 'data/electors.csv').
        schema: An optional dictionary defining expected columns and data types
                for validation.

    Returns:
        A pandas DataFrame containing the elector data.

    Raises:
        FileNotFoundError: If the specified file_path does not exist.
        ValueError: If data validation against the schema fails or file is not CSV.
        pd.errors.EmptyDataError: If the CSV file is empty.
        Exception: For other potential pandas read_csv errors.
    """
    print(f"Attempting to load elector data from: {file_path}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: Elector data file not found at {file_path}")

    if not file_path.lower().endswith('.csv'):
        raise ValueError(f"Error: File path does not point to a CSV file: {file_path}")

    try:
        elector_df = pd.read_csv(file_path)
    except pd.errors.EmptyDataError:
        raise ValueError(f"Error: CSV file is completely empty: {file_path}")
    except Exception as e:
        # Catch other potential pandas errors during read
        raise Exception(f"Error reading CSV file {file_path}: {e}")

    # AI: Check if DataFrame is empty *after* loading (e.g., only header)
    if elector_df.empty:
        raise ValueError(f"Error: CSV file '{file_path}' contains no data (only header or empty rows).")

    print(f"Successfully loaded data from {file_path}")

    # TODO: Implement schema validation if schema is provided
    if schema:
        print("Schema validation requested but not implemented.")
        # raise NotImplementedError("Schema validation not yet implemented.")

    return elector_df


# # AI: Test execution block
if __name__ == "__main__":
    sample_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'electors_sample.csv')
    print(f"--- Testing load_elector_data with {sample_file} ---")
    try:
        df = load_elector_data(sample_file)
        print("--- Loaded DataFrame ---:")
        print(df)
        print("------------------------")
    except FileNotFoundError as e:
        print(e)
    except ValueError as e:
        print(e)
    except pd.errors.EmptyDataError as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    print("--- Test complete --- ")
