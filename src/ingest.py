import pandas as pd
from typing import Optional, Dict, List
import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re # Import regex module
import datetime # Import datetime
import numpy as np # Import numpy for merging/consolidation
import json
import logging
import subprocess
import sys

# Constants
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
GCATHOLIC_OUTPUT_PATH = os.path.join(DATA_DIR, 'scraped_gcatholic_raw.csv')
CH_OUTPUT_PATH = os.path.join(DATA_DIR, 'scraped_ch_raw.csv')
LLM_MATCHES_PATH = os.path.join(DATA_DIR, 'llm_matched_pairs.json')
MERGED_ELECTORS_PATH = os.path.join(DATA_DIR, 'merged_electors.csv')

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


def scrape_gcatholic_roster(url: str = "https://gcatholic.org/hierarchy/cardinals-alive-age.htm") -> Optional[pd.DataFrame]:
    """Scrapes the roster of living cardinals from GCatholic.org.

    Args:
        url: The URL of the GCatholic cardinals page.

    Returns:
        A Pandas DataFrame containing the scraped cardinal data, or None if scraping fails.
        The DataFrame includes headers derived from the table and extracted data rows.
    Raises:
        requests.exceptions.RequestException: If the HTTP request fails.
    """
    print(f"Attempting to scrape GCatholic roster from: {url}")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an exception for bad status codes
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL {url}: {e}")
        return None

    soup = BeautifulSoup(response.content, 'html.parser')

    # AI: --- Updated Table Finding Logic --- 
    target_table = None
    all_tables = soup.find_all('table')
    print(f"Found {len(all_tables)} tables on the page. Checking headers...")
    
    # Define expected header substrings (flexible matching)
    expected_header_parts = ['Cardinal', 'Age', 'Birth'] 

    for table in all_tables:
        first_row = table.find('tr')
        if not first_row:
            continue
        
        header_cells = first_row.find_all(['th', 'td']) # Check both th and td
        if not header_cells:
            continue
            
        actual_headers = [cell.get_text(strip=True) for cell in header_cells]
        
        # Check if the actual headers contain the expected parts
        # This check needs to be robust - ensure all parts are found
        header_match_count = 0
        for expected_part in expected_header_parts:
            if any(expected_part in actual_header for actual_header in actual_headers):
                header_match_count += 1
                
        if header_match_count == len(expected_header_parts):
            target_table = table
            print(f"Found target table with matching headers: {actual_headers}")
            break # Stop after finding the first matching table
        # else: 
            # Optional: print headers of non-matching tables for debugging
            # print(f"  Skipping table with headers: {actual_headers}")

    # Replace the old table finding logic with the result
    table = target_table 
    # --- End Updated Table Finding Logic ---
    
    if not table:
        print("Error: Could not find the target table on the page.")
        return None
    print("Successfully found the cardinals table.")

    cardinals_data: List[Dict[str, str]] = []
    headers = []

    # Extract headers from the first row (assuming it uses <th>)
    header_row = table.find('tr')
    if header_row:
        th_cells = header_row.find_all('th')
        if th_cells:
            headers = [th.get_text(strip=True) for th in th_cells]
            print(f"Found headers: {headers}")
        else:
             # Fallback: if no <th>, try using <td> in the first row as headers
             td_cells = header_row.find_all('td')
             if td_cells:
                 headers = [td.get_text(strip=True) for td in td_cells]
                 print(f"Found headers (using td fallback): {headers}")
             else:
                print("Warning: Could not find header cells (th or td) in the first row.")
                # Decide if processing should continue without headers or stop
                # For now, let's stop if headers are crucial
                return None 
    else:
        print("Warning: Could not find the header row (tr) in the table.")
        return None

    # Extract data rows (skip the header row)
    data_rows = table.find_all('tr')[1:] 
    print(f"Found {len(data_rows)} data rows.")
    for row in data_rows:
        cells = row.find_all('td')
        if len(cells) == len(headers):
            row_data = {headers[i]: cells[i].get_text(strip=True) for i in range(len(cells))}
            cardinals_data.append(row_data)
        # else: 
            # Optional: Log or handle rows with mismatched cell counts
            # print(f"Skipping row with {len(cells)} cells (expected {len(headers)}): {[c.get_text(strip=True) for c in cells]}")

    print(f"Successfully scraped {len(cardinals_data)} records.")
    if cardinals_data:
        print("First 2 records:")
        print(cardinals_data[:2])
        return pd.DataFrame(cardinals_data) # AI: Return DataFrame
    else:
        print("Warning: No data records were successfully scraped.")
        return None


def scrape_catholic_hierarchy_roster(url: str = "https://www.catholic-hierarchy.org/bishop/scardc3.html") -> Optional[pd.DataFrame]:
    """Scrapes the roster of living cardinal electors from Catholic-Hierarchy.org.

    Args:
        url: The URL of the Catholic Hierarchy cardinals page (scardc3.html recommended).

    Returns:
        A Pandas DataFrame containing the scraped elector cardinal data, or None if scraping fails.
        Includes dynamically identified headers and specific extraction for 'Name' and 'ProfileLink'.

    Raises:
        requests.exceptions.RequestException: If the HTTP request fails.
    """
    print(f"\nAttempting to scrape Catholic Hierarchy roster from: {url}")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL {url}: {e}")
        return None

    table = None
    for parser in ['lxml', 'html.parser']:
        print(f"Attempting to parse with {parser}...")
        try:
            soup = BeautifulSoup(response.content, parser)

            # AI: Target the first table found on the page (b0c2.html has only one)
            all_tables = soup.find_all('table')
            print(f"Found {len(all_tables)} tables on the page.")
            if all_tables:
                table = all_tables[0]
                print(f"Identified the first table as the target for inspection.")
                # AI: Inspect the structure (td count in 3rd row)
                all_rows = table.find_all('tr')
                if len(all_rows) > 2:
                    sample_row = all_rows[2]
                    cells = sample_row.find_all('td')
                    print(f"Inspection: First table's 3rd row (index 2) has {len(cells)} cells (td elements).")
                else:
                    print("Inspection: First table has 2 or fewer rows.")
            else:
                print("Error: No tables found on page.")
                continue # Try next parser if available
        
            # AI: Exit after inspecting with the first parser
            # break 

        except Exception as e:
            print(f"Error parsing with {parser}: {e}")
            # Continue, table might have been found by lxml
    
    # AI: --- Inspection Logic --- 
    if not table:
        print(f"Error: Could not find any table on {url} using available parsers.")
        return None
    
    all_rows = table.find_all('tr')

    # AI: --- Restore and Adapt Processing Logic --- 
    print("Successfully found the target table (assumed first table).")
    cardinals_data: List[Dict[str, str]] = [] 
    headers = []
    if all_rows:
        # AI: Dynamically extract headers from <th> elements in the first row
        header_cells = all_rows[0].find_all('th')
        if header_cells:
            headers = [th.get_text(strip=True) for th in header_cells]
            print(f"Dynamically identified headers: {headers}")
        else:
            print("Warning: Could not find header row (<th> elements) in the first row. Cannot process.")
            return None
    else:
        print("Warning: Table has no rows.")
        return None

    expected_columns = len(headers)
    if expected_columns == 0:
        print("Error: Header identification resulted in zero headers.")
        return None

    # AI: Find the index of the 'Name' column header
    try:
        name_column_index = headers.index('Name')
        print(f"Found 'Name' header at index {name_column_index}.")
    except ValueError:
        print("Error: Could not find 'Name' column header. Cannot extract name/link.")
        return None

    print(f"Expecting {expected_columns} columns based on headers.")

    # AI: Skip header row (index 0) and process the rest
    data_rows_processed = 0
    # AI: Iterate starting from the second row (index 1)
    for row in all_rows[1:]:
        cells = row.find_all('td')
        # AI: Check for correct number of cells based on dynamic headers
        if len(cells) == expected_columns: 
            data_rows_processed += 1
            # AI: Assign raw data first based on dynamic headers
            row_data = {headers[i]: cells[i].get_text(strip=True) for i in range(expected_columns)}
            
            # AI: Extract name and link from the cell at the 'Name' column index
            name_cell = cells[name_column_index]
            name_link_tag = name_cell.find('a')
            if name_link_tag:
                row_data['Name'] = name_link_tag.get_text(strip=True)
                # AI: Construct absolute URL if relative
                link = name_link_tag.get('href')
                if link:
                    # Use urljoin for robust path combination
                    row_data['ProfileLink'] = urljoin(url, link) 
                else:
                    row_data['ProfileLink'] = None
            else:
                 row_data['Name'] = name_cell.get_text(strip=True) # Fallback if no link in name cell
                 row_data['ProfileLink'] = None

            cardinals_data.append(row_data)
        # else:
             # AI: Log skipped rows for debugging if needed
             # print(f"Skipping row with {len(cells)} cells (expected {expected_columns}): {[c.get_text(strip=True) for c in cells]}")

    # AI: Print count of rows actually processed
    print(f"Processed {data_rows_processed} rows matching the {expected_columns}-column data structure.")

    if not cardinals_data:
        print("Warning: Data extraction resulted in an empty list (check table selector and structure).")

    return pd.DataFrame(cardinals_data) # AI: Return DataFrame


def process_scraped_data(
    scraped_gc_path: str = GCATHOLIC_OUTPUT_PATH,
    scraped_ch_path: str = CH_OUTPUT_PATH,
    matches_path: str = LLM_MATCHES_PATH,
    output_path: str = MERGED_ELECTORS_PATH
) -> Optional[pd.DataFrame]:
    """Processes scraped data using LLM matches, merges, and saves the result.

    Args:
        scraped_gc_path: Path to the raw scraped GCatholic CSV file.
        scraped_ch_path: Path to the raw scraped Catholic Hierarchy CSV file.
        matches_path: Path to the JSON file containing LLM-matched gc_id/ch_id pairs.
        output_path: Path to save the final merged elector CSV file.

    Returns:
        The merged DataFrame, or None if processing fails.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting data processing using LLM matches...")

    # 1. Load Raw Data
    try:
        df_gc_raw = pd.read_csv(scraped_gc_path)
        logger.info(f"Loaded raw GCatholic data: {df_gc_raw.shape}")
    except FileNotFoundError:
        logger.error(f"Raw GCatholic file not found: {scraped_gc_path}")
        return None
    except Exception as e:
        logger.error(f"Error loading raw GCatholic file: {e}")
        return None

    try:
        df_ch_raw = pd.read_csv(scraped_ch_path)
        logger.info(f"Loaded raw Catholic Hierarchy data: {df_ch_raw.shape}")
    except FileNotFoundError:
        logger.error(f"Raw Catholic Hierarchy file not found: {scraped_ch_path}")
        return None
    except Exception as e:
        logger.error(f"Error loading raw Catholic Hierarchy file: {e}")
        return None

    # 2. Load LLM Matches
    try:
        with open(matches_path, 'r') as f:
            matches = json.load(f)
        df_matches = pd.DataFrame(matches)
        if not all(col in df_matches.columns for col in ['gc_id', 'ch_id']):
             raise ValueError("Matches JSON missing 'gc_id' or 'ch_id'")
        logger.info(f"Loaded {len(df_matches)} LLM matches from {matches_path}")
    except FileNotFoundError:
        logger.error(f"LLM matches file not found: {matches_path}. Run src/match_names.py first.")
        return None
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from {matches_path}.")
        return None
    except Exception as e:
        logger.error(f"Error loading LLM matches: {e}")
        return None

    # 3. Preprocess Raw Dataframes (Minimal)
    # Add IDs based on original index (needed for matching)
    df_gc_raw['gc_id'] = df_gc_raw.index
    df_ch_raw['ch_id'] = df_ch_raw.index

    # Rename columns for clarity before merge
    df_gc_raw = df_gc_raw.rename(columns={'Cardinal Electors(135)': 'gc_description', 'Age': 'gc_age', 'Date ofBirth': 'gc_birthdate'})
    df_ch_raw = df_ch_raw.rename(columns={'Name': 'ch_name', 'Age': 'ch_age', 'Birthdate': 'ch_birthdate', 'Elevated': 'ch_elevated_date', 'Current Title': 'ch_title'})

    # Apply age filter to GCatholic (Age < 80)
    df_gc_raw['gc_age'] = pd.to_numeric(df_gc_raw['gc_age'], errors='coerce')
    df_gc_filtered = df_gc_raw[df_gc_raw['gc_age'] < 80].copy()
    logger.info(f"Filtered GCatholic electors (Age < 80): {len(df_gc_filtered)}")

    # Select necessary columns before merge
    gc_cols = ['gc_id', 'gc_description', 'gc_age', 'gc_birthdate']
    ch_cols = ['ch_id', 'ch_name', 'ch_age', 'ch_birthdate', 'ch_elevated_date', 'ch_title']
    df_gc_to_merge = df_gc_filtered[gc_cols]
    df_ch_to_merge = df_ch_raw[ch_cols]

    # 4. Merge using LLM Matches
    logger.info(f"Merging GC ({len(df_gc_to_merge)}) and CH ({len(df_ch_to_merge)}) using {len(df_matches)} matches...")

    # Merge GC with matches
    df_merged = pd.merge(df_matches, df_gc_to_merge, on='gc_id', how='inner')
    logger.info(f"Shape after merging matches with GC: {df_merged.shape}")

    # Merge the result with CH
    df_merged = pd.merge(df_merged, df_ch_to_merge, on='ch_id', how='inner')
    logger.info(f"Shape after merging with CH: {df_merged.shape}")

    if len(df_merged) == 0:
        logger.warning("Merge resulted in an empty DataFrame. Check matches and preprocessing.")
        return None
    elif len(df_merged) < len(df_matches):
         logger.warning(f"Merge resulted in {len(df_merged)} rows, but expected {len(df_matches)}. Some matches might not have corresponding raw data after filtering.")

    # 5. Final Processing & Column Selection
    # Consolidate information - prioritize CH data where available, use GC as fallback
    # Example: Use CH name and title, GC age/birthdate if CH is missing/different format
    # For now, let's just select key columns from both for inspection

    # Standardize Name from CH (likely cleaner now)
    # Basic split, assuming 'FirstName MiddleName Cardinal LastName'
    def extract_std_name(ch_name):
        if not isinstance(ch_name, str):
            return None
        parts = ch_name.replace('Cardinal', '').strip().split()
        if len(parts) >= 2:
            return f"{parts[-1]}, {parts[0]}" # LastName, FirstName
        elif len(parts) == 1:
            return parts[0] # Fallback
        return None

    df_merged['name_standardized'] = df_merged['ch_name'].apply(extract_std_name)
    missing_names = df_merged['name_standardized'].isnull().sum()
    if missing_names > 0:
        logger.warning(f"{missing_names} rows failed standardized name extraction from CH name.")

    # Select final columns
    final_cols = [
        'name_standardized', # Primary identifier
        'gc_age',             # Use GC age (already filtered)
        'gc_birthdate',
        'ch_elevated_date',
        'ch_title',
        'ch_name',            # Keep original CH name for reference
        'gc_description',     # Keep original GC desc for reference
        'gc_id',              # Keep IDs for traceability
        'ch_id'
    ]
    # Filter for columns that actually exist after merge
    final_cols_present = [col for col in final_cols if col in df_merged.columns]
    df_final = df_merged[final_cols_present].copy()

    # Rename for consistency
    df_final = df_final.rename(columns={
        'name_standardized': 'Name',
        'gc_age': 'Age',
        'gc_birthdate': 'Birthdate_GC',
        'ch_elevated_date': 'Elevated',
        'ch_title': 'Title'
    })

    logger.info(f"Final merged dataset shape: {df_final.shape}")

    # 6. Save Processed Data
    try:
        df_final.to_csv(output_path, index=False)
        logger.info(f"Successfully saved merged elector data to {output_path}")
    except Exception as e:
        logger.error(f"Error saving merged data: {e}")
        return None

    return df_final


# --- Main Execution Logic (Example) ---
# This part is typically called from __main__.py or another orchestrator script
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logger = logging.getLogger(__name__)
    logger.info("Running ingest script directly...")

    # 1. Scrape GCatholic data
    gc_data = scrape_gcatholic_roster()
    if gc_data is not None: # AI: Correct boolean check for DataFrame
        # df_gc = pd.DataFrame(gc_data) # No longer needed, function returns DF
        gc_raw_output_path = os.path.join(DATA_DIR, 'scraped_gcatholic_raw.csv')
        try:
            gc_data.to_csv(gc_raw_output_path, index=False) # Use gc_data directly
            logger.info(f"GCatholic raw data saved to {gc_raw_output_path}")
        except Exception as e:
            logger.error(f"Failed to save GCatholic raw data: {e}")
    else:
        logger.warning("Failed to scrape GCatholic data.")

    # 2. Scrape Catholic Hierarchy data
    ch_data = scrape_catholic_hierarchy_roster()
    if ch_data is not None: # AI: Correct boolean check for DataFrame
        # df_ch = pd.DataFrame(ch_data) # No longer needed, function returns DF
        ch_raw_output_path = os.path.join(DATA_DIR, 'scraped_ch_raw.csv')
        try:
            ch_data.to_csv(ch_raw_output_path, index=False) # Use ch_data directly
            logger.info(f"Catholic Hierarchy raw data saved to {ch_raw_output_path}")
        except Exception as e:
            logger.error(f"Failed to save Catholic Hierarchy raw data: {e}")
    else:
        logger.warning("Failed to scrape Catholic Hierarchy data.")

    # 3. Run LLM Matching (only if matches don't exist)
    match_file_path = os.path.join(DATA_DIR, 'llm_matched_pairs.json')
    if not os.path.exists(match_file_path):
        logger.info("LLM match file not found. Running LLM matching script (match_names.py)... Ensure GOOGLE_API_KEY is set.")
        match_script_path = os.path.join(os.path.dirname(__file__), 'match_names.py')
        try:
            result = subprocess.run(['python', match_script_path], capture_output=True, text=True, check=True, cwd=os.path.dirname(__file__))
            logger.info(f"match_names.py stdout:\n{result.stdout}")
            if result.stderr:
                logger.warning(f"match_names.py stderr:\n{result.stderr}")
            logger.info("LLM matching script completed.")
        except FileNotFoundError:
            logger.error(f"Error: The script '{match_script_path}' was not found.")
            # Decide how to handle this: exit, raise, etc.
            sys.exit(1)
        except subprocess.CalledProcessError as e:
            logger.error(f"LLM matching script failed with exit code {e.returncode}.")
            logger.error(f"Stdout:\n{e.stdout}")
            logger.error(f"Stderr:\n{e.stderr}")
            # Decide how to handle this: exit, raise, etc.
            sys.exit(1) # Exit if matching fails
        except Exception as e:
             logger.error(f"An unexpected error occurred while running match_names.py: {e}")
             sys.exit(1)
    else:
        logger.info(f"Found existing LLM match file at {match_file_path}. Skipping LLM matching script.")

    # 4. Process data using matches
    logger.info("Processing data using LLM matches...")
    merged_df = process_scraped_data()

    if merged_df is not None:
        logger.info("Data processing successful.")
        # Display first 5 rows of the merged data
        # print("\n--- Merged Elector Data (First 5 Rows) ---")
        # print(merged_df.head().to_markdown(index=False))
        # print("\n")
        # print(merged_df.info())
    else:
        logger.error("Data processing failed.")

    logger.info("Ingest script finished.")
