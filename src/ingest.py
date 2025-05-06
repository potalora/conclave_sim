# AI: Refactored src/ingest.py
"""
Handles the ingestion pipeline for cardinal elector data:
1. Scrapes raw cardinal rosters from GCatholic and Catholic Hierarchy.
2. Optionally triggers the LLM matching script ([match_names.py](cci:7://file:///Users/potalora/ai_workspace/conlave_simulation/src/match_names.py:0:0-0:0)) if matches don't exist.
3. Loads raw data and LLM-generated matches.
4. Preprocesses, merges, and standardizes the data.
5. Saves the final merged elector dataset.
"""

import pandas as pd
from typing import Optional, Dict, List, Tuple
import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import datetime
import numpy as np
import json
import logging
import subprocess
import sys
from pathlib import Path
import importlib # To dynamically import match_names
from datetime import datetime # AI: Add datetime import

# --- Constants ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SRC_DIR = BASE_DIR / "src"
GCATHOLIC_RAW_PATH = DATA_DIR / "scraped_gcatholic_raw.csv"
CH_RAW_PATH = DATA_DIR / "scraped_ch_raw.csv"
LLM_MATCHES_PATH = DATA_DIR / "llm_matched_pairs.json"
MERGED_ELECTORS_PATH = DATA_DIR / "merged_electors.csv"
MATCH_NAMES_SCRIPT_PATH = SRC_DIR / "match_names.py"

# --- Logging Setup ---
# AI: Using standard logging instead of just print
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# --- Scraping Functions ---

def scrape_gcatholic_roster(url: str = "https://gcatholic.org/hierarchy/cardinals-alive-age.htm",
                            output_path: Path = GCATHOLIC_RAW_PATH) -> bool:
    """Scrapes the roster of living cardinals from GCatholic.org and saves to CSV.

    Args:
        url: The URL of the GCatholic cardinals page.
        output_path: The path to save the scraped data CSV.

    Returns:
        True if scraping and saving were successful, False otherwise.

    Raises:
        Prints error details to stderr via logging.
    """
    log.info(f"Attempting to scrape GCatholic roster from: {url}")
    try:
        response = requests.get(url, timeout=20) # Increased timeout
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        log.error(f"Error fetching GCatholic URL {url}: {e}", exc_info=True)
        return False

    try:
        soup = BeautifulSoup(response.content, 'lxml') # Use lxml parser

        target_table = None
        all_tables = soup.find_all('table')
        log.info(f"Found {len(all_tables)} tables on GCatholic page.")
        expected_header_parts = ['Cardinal', 'Age', 'Birth']

        for table in all_tables:
            first_row = table.find('tr')
            if not first_row: continue
            header_cells = first_row.find_all(['th', 'td'])
            if not header_cells: continue
            actual_headers = [cell.get_text(strip=True) for cell in header_cells]

            if sum(any(part in hdr for hdr in actual_headers) for part in expected_header_parts) == len(expected_header_parts):
                target_table = table
                log.info(f"Found target GCatholic table with headers: {actual_headers}")
                break

        if not target_table:
            log.error("Could not find the target table on the GCatholic page.")
            return False

        cardinals_data: List[Dict[str, str]] = []
        headers = []
        header_row = target_table.find('tr')

        if header_row:
            th_cells = header_row.find_all('th')
            if th_cells:
                headers = [th.get_text(strip=True) for th in th_cells]
            else: # Fallback to td
                td_cells = header_row.find_all('td')
                if td_cells: headers = [td.get_text(strip=True) for td in td_cells]

        if not headers:
             log.warning("Could not find header cells (th or td) in the GCatholic table first row.")
             # Attempt to proceed without strict header matching for rows
             # This might require index-based access later if needed
        else:
            log.info(f"GCatholic Headers identified: {headers}")

        data_rows = target_table.find_all('tr')[1:]
        log.info(f"Found {len(data_rows)} data rows in GCatholic table.")
        for row in data_rows:
            cells = row.find_all('td')
            if headers and len(cells) == len(headers):
                row_data = {headers[i]: cells[i].get_text(strip=True) for i in range(len(cells))}
                cardinals_data.append(row_data)
            elif not headers and cells: # If no headers, just grab cell text
                 row_data_list = [cell.get_text(strip=True) for cell in cells]
                 # Need a way to structure this - maybe placeholder keys?
                 row_data = {f"col_{i}": val for i, val in enumerate(row_data_list)}
                 cardinals_data.append(row_data)
                 if len(cardinals_data) == 1: # Log headers warning only once
                     log.warning("Processing GCatholic rows without identified headers. Using placeholder keys.")
            else:
                 log.debug(f"Skipping GCatholic row with mismatched cell count ({len(cells)} cells, {len(headers)} headers): {[c.get_text(strip=True) for c in cells]}")

        if not cardinals_data:
            log.warning("No data records were successfully scraped from GCatholic.")
            return False

        log.info(f"Successfully scraped {len(cardinals_data)} records from GCatholic.")
        df = pd.DataFrame(cardinals_data)

        # Save the raw data
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding='utf-8')
        log.info(f"GCatholic raw data saved to {output_path}")
        return True

    except Exception as e:
        log.error(f"Error parsing GCatholic HTML or saving data: {e}", exc_info=True)
        return False


def scrape_catholic_hierarchy_roster(url: str = "https://www.catholic-hierarchy.org/bishop/scardc3.html",
                                     output_path: Path = CH_RAW_PATH) -> bool:
    """Scrapes the roster of living cardinal electors from Catholic-Hierarchy.org and saves to CSV.

    Args:
        url: The URL of the Catholic Hierarchy cardinals page (scardc3.html recommended).
        output_path: The path to save the scraped data CSV.

    Returns:
        True if scraping and saving were successful, False otherwise.

    Raises:
         Prints error details to stderr via logging.
    """
    log.info(f"Attempting to scrape Catholic Hierarchy roster from: {url}")
    try:
        response = requests.get(url, timeout=20) # Increased timeout
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        log.error(f"Error fetching Catholic Hierarchy URL {url}: {e}", exc_info=True)
        return False

    try:
        soup = BeautifulSoup(response.content, 'lxml') # Use lxml parser

        # Find the specific table - relying on structure/content may be needed
        # Let's look for a table where the first row contains 'Birthdate' and 'Elevated'
        target_table = None
        all_tables = soup.find_all('table')
        log.info(f"Found {len(all_tables)} tables on Catholic Hierarchy page.")
        expected_header_parts = ['Birthdate', 'Elevated', 'Name']

        for table in all_tables:
            first_row = table.find('tr')
            if not first_row: continue
            header_cells = first_row.find_all(['th', 'td'])
            if not header_cells: continue
            actual_headers = [cell.get_text(strip=True).replace(':', '') for cell in header_cells] # Clean ':'

            if sum(any(part in hdr for hdr in actual_headers) for part in expected_header_parts) >= 2: # More flexible match
                target_table = table
                log.info(f"Found target Catholic Hierarchy table with headers: {actual_headers}")
                # Use these found headers
                headers = actual_headers
                break
        else: # If loop finishes without break
             headers = [] # Reset headers if no table found this way

        if not target_table:
            log.warning("Could not find the primary target table based on headers ('Birthdate', 'Elevated', 'Name').")
            # Fallback: Look for the first large table as a heuristic (may need adjustment)
            potential_tables = [tbl for tbl in all_tables if len(tbl.find_all('tr')) > 10]
            if potential_tables:
                 target_table = potential_tables[0]
                 log.info("Using fallback: Selecting the first table with > 10 rows.")
                 # Try to get headers from this fallback table
                 header_row = target_table.find('tr')
                 if header_row:
                     th_cells = header_row.find_all('th')
                     if th_cells: headers = [th.get_text(strip=True).replace(':', '') for th in th_cells]
                     else: # Fallback td
                         td_cells = header_row.find_all('td')
                         if td_cells: headers = [td.get_text(strip=True).replace(':', '') for td in td_cells]
                 if headers: log.info(f"Fallback table headers: {headers}")
                 else: log.warning("Could not determine headers for fallback table.")
            else:
                 log.error("No suitable table found on Catholic Hierarchy page using primary or fallback methods.")
                 return False

        cardinals_data: List[Dict[str, str]] = []
        data_rows = target_table.find_all('tr')[1:] # Skip header row
        log.info(f"Found {len(data_rows)} data rows in Catholic Hierarchy table.")

        # Validate header count before processing rows if headers were found
        expected_cols = len(headers) if headers else -1 # Use -1 if no headers found

        for row in data_rows:
            cells = row.find_all('td')
            if not cells: continue # Skip empty rows or separator rows

            row_values = [cell.get_text(strip=True) for cell in cells]

            # Extract name and link separately as they are often in the first cell's 'a' tag
            name = "N/A"
            profile_link = "N/A"
            first_cell_link = cells[0].find('a', href=True)
            if first_cell_link:
                name = first_cell_link.get_text(strip=True)
                profile_link = urljoin(url, first_cell_link['href']) # Make absolute URL
            elif cells: # Fallback if no link
                 name = cells[0].get_text(strip=True)


            row_data = {}
            # If headers match cell count, use headers as keys
            if headers and len(cells) == expected_cols:
                row_data = {headers[i]: row_values[i] for i in range(expected_cols)}
                # Overwrite 'Name' if we extracted it from link
                if 'Name' in headers: row_data['Name'] = name
                else: row_data['Name'] = name # Add if 'Name' wasn't a header
            # If headers don't match, or no headers, use placeholders but ensure Name is captured
            else:
                 row_data = {f"col_{i}": val for i, val in enumerate(row_values)}
                 row_data['Name'] = name # Ensure name is included
                 if len(cardinals_data) == 0: # Log only once
                      log.warning(f"Catholic Hierarchy row cell count ({len(cells)}) doesn't match headers ({expected_cols}) or no headers. Using placeholder keys.")

            row_data['ProfileLink'] = profile_link # Always add the profile link
            cardinals_data.append(row_data)


        if not cardinals_data:
            log.warning("No data records were successfully scraped from Catholic Hierarchy.")
            return False

        log.info(f"Successfully scraped {len(cardinals_data)} records from Catholic Hierarchy.")
        df = pd.DataFrame(cardinals_data)

        # AI: Manual Header Renaming based on observed CH structure (if placeholders were used)
        # This assumes a typical column order if headers failed. Adjust if needed.
        if not headers or len(headers) != df.shape[1] -1 : # -1 because we added ProfileLink
            log.warning("Attempting manual header assignment for Catholic Hierarchy data.")
            # Example: based on scardc3.html structure (Name, Birthdate, Elevated, Title, Age?)
            # Check actual number of columns BEFORE adding ProfileLink
            num_cols_before_link = df.shape[1] -1
            potential_headers = ['Name', 'Birthdate', 'Elevated', 'Current Title', 'Age', 'ProfileLink'] # Include link now
            if num_cols_before_link == 5: # Common case seen
                 rename_map = {
                     'col_0': 'Name_temp', # Will be overwritten by extracted Name
                     'col_1': 'Birthdate',
                     'col_2': 'Elevated',
                     'col_3': 'Current Title',
                     'col_4': 'Age',
                     'Name': 'Name' # Keep the extracted Name
                 }
                 # Only rename columns that actually exist from placeholder scraping
                 actual_rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
                 df.rename(columns=actual_rename_map, inplace=True)
                 # Ensure final 'Name' column exists correctly
                 if 'Name_temp' in df.columns and 'Name' in df.columns: del df['Name_temp']
                 # Reorder columns for consistency
                 final_cols_order = [h for h in potential_headers if h in df.columns]
                 df = df[final_cols_order]
                 log.info(f"Applied manual headers. Columns: {df.columns.tolist()}")
            else:
                log.warning(f"Cannot apply manual headers: unexpected number of columns ({num_cols_before_link}) found before adding ProfileLink.")


        # Save the raw data
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding='utf-8')
        log.info(f"Catholic Hierarchy raw data saved to {output_path}")
        return True

    except Exception as e:
        log.error(f"Error parsing Catholic Hierarchy HTML or saving data: {e}", exc_info=True)
        return False


# --- Data Processing Functions ---

def _load_raw_and_matches(gc_path: Path, ch_path: Path, matches_path: Path) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[List[Dict]]]:
    """Loads raw GC, raw CH, and the LLM matches JSON file."""
    log.info(f"Loading raw data from {gc_path} and {ch_path}")
    log.info(f"Loading LLM matches from {matches_path}")

    df_gc, df_ch, matches = None, None, None

    try:
        df_gc = pd.read_csv(gc_path)
    except Exception as e:
        log.error(f"Failed to load GC raw data: {e}", exc_info=True)

    try:
        df_ch = pd.read_csv(ch_path)
    except Exception as e:
        log.error(f"Failed to load CH raw data: {e}", exc_info=True)

    try:
        with open(matches_path, 'r', encoding='utf-8') as f:
            matches = json.load(f)
        if not isinstance(matches, list):
            log.error(f"LLM Matches file ({matches_path}) is not a JSON list.")
            matches = None # Invalidate if not list
    except FileNotFoundError:
         log.error(f"LLM Matches file not found: {matches_path}")
    except json.JSONDecodeError as e:
        log.error(f"Error decoding JSON from LLM matches file {matches_path}: {e}", exc_info=True)
    except Exception as e:
        log.error(f"Error reading LLM matches file {matches_path}: {e}", exc_info=True)

    if df_gc is None or df_ch is None or matches is None:
        log.error("Failed to load all required data/match inputs.")
        return None, None, None

    log.info(f"Loaded GC: {df_gc.shape}, CH: {df_ch.shape}, Matches: {len(matches)}")
    return df_gc, df_ch, matches


def _preprocess_gc_for_merge(df_gc: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses GCatholic data specifically for merging."""
    log.info("Preprocessing GCatholic data for merging...")
    df = df_gc.copy()
    # Rename for consistency (ensure these columns exist from scraping)
    rename_map = {'Cardinal Electors(135)': 'gc_description', 'Age': 'gc_age', 'Date ofBirth': 'gc_birthdate'}
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    # Ensure 'gc_id' exists or create from index if needed (should be created in match_names now)
    if 'gc_id' not in df.columns:
         log.warning("gc_id not found in loaded GC data, creating from index.")
         df['gc_id'] = df.index # Fallback

    # Filter for electors (< 80 years old)
    if 'gc_age' in df.columns:
        df['gc_age'] = pd.to_numeric(df['gc_age'], errors='coerce')
        original_count = len(df)
        df = df[df['gc_age'].fillna(999) < 80].copy()
        log.info(f"Filtered GCatholic electors by age (<80): {original_count} -> {len(df)}")
    else:
        log.warning("'gc_age' column not found, cannot filter GCatholic electors by age.")

    # Select necessary columns
    required_cols = ['gc_id', 'gc_description', 'gc_age', 'gc_birthdate']
    available_cols = [col for col in required_cols if col in df.columns]
    log.info(f"Selected GC columns for merge: {available_cols}")
    return df[available_cols]


def _preprocess_ch_for_merge(df_ch: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses Catholic Hierarchy data specifically for merging."""
    log.info("Preprocessing Catholic Hierarchy data for merging...")
    df = df_ch.copy()
    # Rename for consistency (ensure these columns exist from scraping/manual assignment)
    rename_map = {'Name': 'ch_name', 'Age': 'ch_age', 'Birthdate': 'ch_birthdate', 'Elevated': 'ch_elevated_date', 'Current Title': 'ch_title'}
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    # Ensure 'ch_id' exists or create from index if needed (should be created in match_names now)
    if 'ch_id' not in df.columns:
        log.warning("ch_id not found in loaded CH data, creating from index.")
        df['ch_id'] = df.index # Fallback

    # Select necessary columns
    required_cols = ['ch_id', 'ch_name', 'ch_age', 'ch_birthdate', 'ch_elevated_date', 'ch_title', 'ProfileLink']
    available_cols = [col for col in required_cols if col in df.columns]
    log.info(f"Selected CH columns for merge: {available_cols}")

    # AI: Add appointing_pope column based on ch_elevated_date
    if 'ch_elevated_date' in df.columns:
        log.info("Inferring appointing pope from 'ch_elevated_date'.")
        df['appointing_pope'] = df['ch_elevated_date'].apply(_infer_appointing_pope)
        log.debug(f"Sample appointing popes:\n{df[['ch_id', 'ch_name', 'ch_elevated_date', 'appointing_pope']].head()}")
    else:
        log.warning("'ch_elevated_date' column not found, cannot infer appointing pope.")
        df['appointing_pope'] = None # Ensure column exists even if empty

    return df[available_cols + ['appointing_pope']]


_REIGN_DATES = {
    'John Paul II': (datetime(1978, 10, 16), datetime(2005, 4, 2)),
    'Benedict XVI': (datetime(2005, 4, 19), datetime(2013, 2, 28)),
    'Francis': (datetime(2013, 3, 13), datetime.now()) # Assume present for end
}

def _infer_appointing_pope(elevation_date_str: Optional[str]) -> Optional[str]:
    """Infers appointing Pope based on elevation date string."""
    if pd.isna(elevation_date_str):
        return None

    try:
        # Common date formats from CH data seem to be like '22 Feb 2014'
        elevation_date = pd.to_datetime(elevation_date_str, errors='coerce')
        if pd.isna(elevation_date):
             log.warning(f"Could not parse elevation date: {elevation_date_str}")
             return None

        for pope, (start, end) in _REIGN_DATES.items():
            if start <= elevation_date <= end:
                return pope
        log.warning(f"Elevation date {elevation_date_str} ({elevation_date}) outside known reigns.")
        return None # Or handle differently if needed
    except Exception as e:
        log.error(f"Error inferring pope for date '{elevation_date_str}': {e}")
        return None


def _merge_datasets(df_gc: pd.DataFrame, df_ch: pd.DataFrame, matches: List[Dict[str, int]]) -> Optional[pd.DataFrame]:
    """Merges the preprocessed GC and CH DataFrames based on LLM matches."""
    log.info(f"Merging datasets using {len(matches)} matches...")
    if df_gc.empty or df_ch.empty or not matches:
        log.error("Cannot merge: One or more input DataFrames or the matches list is empty.")
        return None

    df_matches = pd.DataFrame(matches)

    # Ensure match IDs are integers
    try:
        df_matches['gc_id'] = df_matches['gc_id'].astype(int)
        df_matches['ch_id'] = df_matches['ch_id'].astype(int)
    except (TypeError, ValueError) as e:
        log.error(f"Error converting match IDs to integers: {e}", exc_info=True)
        log.error(f"Problematic matches data types:\n{df_matches.dtypes}")
        return None

    # --- Perform Merge ---
    # Merge GC with matches
    df_merged_gc = pd.merge(df_gc, df_matches, on='gc_id', how='inner')
    log.info(f"GC merge result: {len(df_merged_gc)} rows (started with {len(df_gc)} GC, {len(df_matches)} matches)")

    # Merge the result with CH
    df_merged_final = pd.merge(df_merged_gc, df_ch, on='ch_id', how='inner')
    log.info(f"Final merge result: {len(df_merged_final)} rows (started with {len(df_merged_gc)} after GC merge, {len(df_ch)} CH)")

    if len(df_merged_final) != len(df_matches):
         log.warning(f"Final merged count ({len(df_merged_final)}) doesn't match the number of input matches ({len(df_matches)}). Some IDs might have been missing in raw data or duplicates existed.")
         # Log missing IDs for debugging
         missing_gc = set(df_matches['gc_id']) - set(df_gc['gc_id'])
         missing_ch = set(df_matches['ch_id']) - set(df_ch['ch_id'])
         if missing_gc: log.warning(f"GC IDs from matches missing in GC data: {missing_gc}")
         if missing_ch: log.warning(f"CH IDs from matches missing in CH data: {missing_ch}")


    if df_merged_final.empty:
        log.error("Merging resulted in an empty DataFrame.")
        return None

    log.info("Merge completed successfully.")
    return df_merged_final

def _standardize_final_names(df: pd.DataFrame) -> pd.DataFrame:
    """Applies final standardization to names in the merged DataFrame."""
    if 'ch_name' in df.columns:
        log.info("Applying final name standardization based on 'ch_name'.")
        # Simple standardization: remove periods and strip whitespace
        df['name_clean'] = df['ch_name'].astype(str).str.replace('.', '', regex=False).str.strip()
    else:
        log.warning("Column 'ch_name' not found. Cannot apply final name standardization.")
        df['name_clean'] = None # Add empty column if source is missing
    return df

def _add_ideology_score(df: pd.DataFrame) -> pd.DataFrame:
    """Adds a placeholder 'ideology_score' column with random floats between -1 and 1."""
    log.info("Adding placeholder 'ideology_score' column.")
    num_electors = len(df)
    df['ideology_score'] = np.random.uniform(-1.0, 1.0, num_electors)
    log.debug(f"Sample ideology scores:\n{df[['gc_id', 'name_clean', 'ideology_score']].head()}")
    return df

def _save_merged_data(df: pd.DataFrame, output_path: Path) -> bool:
    """Saves the final merged and processed DataFrame to CSV."""
    if df.empty:
        log.error("Cannot save merged data: DataFrame is empty.")
        return False
    try:
        log.info(f"Saving final merged data ({df.shape}) to {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding='utf-8')
        log.info("Final merged data saved successfully.")
        return True
    except Exception as e:
        log.error(f"Error saving final merged data to {output_path}: {e}", exc_info=True)
        return False

# --- Orchestration ---

def run_llm_matching_script() -> bool:
    """Runs the match_names.py script using subprocess or direct call."""
    log.info(f"Attempting to run LLM matching script: {MATCH_NAMES_SCRIPT_PATH}")

    # Option 1: Direct Import and Call (Preferred)
    try:
        # Ensure the src directory is in the Python path
        if str(SRC_DIR) not in sys.path:
            sys.path.insert(0, str(SRC_DIR))

        # Import the module
        match_names_module = importlib.import_module("match_names")
        # Reload in case it was imported before and changed
        match_names_module = importlib.reload(match_names_module)

        # Call its main function
        log.info("Executing match_names.main() function...")
        match_names_module.main() # Assuming main() exits with 0 on success, raises SystemExit on error
        log.info("match_names.main() executed.")
        return True
    except SystemExit as e:
         if e.code == 0:
              log.info("match_names.main() exited successfully.")
              return True
         else:
              log.error(f"match_names.main() exited with error code: {e.code}")
              return False
    except ModuleNotFoundError:
        log.error("Could not import match_names module. Check structure and __init__.py files if needed.")
        return False
    except AttributeError:
         log.error("Could not find main() function in match_names module.")
         return False
    except Exception as e:
        log.error(f"Error running match_names.main() directly: {e}", exc_info=True)
        return False

    # # Option 2: Subprocess (Fallback if direct import fails)
    # log.warning("Falling back to running match_names.py via subprocess.")
    # try:
    #     # Use sys.executable to ensure the correct Python interpreter is used
    #     process = subprocess.run([sys.executable, str(MATCH_NAMES_SCRIPT_PATH)],
    #                              capture_output=True, text=True, check=True, cwd=BASE_DIR)
    #     log.info("match_names.py subprocess finished successfully.")
    #     log.info(f"Subprocess stdout:\n{process.stdout}")
    #     if process.stderr:
    #          log.warning(f"Subprocess stderr:\n{process.stderr}")
    #     return True
    # except FileNotFoundError:
    #     log.error(f"Error: Python executable or script not found: {sys.executable}, {MATCH_NAMES_SCRIPT_PATH}")
    #     return False
    # except subprocess.CalledProcessError as e:
    #     log.error(f"match_names.py subprocess failed with exit code {e.returncode}")
    #     log.error(f"Subprocess stdout:\n{e.stdout}")
    #     log.error(f"Subprocess stderr:\n{e.stderr}")
    #     return False
    # except Exception as e:
    #      log.error(f"Unexpected error running subprocess: {e}", exc_info=True)
    #      return False


def process_and_merge_data(gc_raw_path: Path = GCATHOLIC_RAW_PATH,
                           ch_raw_path: Path = CH_RAW_PATH,
                           matches_path: Path = LLM_MATCHES_PATH,
                           output_path: Path = MERGED_ELECTORS_PATH) -> bool:
    """Loads raw data and matches, preprocesses, merges, and saves the final dataset."""
    log.info("--- Starting Data Processing and Merging ---")

    ## AI: Continuing from the previous response...

    # 1. Load Data
    df_gc_raw, df_ch_raw, matches = _load_raw_and_matches(gc_raw_path, ch_raw_path, matches_path)
    if df_gc_raw is None or df_ch_raw is None or matches is None:
        log.error("Aborting merge due to data loading failures.")
        return False

    # 2. Preprocess each dataset
    df_gc_processed = _preprocess_gc_for_merge(df_gc_raw)
    df_ch_processed = _preprocess_ch_for_merge(df_ch_raw)

    if df_gc_processed.empty or df_ch_processed.empty:
        log.error("Aborting merge: Preprocessing resulted in empty DataFrames.")
        return False

    # 3. Merge based on LLM matches
    df_merged = _merge_datasets(df_gc_processed, df_ch_processed, matches)
    if df_merged is None or df_merged.empty:
        log.error("Aborting: Merging failed or produced an empty DataFrame.")
        return False

    # 4. Final Standardization (e.g., cleaning names)
    df_final = _standardize_final_names(df_merged)

    # 5. Add placeholder ideology score
    df_final = _add_ideology_score(df_final)

    # 6. Rename ID column (AI: Added Step)
    if 'gc_id' in df_final.columns:
        log.info("Renaming 'gc_id' to 'elector_id' for simulation compatibility.")
        df_final = df_final.rename(columns={'gc_id': 'elector_id'})
    else:
        log.warning("'gc_id' column not found for renaming to 'elector_id'.")

    # 7. Save the result
    success = _save_merged_data(df_final, output_path)
    if success:
        log.info("--- Data Processing and Merging Completed Successfully ---")
    else:
        log.error("--- Data Processing and Merging Failed during save ---")

    return success


def run_ingestion_pipeline(force_scrape: bool = False, force_match: bool = False):
    """Executes the full data ingestion pipeline.

    Steps:
    1. Scrapes GCatholic data (if forced or file missing).
    2. Scrapes Catholic Hierarchy data (if forced or file missing).
    3. Runs LLM matching script (if forced or matches file missing).
    4. Processes and merges the data.

    Args:
        force_scrape: If True, scrapes data even if raw files exist.
        force_match: If True, runs LLM matching even if matches file exists.
    """
    log.info("======= Starting Conclave Elector Ingestion Pipeline =======")

    # --- Step 1 & 2: Scraping ---
    gc_scraped, ch_scraped = True, True # Assume success unless scraping is attempted and fails
    if force_scrape or not GCATHOLIC_RAW_PATH.exists():
        log.info("Scraping GCatholic data...")
        gc_scraped = scrape_gcatholic_roster(output_path=GCATHOLIC_RAW_PATH)
    else:
        log.info(f"GCatholic raw data already exists: {GCATHOLIC_RAW_PATH}. Skipping scrape.")

    if force_scrape or not CH_RAW_PATH.exists():
        log.info("Scraping Catholic Hierarchy data...")
        ch_scraped = scrape_catholic_hierarchy_roster(output_path=CH_RAW_PATH)
    else:
        log.info(f"Catholic Hierarchy raw data already exists: {CH_RAW_PATH}. Skipping scrape.")

    if not gc_scraped or not ch_scraped:
        log.error("Scraping failed for one or both sources. Aborting pipeline.")
        return

    # --- Step 3: LLM Matching ---
    match_run_needed = force_match or not LLM_MATCHES_PATH.exists()
    matches_exist_or_created = False
    if match_run_needed:
        log.info("Running LLM name matching script...")
        match_success = run_llm_matching_script()
        if match_success and LLM_MATCHES_PATH.exists():
             log.info("LLM Matching script completed successfully.")
             matches_exist_or_created = True
        else:
             log.error("LLM Matching script failed or did not produce output file. Aborting pipeline.")
             return
    else:
        log.info(f"LLM matches file already exists: {LLM_MATCHES_PATH}. Skipping LLM matching run.")
        matches_exist_or_created = True # It already exists

    if not matches_exist_or_created:
        log.error("LLM matches file is required but is missing or could not be created. Aborting.")
        return

    # --- Step 4: Process and Merge ---
    log.info("Proceeding to process and merge data...")
    merge_success = process_and_merge_data(
        gc_raw_path=GCATHOLIC_RAW_PATH,
        ch_raw_path=CH_RAW_PATH,
        matches_path=LLM_MATCHES_PATH,
        output_path=MERGED_ELECTORS_PATH
    )

    if merge_success:
        log.info(f"======= Ingestion Pipeline Completed Successfully. Output: {MERGED_ELECTORS_PATH} =======")
    else:
        log.error("======= Ingestion Pipeline Failed During Processing/Merging =======")


# --- Main Execution Guard ---
if __name__ == "__main__":
    # Example: Run the full pipeline, forcing scrape and matching
    # run_ingestion_pipeline(force_scrape=True, force_match=True)

    # Default: Run only if necessary
    run_ingestion_pipeline()