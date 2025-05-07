import sys
import os
import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

import pandas as pd
import requests
from bs4 import BeautifulSoup
from unidecode import unidecode
import google.generativeai as genai

# AI: Use relative import for utils
from .utils import setup_logging
from .match_names import match_datasets_llm # Updated import

def _process_and_merge_data(
    gc_raw_path: Path,
    ch_raw_path: Path,
    llm_match_path: Path,
    conclavoscope_json_path: Path,
    conclavoscope_llm_match_path: Path,
    merged_output_path: Path,
    cache_dir: Path,
    force_llm_match: bool, # Base GC/CH match (currently unused here, handled externally)
    force_conclavoscope_match: bool
) -> bool:
    """Processes raw data, runs matching, merges, and saves the final elector dataset.

    Args:
        gc_raw_path: Path to raw GCatholic data CSV.
        ch_raw_path: Path to raw Catholic Hierarchy data CSV.
        llm_match_path: Path to JSON file with base LLM matches (gc_id <-> ch_id).
        conclavoscope_json_path: Path to parsed Conclavoscope JSON data.
        conclavoscope_llm_match_path: Path to JSON file with Conclavoscope LLM matches.
        merged_output_path: Path to save the final merged CSV.
        cache_dir: Path to the cache directory.
        force_llm_match: Flag to force base GC/CH matching (external process).
        force_conclavoscope_match: Flag to force Conclavoscope LLM matching.

    Returns:
        True if processing and saving were successful, False otherwise.
    """
    log = setup_logging(level=logging.DEBUG) # Set level to DEBUG

    try: # AI: Added try block to wrap the main processing steps
        # --- 1. Create Cache Directory --- (Already done in main block, but ensure here too)
        cache_dir.mkdir(parents=True, exist_ok=True)

        # --- 2. Load Raw Data --- #
        try:
            gc_df = pd.read_csv(gc_raw_path, dtype=str) # Load all as string initially
            ch_df = pd.read_csv(ch_raw_path, dtype=str)
            log.info(f"Loaded raw GCatholic ({len(gc_df)} records) and Catholic Hierarchy ({len(ch_df)} records).")
        except FileNotFoundError as e:
            log.error(f"Raw data file not found: {e}")
            return False

        # --- 3. Preprocess Raw Data --- #
        # GCatholic: Assign ID, extract name
        # Assuming the first column is the ID column if unnamed
        if gc_df.columns[0].startswith('Unnamed'):
            gc_df.rename(columns={gc_df.columns[0]: 'gc_id'}, inplace=True)
        elif 'gc_id' not in gc_df.columns:
            log.warning("Could not reliably identify 'gc_id' column in GCatholic data. Using index.")
            gc_df['gc_id'] = gc_df.index.astype(str)

        gc_df['gc_id'] = gc_df['gc_id'].astype(str).str.strip()

        # Extract name from the specific column 'Cardinal Electors(135)' if it exists
        gc_name_col = 'Cardinal Electors(135)' # Specific column from previous context
        if gc_name_col in gc_df.columns:
            gc_df['gc_name_extracted'] = gc_df[gc_name_col].str.split('\\s*\\(').str[0].str.strip()
            log.info(f"Successfully extracted name from GCatholic '{gc_name_col}' column.")
        else:
            log.warning(f"Column '{gc_name_col}' not found in GCatholic data. Cannot extract names this way.")
            gc_df['gc_name_extracted'] = pd.NA

        # Assume GCatholic 'Country' column exists - keep it for later coalescing
        if 'Country' not in gc_df.columns:
             log.warning("Column 'Country' not found in GCatholic data. Country fallback will not work.")

        # Catholic Hierarchy: Assign ID, extract country
        if ch_df.columns[0].startswith('Unnamed'):
             ch_df.rename(columns={ch_df.columns[0]: 'ch_id'}, inplace=True)
        elif 'ch_id' not in ch_df.columns:
            log.warning("Could not reliably identify 'ch_id' column in CH data. Using index.")
            ch_df['ch_id'] = ch_df.index.astype(str)

        ch_df['ch_id'] = ch_df['ch_id'].astype(str).str.strip()

        # Extract Country from 'Current Title'
        title_col = 'Current Title' # As identified previously
        if title_col in ch_df.columns:
             ch_df['country'] = ch_df[title_col].str.extract(r'of\s+([^,]+),')[0].str.strip()
             missing_country_count = ch_df['country'].isna().sum()
             if missing_country_count > 0:
                  log.warning(f"Could not extract country for {missing_country_count} / {len(ch_df)} records from CH '{title_col}'.")
        else:
             log.warning(f"Column '{title_col}' not found in CH data. Cannot extract country.")
             ch_df['country'] = pd.NA

        # --- 4. Load Base LLM Matches (GC <-> CH) --- #
        try:
            with open(llm_match_path, 'r', encoding='utf-8') as f:
                base_matches_list = json.load(f)
            base_matches_df = pd.DataFrame(base_matches_list)
            # Ensure IDs are strings for merging
            base_matches_df['gc_id'] = base_matches_df['gc_id'].astype(str)
            base_matches_df['ch_id'] = base_matches_df['ch_id'].astype(str)
            log.info(f"Loaded {len(base_matches_df)} base GC/CH matches from {llm_match_path}.")
        except FileNotFoundError:
            log.error(f"Base LLM match file not found: {llm_match_path}. Cannot proceed with merging.")
            return False
        except Exception as e:
            log.error(f"Error loading base LLM matches from {llm_match_path}: {e}")
            return False

        # --- 5. Merge GC and CH Data --- #
        log.info("Merging GCatholic and Catholic Hierarchy data based on LLM matches...")
        # Start with GC data, keeping potentially relevant raw columns
        merged_df = gc_df[['gc_id', 'gc_name_extracted']].copy() # Removed 'Country' as it's not in the parsed gc_raw data
        # Add ch_id based on LLM matches
        merged_df['ch_id'] = merged_df['gc_id'].map(base_matches_df.set_index('gc_id')['ch_id']).astype('string')

        # Prepare CH columns for merge - Use correct names found in CSV
        ch_cols_to_merge = ['ch_id', 'Name', 'Birthdate', 'Elevated', 'country'] # Keep extracted country
        missing_ch_cols = [col for col in ch_cols_to_merge if col not in ch_df.columns]
        if missing_ch_cols:
            log.error(f"Missing expected columns in CH data for merge: {missing_ch_cols}")
            # Exclude missing columns from merge list
            ch_cols_to_merge = [col for col in ch_cols_to_merge if col not in missing_ch_cols]
            if not ch_cols_to_merge:
                 log.error("No valid CH columns left to merge.")
                 return False

        # Merge CH data into GC data using the mapped ch_id
        # Ensure ch_id in ch_df is also string for merge
        ch_df['ch_id'] = ch_df['ch_id'].astype('string')
        merged_df = pd.merge(merged_df, ch_df[ch_cols_to_merge], on='ch_id', how='left', suffixes=('_gc', '_ch'))

        # Handle unmatched GC entries (those without a ch_id match)
        log.info(f"Merged {len(merged_df[merged_df['ch_id'].notna()])} records based on LLM matches.")
        unmatched_gc_count = merged_df['ch_id'].isna().sum()
        if unmatched_gc_count > 0:
            log.warning(f"{unmatched_gc_count} GCatholic entries did not have an LLM match to Catholic Hierarchy.")

        # Assign elector_id before renaming and checks
        merged_df['elector_id'] = merged_df['gc_id']

        # Rename columns for clarity and consistency after merge
        # Preserve original source columns with suffixes for coalescing
        rename_map = {
            'gc_name_extracted': 'gc_name_raw',
            'Name': 'ch_name_raw',        # Raw CH name
            'Birthdate': 'dob',            # CH Birthdate
            'Elevated': 'date_cardinal',  # CH Elevated date
            'country': 'ch_country_extracted' # CH Extracted country
        }
        cols_to_rename = {k: v for k, v in rename_map.items() if k in merged_df.columns}
        merged_df.rename(columns=cols_to_rename, inplace=True)

        # --- 6. Create Clean Names and Dates --- #
        # Create unified name column (prefer CH if available)
        merged_df['name_clean'] = merged_df['ch_name_raw'].fillna(merged_df['gc_name_raw'])
        merged_df['name_clean'] = merged_df['name_clean'].str.lower().str.strip()
        merged_df['name_clean_ascii'] = merged_df['name_clean'].apply(lambda x: unidecode(x) if pd.notna(x) else None)

        # Convert date columns
        merged_df['dob'] = pd.to_datetime(merged_df['dob'], errors='coerce').dt.strftime('%Y-%m-%d')
        merged_df['date_cardinal'] = pd.to_datetime(merged_df['date_cardinal'], errors='coerce').dt.strftime('%Y-%m-%d')

        # Define intermediate base columns
        base_cols = ['elector_id', 'gc_id', 'ch_id', 'name_clean', 'name_clean_ascii', 'dob', 'date_cardinal', 'ch_country_extracted']
        # Add missing columns with NA
        for col in base_cols:
            if col not in merged_df.columns:
                merged_df[col] = pd.NA
        base_merged_df = merged_df[base_cols].copy()

        # --- 7. Load and Process Conclavoscope Data --- #
        conclave_df = pd.DataFrame() # Initialize empty df
        final_merged_df = base_merged_df.copy() # Initialize final df with base merge

        # Check if the Conclavoscope input file exists
        if os.path.exists(conclavoscope_json_path):
            log.info(f"Found Conclavoscope data file: {conclavoscope_json_path}. Attempting to load and process.")
            try:
                with open(conclavoscope_json_path, 'r', encoding='utf-8') as f:
                    conclave_data = json.load(f)
                conclave_df = pd.DataFrame(conclave_data)
                # Rename columns for clarity and to avoid clashes
                conclave_df.rename(columns={
                    'id': 'cs_id',
                    'name': 'cs_name_raw',
                    # 'cs_country' is already named correctly from the scrape step
                    'papabile_score': 'cs_papabile_score',
                    'alignment_score': 'cs_alignment_score',
                    'total_score': 'cs_total_score'
                }, inplace=True)
                conclave_df['cs_id'] = conclave_df['cs_id'].astype(str)
                log.info(f"Loaded and preprocessed {len(conclave_df)} Conclavoscope records.")

                # Check if cs_country column exists after scraping
                if 'cs_country' not in conclave_df.columns:
                    log.warning("Column 'cs_country' not found in Conclavoscope data. Country coalescing might be incomplete.")
                    conclave_df['cs_country'] = pd.NA # Add column if missing

                # --- 8. Match Conclavoscope Data --- #
                # Requires GOOGLE_API_KEY environment variable
                api_key = os.getenv("GOOGLE_API_KEY")
                conclave_llm_matches_df = pd.DataFrame() # Initialize empty matches df
                run_conclavoscope_matching = False

                # Check if matching is needed (file doesn't exist or forced)
                if not conclavoscope_llm_match_path.exists() or force_conclavoscope_match:
                    if api_key:
                        log.info(f"Conclavoscope LLM match file {'not found' if not conclavoscope_llm_match_path.exists() else 'found but forcing re-match'}. Running LLM matching.")
                        run_conclavoscope_matching = True
                    else:
                        log.warning("GOOGLE_API_KEY not set. Cannot perform Conclavoscope LLM matching. Skipping.")
                else:
                    log.info(f"Using existing Conclavoscope LLM matches from: {conclavoscope_llm_match_path}")
                    try:
                        with open(conclavoscope_llm_match_path, 'r', encoding='utf-8') as f:
                            conclave_matches_list = json.load(f)
                        conclave_llm_matches_df = pd.DataFrame(conclave_matches_list)
                        if not conclave_llm_matches_df.empty:
                            conclave_llm_matches_df['elector_id'] = conclave_llm_matches_df['elector_id'].astype(str)
                            conclave_llm_matches_df['cs_id'] = conclave_llm_matches_df['cs_id'].astype(str)
                        log.info(f"Loaded {len(conclave_llm_matches_df)} existing Conclavoscope matches.")
                    except Exception as e:
                        log.error("Error loading existing Conclavoscope matches from %s: %s. Attempting re-match if API key available.", conclavoscope_llm_match_path, e)
                        if api_key:
                            run_conclavoscope_matching = True # Try to re-generate matches
                        else:
                            log.warning("GOOGLE_API_KEY not set. Cannot re-run Conclavoscope LLM matching after load error. Skipping.")
                            # Proceeding without Conclavoscope scores as we can't load or regenerate

                # Perform LLM matching if needed and possible
                if run_conclavoscope_matching and not base_merged_df.empty and not conclave_df.empty:
                    log.info("Starting LLM matching for Conclavoscope data...")
                    # Ensure match_datasets_llm is imported or defined
                    merged_df_for_cs_match = base_merged_df.copy()
                    if 'elector_id' not in merged_df_for_cs_match.columns:
                        # This case should ideally not happen if _merge_gc_ch creates elector_id
                        log.warning("'elector_id' not found in merged_df. Attempting to use index as fallback for matching.")
                        merged_df_for_cs_match['elector_id'] = merged_df_for_cs_match.index
                    if 'name_clean_ascii' not in merged_df_for_cs_match.columns: # Changed 'name' to 'name_clean_ascii'
                        log.error("'name_clean_ascii' column not found in merged_df_for_cs_match. Cannot proceed with Conclavoscope matching.")
                        return False
                    
                    if 'cs_id' not in conclave_df.columns or 'cs_name_raw' not in conclave_df.columns: # Changed 'cs_name' to 'cs_name_raw'
                        log.error("'cs_id' or 'cs_name_raw' not found in Conclavoscope data. Cannot proceed with matching.")
                        return False

                    cs_matches = match_datasets_llm(
                        df1=merged_df_for_cs_match,
                        df1_name_col='name_clean_ascii',  # Name column in the merged elector list
                        df1_id_col='elector_id',        # ID column in the merged elector list
                        df2=conclave_df,
                        df2_name_col='cs_name_raw', # Name column in Conclavoscope data
                        df2_id_col='cs_id',          # ID column in Conclavoscope data
                        output_path=conclavoscope_llm_match_path,
                        dataset1_label="Current Electors",
                        dataset2_label="Conclavoscope"
                    )
                    if cs_matches: # If matches were returned
                        conclave_llm_matches_df = pd.DataFrame(cs_matches)
                        if not conclave_llm_matches_df.empty:
                            # Ensure column types are consistent with loaded matches
                            if 'elector_id' in conclave_llm_matches_df.columns:
                                conclave_llm_matches_df['elector_id'] = conclave_llm_matches_df['elector_id'].astype(str)
                            if 'cs_id' in conclave_llm_matches_df.columns:
                                conclave_llm_matches_df['cs_id'] = conclave_llm_matches_df['cs_id'].astype(str)
                            log.info(f"Successfully generated and processed {len(conclave_llm_matches_df)} Conclavoscope LLM matches.")
                        else:
                            log.warning("LLM matching for Conclavoscope data returned an empty list/DataFrame.")
                            conclave_llm_matches_df = pd.DataFrame() # Ensure it's an empty DataFrame
                    else: # cs_matches is None or empty list
                        log.warning("LLM matching for Conclavoscope data did not return any matches or failed.")
                        conclave_llm_matches_df = pd.DataFrame() # Ensure it's an empty DataFrame
                elif run_conclavoscope_matching:
                    log.warning("Cannot run Conclavoscope LLM matching because base or Conclavoscope data is empty.")

                # --- 9. Merge Conclavoscope Scores --- #
                if not conclave_df.empty and not conclave_llm_matches_df.empty and 'elector_id' in conclave_llm_matches_df.columns:
                    # Prepare Conclavoscope columns to merge
                    cs_cols_to_merge = ['cs_id', 'cs_name_raw', 'cs_country', 'cs_papabile_score', 'cs_alignment_score', 'cs_total_score']
                    missing_cs_cols = [col for col in cs_cols_to_merge if col not in conclave_df.columns]
                    if missing_cs_cols:
                        log.warning(f"Missing expected columns in Conclavoscope data for merge: {missing_cs_cols}")
                        cs_cols_to_merge = [col for col in cs_cols_to_merge if col not in missing_cs_cols]

                    if cs_cols_to_merge:
                        # Merge based on LLM matches (elector_id <-> cs_id)
                        log.info("Merging Conclavoscope scores into the main dataset...")
                        # Select only the necessary columns from matches and CS data
                        matches_subset = conclave_llm_matches_df[['elector_id', 'cs_id']] # Map base elector_id to cs_id
                        cs_data_subset = conclave_df[cs_cols_to_merge]
                        # Merge matches with CS data
                        cs_data_to_merge = pd.merge(matches_subset, cs_data_subset, on='cs_id', how='left')
                        # Merge this into the base merged data
                        # NOTE: Re-assign to final_merged_df here
                        final_merged_df = pd.merge(base_merged_df, cs_data_to_merge.drop(columns=['cs_id']), on='elector_id', how='left') # Drop cs_id after merge
                        merged_score_count = final_merged_df['cs_total_score'].notna().sum()
                        log.info(f"Successfully merged Conclavoscope scores for {merged_score_count} electors.")
                    else:
                        log.warning("Skipping Conclavoscope score merge as no valid columns were found.")
                        # final_merged_df remains base_merged_df
                else:
                    log.info("Skipping Conclavoscope score merge (no data or no matches).")
                    # final_merged_df remains base_merged_df

            except FileNotFoundError:
                log.error(f"Conclavoscope data file disappeared unexpectedly after check: {conclavoscope_json_path}.")
            except Exception as e:
                log.error("Error processing Conclavoscope data: %s", e)
                log.error("Detailed error (repr): %r", e)
                log.error("Detailed error type: %s", type(e))
                # Treat error as non-fatal, skip Conclavoscope merge
                log.warning("Proceeding without Conclavoscope scores due to processing error.")
        else:
            # This block executes if the file was not found by os.path.exists()
            log.warning(f"Conclavoscope data file not found at {conclavoscope_json_path}. Proceeding without Conclavoscope scores.")
            # final_merged_df remains base_merged_df (already initialized)

        # --- 10. Coalesce Country Information --- #
        log.info("Coalescing final country information...")
        final_country_col = 'country' # Final desired column name

        # Initialize with NA
        final_merged_df[final_country_col] = pd.NA

        # Apply coalescing with priority: cs_country -> ch_country_extracted -> gc_country_raw
        if 'cs_country' in final_merged_df.columns:
            log.info("Prioritizing country from Conclavoscope ('cs_country').")
            final_merged_df[final_country_col] = final_merged_df[final_country_col].combine_first(final_merged_df['cs_country'])
        else:
            log.warning("'cs_country' column not available for coalescing.")

        if 'ch_country_extracted' in final_merged_df.columns:
            log.info("Falling back to country extracted from Catholic Hierarchy ('ch_country_extracted').")
            final_merged_df[final_country_col] = final_merged_df[final_country_col].combine_first(final_merged_df['ch_country_extracted'])
        else:
            log.warning("'ch_country_extracted' column not available for coalescing.")

        # Log final country count
        country_count = final_merged_df[final_country_col].notna().sum()
        log.info(f"Final dataset contains country information for {country_count} / {len(final_merged_df)} electors.")

        # --- 11. Create Final Ideology Score --- #
        log.info("Creating final 'ideology_score' based on 'cs_total_score'.")
        if 'cs_total_score' in final_merged_df.columns:
            # Convert to numeric, coercing errors
            scores_numeric = pd.to_numeric(final_merged_df['cs_total_score'], errors='coerce')

            # Impute missing values with the median
            median_score = scores_numeric.median()
            if pd.isna(median_score):
                log.warning("Median 'cs_total_score' could not be calculated (all values might be NaN). Imputing missing scores with 0.")
                scores_imputed = scores_numeric.fillna(0)
                median_score = 0 # Update median_score if it was NaN
            else:
                log.info(f"Imputing {scores_numeric.isna().sum()} missing 'cs_total_score' values with median ({median_score}).")
                scores_imputed = scores_numeric.fillna(median_score)

            # Normalize to [0, 1]
            min_score = scores_imputed.min()
            max_score = scores_imputed.max()
            if min_score == max_score: # Avoid division by zero if all scores are the same
                log.warning("All 'cs_total_score' values are identical after imputation. Setting 'ideology_score' to 0.5.")
                final_merged_df['ideology_score'] = 0.5
            else:
                final_merged_df['ideology_score'] = (scores_imputed - min_score) / (max_score - min_score)
                log.info(f"Normalized 'cs_total_score' to 'ideology_score' range [0, 1]. Min: {min_score}, Max: {max_score}")
        else:
            log.warning("'cs_total_score' column not found after merge. Cannot create 'ideology_score'. It will be missing.")
            final_merged_df['ideology_score'] = pd.NA # Explicitly add as NA

        # --- 12. Final Column Selection and Save --- #
        final_cols = [
            'elector_id', 'gc_id', 'ch_id', 'cs_id',
            'name_clean', 'name_clean_ascii',
            'dob', 'date_cardinal', 'country',        # Final coalesced country
            'ideology_score',                          # Derived from cs_total_score
            'cs_total_score',                          # Keep original score for reference
            # Optionally keep raw/intermediate fields for debugging:
            # 'gc_name_raw', 'ch_name_raw', 'gc_country_raw', 'ch_country_extracted', 'cs_country', 'cs_name_raw'
        ]

        # Ensure all final columns exist, add NA if not
        for col in final_cols:
            if col not in final_merged_df.columns:
                log.warning(f"Expected final column '{col}' missing. Adding with NA.")
                final_merged_df[col] = pd.NA

        # Select and save
        final_df_to_save = final_merged_df[final_cols].copy()
        final_df_to_save.replace({pd.NA: None}, inplace=True) # Replace Pandas NA with None for cleaner CSV

        merged_output_path.parent.mkdir(parents=True, exist_ok=True)
        final_df_to_save.to_csv(merged_output_path, index=False, encoding='utf-8')
        log.info(f"Successfully saved final merged data ({len(final_df_to_save)} records) to {merged_output_path}")

        return True # Function completed successfully

    except FileNotFoundError as e:
        log.error(f"File not found during processing: {e}", exc_info=True)
        return False
    except KeyError as e:
        log.error(f"Missing expected column during processing: {e}", exc_info=True)
        return False
    except Exception as e:
        # AI: Catch-all exception for the main processing block
        log.error(f"An unexpected error occurred during data processing and merging: {e}", exc_info=True)
        return False


def main():
    """Main execution function for the ingestion script."""
    # Define base directory relative to this script's location
    script_dir = Path(__file__).parent.resolve()
    base_dir = script_dir.parent # Go up one level from src/ to project root
    data_dir = base_dir / 'data'

    # Define file paths (adjust if naming convention differs)
    gc_raw_path = data_dir / 'gcatholic_cardinals_raw.csv'
    ch_raw_path = data_dir / 'scraped_ch_raw.csv'
    llm_match_path = data_dir / 'llm_matched_pairs.json'
    conclavoscope_json_path = data_dir / 'conclavoscope_parsed.json'
    conclavoscope_llm_match_path = data_dir / 'conclavoscope_llm_matches.json'
    merged_output_path = data_dir / 'merged_electors.csv'
    cache_dir = data_dir / 'cache'

    # Setup logging (redundant if _process_and_merge_data does it, but safe)
    log = setup_logging(level=logging.DEBUG)

    parser = argparse.ArgumentParser(description="Conclave elector data ingestion script.")
    parser.add_argument(
        "--force-gc-ch-match",
        action="store_true",
        help="Force re-running of the base GCatholic-CatholicHierarchy LLM matching process."
    )
    parser.add_argument(
        "--force-cs-match",
        action="store_true",
        help="Force re-running of the Conclavoscope LLM matching process."
    )
    args = parser.parse_args()

    force_llm_match = args.force_gc_ch_match
    force_conclavoscope_match = args.force_cs_match

    log.info(f"Running ingestion with output path: {merged_output_path}")
    log.info(f"Force GC-CH Match: {force_llm_match}")
    log.info(f"Force Conclavoscope Match: {force_conclavoscope_match}")

    success = _process_and_merge_data(
        gc_raw_path=gc_raw_path,
        ch_raw_path=ch_raw_path,
        llm_match_path=llm_match_path,
        conclavoscope_json_path=conclavoscope_json_path,
        conclavoscope_llm_match_path=conclavoscope_llm_match_path,
        merged_output_path=merged_output_path,
        cache_dir=cache_dir,
        force_llm_match=force_llm_match,
        force_conclavoscope_match=force_conclavoscope_match
    )

    if success:
        log.info("Ingestion process completed successfully.")
    else:
        log.error("Ingestion process failed.")
        sys.exit(1) # Exit with error code if failed

# Call main function when script is executed
if __name__ == "__main__":
    main()