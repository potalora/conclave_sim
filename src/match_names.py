# AI: Refactored src/match_names.py
"""
Leverages the Gemini LLM to match cardinal names between GCatholic and
Catholic Hierarchy datasets.

Reads raw scraped CSV files, preprocesses them slightly, generates a prompt
for the LLM, calls the Gemini API, parses the JSON response containing
matched IDs, and saves the matches to a JSON file.

Requires the GOOGLE_API_KEY environment variable to be set.
"""

import os
import pandas as pd
import google.generativeai as genai
import json
import time
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import logging

# Initialize logging
log = logging.getLogger(__name__)

# --- Constants ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
GC_RAW_PATH = DATA_DIR / "scraped_gcatholic_raw.csv"
CH_RAW_PATH = DATA_DIR / "scraped_ch_raw.csv"
MATCH_OUTPUT_PATH = DATA_DIR / "llm_matched_pairs.json"
LLM_MATCHES_PATH = MATCH_OUTPUT_PATH
GEMINI_MODEL_NAME = "gemini-2.5-flash-preview-04-17" # Use the latest flash model
MAX_LLM_RETRIES = 3
RETRY_SLEEP_BASE_SECONDS = 5

# --- API Key Configuration ---
try:
    # AI: Using os.getenv for safer access, allowing None if not set
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")
    genai.configure(api_key=GOOGLE_API_KEY)
    print("Google AI API Key configured successfully.")
except ValueError as e:
    print("Configuration Error: %s" % e, file=sys.stderr)
    print("Please set the GOOGLE_API_KEY environment variable and try again.", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print("Unexpected Error configuring Google AI SDK: %s" % e, file=sys.stderr)
    sys.exit(1)

# --- Helper Functions ---

def load_raw_data(gc_path: Path = GC_RAW_PATH,
                    ch_path: Path = CH_RAW_PATH) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Loads the raw scraped data from GCatholic and Catholic Hierarchy.

    Args:
        gc_path: Path to the GCatholic raw CSV.
        ch_path: Path to the Catholic Hierarchy raw CSV.

    Returns:
        A tuple containing the GCatholic DataFrame and Catholic Hierarchy DataFrame,
        or (None, None) if loading fails.
    """
    log.info(f"Loading raw GCatholic data from: {gc_path}")
    try:
        df_gc = pd.read_csv(gc_path)
        # AI: Add gc_id based on index
        df_gc['gc_id'] = df_gc.index
        # AI: Save back to CSV immediately to persist the ID
        df_gc.to_csv(gc_path, index=False, encoding='utf-8')
        log.info(f"Loaded and updated GCatholic data with gc_id ({df_gc.shape}). Saved back to {gc_path}")
    except FileNotFoundError:
        log.error("GCatholic raw file not found: %s", gc_path)
        return None, None
    except Exception as e:
        log.error("Error loading or saving GCatholic raw data: %s", e, exc_info=True)
        return None, None

    log.info(f"Loading raw Catholic Hierarchy data from: {ch_path}")
    try:
        df_ch = pd.read_csv(ch_path)
        # AI: Add ch_id based on index
        df_ch['ch_id'] = df_ch.index
        # AI: Save back to CSV immediately to persist the ID
        df_ch.to_csv(ch_path, index=False, encoding='utf-8')
        log.info(f"Loaded and updated Catholic Hierarchy data with ch_id ({df_ch.shape}). Saved back to {ch_path}")
    except FileNotFoundError:
        log.error("Catholic Hierarchy raw file not found: %s", ch_path)
        return None, None
    except Exception as e:
        log.error("Error loading or saving Catholic Hierarchy raw data: %s", e, exc_info=True)
        return None, None

    return df_gc, df_ch

def preprocess_gcatholic(df: pd.DataFrame) -> pd.DataFrame:
    """Performs minimal preprocessing on GCatholic data.

    Renames columns, converts age to numeric, filters for electors (age < 80),
    and adds a unique 'gc_id' based on the original index.

    Args:
        df: The raw GCatholic DataFrame.

    Returns:
        A preprocessed DataFrame containing only potential electors with
        relevant columns ('gc_id', 'gc_description', 'gc_age', 'gc_birthdate').
        Returns an empty DataFrame if input is empty or filtering results in no electors.
    """
    if df.empty:
        print("Warning: Input GCatholic DataFrame is empty.", file=sys.stderr)
        return pd.DataFrame(columns=['gc_id', 'gc_description', 'gc_age', 'gc_birthdate'])

    print("Preprocessing GCatholic data...")
    # Use a copy to avoid SettingWithCopyWarning
    df_processed = df.copy()
    # Rename columns robustly, checking if they exist
    rename_map = {'Cardinal Electors(135)': 'gc_description', 'Age': 'gc_age', 'Date ofBirth': 'gc_birthdate'}
    df_processed.rename(columns={k: v for k, v in rename_map.items() if k in df_processed.columns}, inplace=True)

    # Convert age if column exists
    if 'gc_age' in df_processed.columns:
        df_processed['gc_age'] = pd.to_numeric(df_processed['gc_age'], errors='coerce')
        # Filter electors (age < 80) - handle potential NaN ages by filling with a value > 80
        df_filtered = df_processed[df_processed['gc_age'].fillna(999) < 80].copy()
        print(f"GCatholic Electors (Age < 80): {len(df_filtered)}")
    else:
        print("Warning: 'Age' column not found in GCatholic data, cannot filter by age.", file=sys.stderr)
        df_filtered = df_processed.copy() # Keep all if age cannot be used

    # Select and return final columns, ensuring they exist
    final_cols = ['gc_id', 'gc_description', 'gc_age', 'gc_birthdate']
    required_cols = [col for col in final_cols if col in df_filtered.columns]
    if 'gc_id' not in required_cols or ('gc_description' not in required_cols):
         print("Error: Missing essential columns ('gc_id', 'gc_description') after GC preprocessing.", file=sys.stderr)
         # Return empty with schema to prevent downstream errors
         return pd.DataFrame(columns=final_cols)
    return df_filtered[required_cols]


def preprocess_catholic_hierarchy(df: pd.DataFrame) -> pd.DataFrame:
    """Performs minimal preprocessing on Catholic Hierarchy data.

    Renames columns and adds a unique 'ch_id' based on the original index.

    Args:
        df: The raw Catholic Hierarchy DataFrame.

    Returns:
        A preprocessed DataFrame with relevant columns ('ch_id', 'ch_name',
        'ch_age', 'ch_birthdate', 'ch_elevated_date', 'ch_title').
        Returns an empty DataFrame if input is empty.
    """
    if df.empty:
        print("Warning: Input Catholic Hierarchy DataFrame is empty.", file=sys.stderr)
        return pd.DataFrame(columns=['ch_id', 'ch_name', 'ch_age', 'ch_birthdate', 'ch_elevated_date', 'ch_title'])

    print("Preprocessing Catholic Hierarchy data...")
    df_processed = df.copy()
    # Rename columns robustly
    rename_map = {'Name': 'ch_name', 'Age': 'ch_age', 'Birthdate': 'ch_birthdate', 'Elevated': 'ch_elevated_date', 'Current Title': 'ch_title'}
    df_processed.rename(columns={k: v for k, v in rename_map.items() if k in df_processed.columns}, inplace=True)

    # Select and return final columns, ensuring they exist
    final_cols = ['ch_id', 'ch_name', 'ch_age', 'ch_birthdate', 'ch_elevated_date', 'ch_title']
    required_cols = [col for col in final_cols if col in df_processed.columns]
    if 'ch_id' not in required_cols or 'ch_name' not in required_cols:
        print("Error: Missing essential columns ('ch_id', 'ch_name') after CH preprocessing.", file=sys.stderr)
        # Return empty with schema
        return pd.DataFrame(columns=final_cols)
    return df_processed[required_cols]


def format_list_for_prompt(data_list: List[Dict[str, Any]], prefix: str) -> List[str]:
    """Formats a list of dictionaries into strings suitable for the LLM prompt.

    Dynamically includes available keys from the preprocessed data.

    Args:
        data_list: List of dictionaries (rows from preprocessed DataFrame).
        prefix: 'gc' or 'ch', used to determine ID and primary name/description field.

    Returns:
        A list of strings, each representing an item for the prompt.
    """
    formatted_lines = []
    id_key = f"{prefix}_id"
    primary_field = f"{prefix}_description" if prefix == "gc" else f"{prefix}_name"

    for item in data_list:
        # Ensure required ID and primary field are present
        if id_key not in item or primary_field not in item:
            print("Warning: Skipping item due to missing required field (%s or %s): %s" % (id_key, primary_field, item), file=sys.stderr)
            continue

        line = f"{id_key}: {item[id_key]}, {primary_field.split('_')[1]}: {item[primary_field]}"
        # Add other available fields dynamically
        other_fields = [
            f"{k.split('_')[1]}: {v}" for k, v in item.items()
            if k != id_key and k != primary_field and pd.notna(v) # Exclude NAs
        ]
        if other_fields:
            line += ", " + ", ".join(other_fields)
        formatted_lines.append(line)
    return formatted_lines

def generate_matching_prompt(gc_list: List[Dict[str, Any]], ch_list: List[Dict[str, Any]]) -> str:
    """Creates the prompt for the Gemini LLM to match names.

    Args:
        gc_list: Preprocessed GCatholic data as a list of dictionaries.
        ch_list: Preprocessed Catholic Hierarchy data as a list of dictionaries.

    Returns:
        The complete prompt string for the LLM.
    """
    log.info("Generating LLM prompt...")
    prompt_header = """
You are an expert data analyst specializing in Catholic Church hierarchy. Your task is to match individuals between two lists of cardinals based on their names and potentially other provided details (like age or birthdate if available and consistent).

List 1 contains descriptions like: 'LastName, FirstName, Suffix.(Age)...Title...'
List 2 contains names like: 'FirstName [MiddleName] CardinalLastName [Suffix]'

Please compare the two lists below and identify pairs of individuals who are likely the same person. Consider variations in name format, middle names/initials, titles (like C.SS.R., O.F.M.), and potential minor discrepancies in age or birthdate.

Provide your matches **only** as a valid JSON list of dictionaries, where each dictionary represents a matched pair and has the keys 'gc_id' and 'ch_id' corresponding to the IDs provided in the input lists. Do not include any text before or after the JSON list. Ensure the JSON is perfectly formatted.

Example Output Format:
[{"gc_id": 10, "ch_id": 5}, {"gc_id": 25, "ch_id": 112}]

If you are uncertain about a match, do not include it. Only include high-confidence matches.

--- List 1 ---
"""
    gc_prompt_lines = format_list_for_prompt(gc_list, "gc")
    ch_prompt_lines = format_list_for_prompt(ch_list, "ch")

    prompt = (
        prompt_header +
        "\n".join(gc_prompt_lines) +
        "\n\n--- List 2 ---\n" +
        "\n".join(ch_prompt_lines) +
        "\n\n--- Matched Pairs (JSON List Only) ---\n[" # Prime the LLM for JSON list output
    )
    print(f"Prompt generated. Length: {len(prompt)} characters.")
    return prompt

def call_gemini_api(prompt: str, model_name: str = GEMINI_MODEL_NAME) -> Optional[str]:
    """Sends the prompt to the Gemini model and returns the text response.

    Handles potential API errors and provides limited response info on failure.

    Args:
        prompt: The prompt string to send to the LLM.
        model_name: The name of the Gemini model to use.

    Returns:
        The text content of the LLM response, or None if an error occurs.
    """
    print(f"Sending prompt to Gemini model ({model_name})...")
    try:
        model = genai.GenerativeModel(model_name)
        # AI: Adjusting generation config for potentially better JSON output
        generation_config = genai.types.GenerationConfig(
            # candidate_count=1, # Default is 1
            # stop_sequences=['}'], # Could try stopping if it adds extra text
            # max_output_tokens=8192, # Model default usually sufficient
            # temperature=0.1, # Lower temp for more deterministic JSON
            # response_mime_type="application/json" # Temporarily removed for debugging
        )
        # Note: response_mime_type might not be supported by all models/versions
        # If it causes errors, remove it and rely on prompt engineering.
        response = model.generate_content(prompt, generation_config=generation_config)

        # Check for safety ratings or blocked content *before* accessing .text
        if response.prompt_feedback.block_reason:
            print("Warning: Prompt was blocked. Reason: %s" % response.prompt_feedback.block_reason, file=sys.stderr)
            return None
        if not response.candidates:
             print("Warning: No candidates returned by the API.", file=sys.stderr)
             return None
        if response.candidates[0].finish_reason != 'STOP':
            print("Warning: Generation finished unexpectedly. Reason: %s" % response.candidates[0].finish_reason, file=sys.stderr)
            # Still attempt to get text if available
            if response.candidates[0].content and response.candidates[0].content.parts:
                 print("Attempting to extract partial text...")
                 return response.candidates[0].text # Use .text for safer access
            else:
                 return None

        print("Received response from Gemini.")
        # Use .text attribute for safer access to the response string
        return response.text

    except AttributeError as ae:
         print("Error: Potentially invalid response structure from API: %s" % ae, file=sys.stderr)
         # Log the raw response if possible without causing further errors
         try:
            print("Raw response object type: %s" % type(response), file=sys.stderr)
            # print(f"Raw response content: {response}", file=sys.stderr) # Avoid printing potentially huge objects
         except Exception as log_e:
             print("Could not log raw response details: %s" % log_e, file=sys.stderr)
         return None
    except Exception as e:
        print("Error interacting with Gemini API: %s" % e, file=sys.stderr)
        # Attempt to log more details if available in the exception
        if hasattr(e, 'response'):
            print("API Error Response: %s" % e.response, file=sys.stderr)
        return None


def parse_llm_response(response_text: Optional[str]) -> List[Dict[str, int]]:
    """Parses the JSON string from the LLM response.

    Args:
        response_text: The text content returned by the Gemini API call.

    Returns:
        A list of matched dictionaries (e.g., [{'gc_id': 10, 'ch_id': 5}]),
        or an empty list if parsing fails or input is None.
    """
    if not response_text:
        print("Cannot parse LLM response: Input text is None or empty.", file=sys.stderr)
        return []

    print("Parsing LLM Response Snippet: %s..." % response_text[:200].strip())
    cleaned_text = response_text.strip()

    # Attempt to find JSON list within potential markdown/text
    json_start = cleaned_text.find('[')
    json_end = cleaned_text.rfind(']') + 1

    if json_start != -1 and json_end != 0:
        json_str = cleaned_text[json_start:json_end]
        try:
            matches = json.loads(json_str)
            # Validate structure
            if isinstance(matches, list) and all(isinstance(item, dict) and 'gc_id' in item and 'ch_id' in item for item in matches):
                print(f"Successfully parsed {len(matches)} matches from LLM response.")
                # Additional check for correct types (integers)
                valid_matches = []
                for item in matches:
                    try:
                        valid_matches.append({'gc_id': int(item['gc_id']), 'ch_id': int(item['ch_id'])})
                    except (ValueError, TypeError):
                         print("Warning: Skipping match with non-integer IDs: %s" % item, file=sys.stderr)
                return valid_matches
            else:
                print("Error: Parsed JSON from LLM response was not in the expected format (list of dicts with 'gc_id', 'ch_id').", file=sys.stderr)
                print("Parsed JSON structure: %s" % type(matches), file=sys.stderr)
                # print(f"Problematic JSON String: {json_str}", file=sys.stderr) # Debugging
                return []
        except json.JSONDecodeError as e:
            print("Error decoding JSON from LLM response: %s" % e, file=sys.stderr)
            print("Attempted to parse: %s" % json_str, file=sys.stderr) # Log the substring tried
            return []
    else:
        print("Error: Could not find valid JSON list structure '[]' in LLM response.", file=sys.stderr)
        print("Full Response Text: %s" % cleaned_text, file=sys.stderr)
        return []


def save_matches(matches: List[Dict[str, int]], output_path: Path) -> bool:
    """Saves the list of matched dictionaries to a JSON file.

    Args:
        matches: The list of matched dictionaries.
        output_path: The Path object for the output JSON file.

    Returns:
        True if saving was successful, False otherwise.
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(matches, f, indent=4)
        log.info(f"Successfully saved {len(matches)} matches to {output_path}")
        return True
    except IOError as e:
        log.error("Error saving matches to %s: %s", output_path, e, exc_info=True)
        return False
    except Exception as e:
        log.error("Unexpected error saving matches: %s", e, exc_info=True)
        return False

def match_datasets_llm(
    df1: pd.DataFrame,
    df1_name_col: str,
    df1_id_col: str,
    df2: pd.DataFrame,
    df2_name_col: str,
    df2_id_col: str,
    output_path: Path,
    dataset1_label: str,
    dataset2_label: str,
) -> List[Dict[str, Any]]:
    """Performs LLM-based matching between two generic DataFrames.

    Args:
        df1: First DataFrame.
        df1_name_col: Name of the column in df1 containing names/descriptions for matching.
        df1_id_col: Name of the column in df1 containing unique IDs.
        df2: Second DataFrame.
        df2_name_col: Name of the column in df2 containing names/descriptions for matching.
        df2_id_col: Name of the column in df2 containing unique IDs.
        output_path: Path to save the JSON file with matched pairs.
        dataset1_label: Label for the first dataset (e.g., "Electors").
        dataset2_label: Label for the second dataset (e.g., "Conclavoscope").

    Returns:
        A list of dictionaries, where each dictionary represents a matched pair
        with original ID column names as keys.
        Example: [{df1_id_col: id_val1, df2_id_col: id_val2}, ...]
    """
    log.info(f"Starting LLM matching between '{dataset1_label}' and '{dataset2_label}'.")

    # Prepare df1 to look like 'GCatholic' data for generate_matching_prompt
    # The format_list_for_prompt expects 'gc_id' and 'gc_description'
    df1_for_prompt = pd.DataFrame({
        'gc_id': df1[df1_id_col],
        'gc_description': df1[df1_name_col]
    })
    # Add any other columns from df1 that format_list_for_prompt might use implicitly
    # by iterating over item.keys() beyond 'gc_id' and 'gc_description'.
    # For safety, copy all other columns, prepending 'gc_' to avoid clashes if not already prefixed.
    for col in df1.columns:
        if col not in [df1_id_col, df1_name_col]:
            df1_for_prompt[f'gc_{col}'] = df1[col]

    df1_list_for_prompt = df1_for_prompt.to_dict(orient='records')

    # Prepare df2 to look like 'Catholic Hierarchy' data for generate_matching_prompt
    # The format_list_for_prompt expects 'ch_id' and 'ch_name'
    df2_for_prompt = pd.DataFrame({
        'ch_id': df2[df2_id_col],
        'ch_name': df2[df2_name_col]
    })
    for col in df2.columns:
        if col not in [df2_id_col, df2_name_col]:
            df2_for_prompt[f'ch_{col}'] = df2[col]

    df2_list_for_prompt = df2_for_prompt.to_dict(orient='records')

    # Generate prompt using the adapted data lists and provided labels
    prompt_intro = f"""  # Use f-string for labels, but fixed keys for LLM response
You are an expert data analyst specializing in Catholic Church hierarchy. Your task is to match individuals between two lists of cardinals based on their names and potentially other provided details (like age or birthdate if available and consistent).

List 1 ({dataset1_label}) contains descriptions like: 'LastName, FirstName, Suffix.(Age)...Title...'
List 2 ({dataset2_label}) contains names like: 'FirstName [MiddleName] CardinalLastName [Suffix]'

Please compare the two lists below and identify pairs of individuals who are likely the same person. Consider variations in name format, middle names/initials, titles (like C.SS.R., O.F.M.), and potential minor discrepancies in age or birthdate.

Return your response as a JSON list of dictionaries, where each dictionary contains two keys: 'gc_id' (referring to the ID from List 1 ({dataset1_label})) and 'ch_id' (referring to the ID from List 2 ({dataset2_label})), with their corresponding integer IDs.
Example: [{"gc_id": 1, "ch_id": 101}, {"gc_id": 2, "ch_id": 102}]

If you are uncertain about a match, do not include it. Only include high-confidence matches.
"""
    # Use 'gc' and 'ch' prefixes as format_list_for_prompt expects them
    # The dfX_for_prompt DataFrames already have 'gc_id'/'gc_description' and 'ch_id'/'ch_name'
    list1_prompt_lines = format_list_for_prompt(df1_list_for_prompt, "gc") 
    list2_prompt_lines = format_list_for_prompt(df2_list_for_prompt, "ch")

    prompt = (
        prompt_intro +
        f"\n\n--- List 1 ({dataset1_label}) ---\n" +
        "\n".join(list1_prompt_lines) +
        f"\n\n--- List 2 ({dataset2_label}) ---\n" +
        "\n".join(list2_prompt_lines) +
        "\n\n--- Matched Pairs (JSON List Only) ---\n[" # Prime for JSON
    )

    response_text = call_gemini_api(prompt)
    if not response_text:
        log.error("LLM API call failed or returned no response. Cannot proceed with matching.")
        return []

    # Parse response - parse_llm_response expects keys 'gc_id' and 'ch_id'
    raw_matches = parse_llm_response(response_text)
    if not raw_matches:
        log.warning("LLM response parsing failed or yielded no matches.")
        return []

    # Translate matched IDs back to original column names
    translated_matches = []
    for match in raw_matches:
        if 'gc_id' in match and 'ch_id' in match:
            translated_matches.append({
                df1_id_col: match['gc_id'],
                df2_id_col: match['ch_id']
            })
        else:
            log.warning(f"Skipping malformed match from LLM: {match}")

    if not save_matches(translated_matches, output_path):
        log.warning(f"Failed to save matches to {output_path}. Returning matches directly.")
    
    log.info(f"Successfully matched {len(translated_matches)} pairs.")
    return translated_matches

# --- Main Execution ---
def main():
    """Main function to orchestrate the LLM matching process for GC/CH."""
    print("--- Starting LLM Name Matching Script ---")

    # --- Load Data ---
    df_gc_raw, df_ch_raw = load_raw_data(GC_RAW_PATH, CH_RAW_PATH)
    if df_gc_raw is None or df_ch_raw is None:
        print("Error: Failed to load one or both raw datasets. Exiting.", file=sys.stderr)
        sys.exit(1)

    # --- Preprocess Data ---
    df_gc_processed = preprocess_gcatholic(df_gc_raw)
    df_ch_processed = preprocess_catholic_hierarchy(df_ch_raw)

    if df_gc_processed.empty or df_ch_processed.empty:
         print("Error: Preprocessing resulted in empty data for GC or CH. Cannot proceed.", file=sys.stderr)
         sys.exit(1)

    # --- Prepare for Prompt ---
    gc_list_for_prompt = df_gc_processed.to_dict('records')
    ch_list_for_prompt = df_ch_processed.to_dict('records')

    # --- Generate Prompt ---
    prompt = generate_matching_prompt(gc_list_for_prompt, ch_list_for_prompt)
    # print(f"Debug: Prompt start:\n{prompt[:500]}\n...\nPrompt end:\n{prompt[-500:]}") # Optional debug

    # --- Get Matches from LLM (with Retries) ---
    matches = []
    for attempt in range(MAX_LLM_RETRIES):
        print(f"\nAttempting LLM call ({attempt + 1}/{MAX_LLM_RETRIES})...")
        response_text = call_gemini_api(prompt)
        if response_text:
            matches = parse_llm_response(response_text)
            if matches:  # If we got a non-empty list of valid matches, break
                break
            else:
                 print("LLM response received but parsing failed or yielded no valid matches.")
        else:
            print("LLM call failed or returned no text.")

        if attempt < MAX_LLM_RETRIES - 1:
            sleep_time = RETRY_SLEEP_BASE_SECONDS * (attempt + 1)
            print(f"Retrying in {sleep_time} seconds...")
            time.sleep(sleep_time)
        else:
            print("Maximum LLM retries reached.")

    # --- Save Matches ---
    if not matches:
        print("\nFailed to get valid matches from LLM after multiple attempts. No output file generated.", file=sys.stderr)
        # Optionally save the prompt for debugging
        # try:
        #     failed_prompt_path = DATA_DIR / "failed_llm_prompt.txt"
        #     with open(failed_prompt_path, "w", encoding='utf-8') as f:
        #         f.write(prompt)
        #     print(f"Saved failed prompt to {failed_prompt_path}")
        # except Exception as e:
        #      print(f"Could not save failed prompt: {e}", file=sys.stderr)
        sys.exit(1) # Exit if no matches were obtained

    if save_matches(matches, MATCH_OUTPUT_PATH):
        print(f"\n--- LLM Name Matching Script Finished Successfully ({len(matches)} matches found) ---")
    else:
        print("\n--- LLM Name Matching Script Finished with Errors (Save failed) ---", file=sys.stderr)
        sys.exit(1)

# --- Script Execution Guard ---
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # --- Google AI API Key Check ---
    try:
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        log.info("Google AI API Key configured successfully.")
    except KeyError:
        log.error("Error: GOOGLE_API_KEY environment variable not set.")
        sys.exit(1)
    except Exception as e:
        log.error(f"Error configuring Google AI: {e}", exc_info=True)
        sys.exit(1)

    log.info("--- Starting LLM Name Matching Script ---")

    # Check if matches file already exists
    if LLM_MATCHES_PATH.exists():
        log.warning(f"Matches file already exists: {LLM_MATCHES_PATH}")
        log.info("Ensuring raw data files have IDs...")
        # Still load (and save back) raw data to ensure IDs are present
        df_gc_raw, df_ch_raw = load_raw_data(GC_RAW_PATH, CH_RAW_PATH)
        if df_gc_raw is None or df_ch_raw is None:
            log.error("Failed to load or update raw data files even though matches exist. Exiting.")
            sys.exit(1)
        log.info("Raw data files checked/updated. Exiting script as matches already exist.")
        sys.exit(0)
    else:
        # --- Load and Preprocess Data (Only if matches file doesn't exist) ---
        log.info("Matches file not found. Proceeding with LLM matching workflow.")
        log.info("Loading and preprocessing raw data...")
        df_gc_raw, df_ch_raw = load_raw_data(GC_RAW_PATH, CH_RAW_PATH)
        if df_gc_raw is None or df_ch_raw is None:
            log.error("Failed to load initial raw data. Exiting.")
            sys.exit(1)

        log.info("Preprocessing GCatholic data...")
        df_gc_processed = preprocess_gcatholic(df_gc_raw)
        if df_gc_processed is None:
            log.error("Failed to preprocess GCatholic data. Exiting.")
            sys.exit(1)
        log.info(f"GCatholic Electors (Age < 80): {len(df_gc_processed)}")

        log.info("Preprocessing Catholic Hierarchy data...")
        df_ch_processed = preprocess_catholic_hierarchy(df_ch_raw)
        if df_ch_processed is None:
            log.error("Failed to preprocess Catholic Hierarchy data. Exiting.")
            sys.exit(1)

        # --- Generate Prompt ---
        log.info("Generating LLM prompt...")
        prompt = generate_matching_prompt(df_gc_processed.to_dict('records'), df_ch_processed.to_dict('records'))
        if not prompt:
            log.error("Failed to generate LLM prompt. Exiting.")
            sys.exit(1)
        log.info(f"Prompt generated. Length: {len(prompt)} characters.")

        # --- Get Matches from LLM ---
        matches = []
        for attempt in range(MAX_LLM_RETRIES):
            print(f"\nAttempting LLM call ({attempt + 1}/{MAX_LLM_RETRIES})...")
            response_text = call_gemini_api(prompt)
            if response_text:
                matches = parse_llm_response(response_text)
                if matches:  # If we got a non-empty list of valid matches, break
                    break
                else:
                     print("LLM response received but parsing failed or yielded no valid matches.")
            else:
                print("LLM call failed or returned no text.")

            if attempt < MAX_LLM_RETRIES - 1:
                sleep_time = RETRY_SLEEP_BASE_SECONDS * (attempt + 1)
                print(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                print("Maximum LLM retries reached.")

        # --- Save Matches ---
        if not matches:
            print("\nFailed to get valid matches from LLM after multiple attempts. No output file generated.", file=sys.stderr)
            sys.exit(1)

        if save_matches(matches, LLM_MATCHES_PATH):
            print(f"\n--- LLM Name Matching Script Finished Successfully ({len(matches)} matches found) ---")
        else:
            print("\n--- LLM Name Matching Script Finished with Errors (Save failed) ---", file=sys.stderr)
            sys.exit(1)