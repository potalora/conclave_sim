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
import re

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
# AI: Using os.getenv for safer access, allowing None if not set
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        log.info("Google AI API Key configured successfully.")
    except Exception as e:
        log.error(f"Error configuring Google AI SDK with provided GOOGLE_API_KEY: {e}", exc_info=True)
        # LLM features will likely fail if this occurs.
else:
    log.warning("GOOGLE_API_KEY environment variable not set. LLM-dependent features will be unavailable.")

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


def _sanitize_value_for_fstring(value: Any) -> Any:
    """Escapes curly braces in strings to be safely embedded in f-strings."""
    if isinstance(value, str):
        return value.replace("{", "{{").replace("}", "}}")
    return value


def format_list_for_prompt(data_list: List[Dict[str, Any]], id_key_in_dict: str) -> List[str]:
    """Formats a list of dictionaries for inclusion in the LLM prompt.

    Args:
        data_list: The list of dictionaries to format (e.g., preprocessed GC or CH data).
        id_key_in_dict: The key name in each dictionary that holds the ID.

    Returns:
        A list of strings, where each string is a formatted line for the prompt.
    """
    formatted_lines = []
    for item in data_list:
        # Ensure essential keys exist to prevent KeyErrors
        name_desc = item.get('name_desc', 'Name/Description N/A')
        item_id = item.get(id_key_in_dict, 'ID N/A') # Use the dynamic id_key_in_dict
        line = f"{item_id}: {name_desc}"
        formatted_lines.append(line)
    return formatted_lines

def extract_json_from_llm_response(text: str, expected_id_key1: str, expected_id_key2: str) -> Optional[List[Dict[str, Any]]]:
    """Parses the JSON string from the LLM response, robustly handling preambles."""
    if not text:
        log.error("Cannot parse LLM response: Input text is None or empty.")
        return [] 

    log.debug("Attempting to parse LLM Response Snippet: %s", text[:300].strip() + "..." if len(text) > 300 else text.strip())
    cleaned_text = text.strip()
    json_str = "" 

    json_blocks = re.finditer(r"```json\s*([\s\S]*?)\s*```", cleaned_text, re.IGNORECASE)
    parsed_successfully = False
    validated_matches = []
    for i, json_match in enumerate(json_blocks):
        json_str = json_match.group(1).strip()
        log.info(f"Found JSON block {i+1}. Attempting to parse.")
        # Remove common preamble/postamble lines that might be mistakenly included by LLM inside the block
        json_str_lines = json_str.split('\n')
        cleaned_lines = []
        in_json_structure = False
        for line in json_str_lines:
            stripped_line = line.strip()
            if not in_json_structure:
                if stripped_line.startswith('[') or stripped_line.startswith('{'):
                    in_json_structure = True
                    cleaned_lines.append(line) # Keep the line that starts the JSON
            elif in_json_structure:
                cleaned_lines.append(line)
        
        if not cleaned_lines:
            log.warning(f"JSON block {i+1} was empty after basic cleaning.")
            continue

        json_str_cleaned = '\n'.join(cleaned_lines)

        try:
            # Attempt to parse the cleaned JSON string
            parsed_llm_output = json.loads(json_str_cleaned)
            log.debug(f"Successfully parsed JSON from block {i+1}.")
            parsed_successfully = True # Mark that we found and parsed a block
        except json.JSONDecodeError as e:
            log.warning(f"Could not parse JSON from block {i+1}. Error: {e}. Snippet: {json_str_cleaned[:200]}...")
            continue # Try next block

        if not isinstance(parsed_llm_output, list):
            log.warning(f"Parsed JSON from block {i+1} is not a list, but type {type(parsed_llm_output)}. Skipping.")
            continue

        if not parsed_llm_output: # Empty list
            log.warning(f"Parsed JSON list from block {i+1} is empty. Skipping.")
            continue

        # If we got here, this block parsed into a non-empty list. Proceed with validation.
        log.info(f"Successfully parsed JSON from block {i+1} into a list of {len(parsed_llm_output)} items. Validating items...")
        for item_idx, item in enumerate(parsed_llm_output):
            log.debug(f"Processing item {item_idx} from block {i+1}: {item}")
            if not (isinstance(item, dict) and expected_id_key1 in item and expected_id_key2 in item):
                log.warning(f"Skipping item {item_idx} due to unexpected structure or missing keys: {item}")
                continue 
            
            try:
                id1_val = item[expected_id_key1]
                id2_val = item[expected_id_key2]
                log.debug(f"  Item {item_idx}: id1_val='{id1_val}' (type: {type(id1_val)}), id2_val='{id2_val}' (type: {type(id2_val)})")
                
                str_id1 = str(id1_val).strip()
                str_id2 = str(id2_val).strip()
                log.debug(f"  Item {item_idx}: str_id1='{str_id1}', str_id2='{str_id2}'")

                # Validation checks
                check_str_id1_non_empty = bool(str_id1)
                check_str_id2_non_empty = bool(str_id2)
                check_str_id1_chars = all(c in '0123456789-' for c in str_id1.lstrip('-')) if str_id1 else False
                check_str_id2_chars = all(c in '0123456789-' for c in str_id2.lstrip('-')) if str_id2 else False
                check_str_id1_hyphens = str_id1.count('-') <= 1
                check_str_id2_hyphens = str_id2.count('-') <= 1
                check_str_id1_leading_zeros = (str_id1 == '0' or not str_id1.startswith('0') or str_id1.startswith('-0'))
                check_str_id2_leading_zeros = (str_id2 == '0' or not str_id2.startswith('0') or str_id2.startswith('-0'))

                log.debug(f"  Item {item_idx} Validations: non_empty1={check_str_id1_non_empty}, non_empty2={check_str_id2_non_empty}, "
                          f"chars1={check_str_id1_chars}, chars2={check_str_id2_chars}, "
                          f"hyphens1={check_str_id1_hyphens}, hyphens2={check_str_id2_hyphens}, "
                          f"leading_zeros1={check_str_id1_leading_zeros}, leading_zeros2={check_str_id2_leading_zeros}")

                if not (check_str_id1_non_empty and check_str_id2_non_empty and \
                        check_str_id1_chars and check_str_id2_chars and \
                        check_str_id1_hyphens and check_str_id2_hyphens and \
                        check_str_id1_leading_zeros and check_str_id2_leading_zeros
                       ) :
                    log.warning(f"Skipping item {item_idx} due to invalid ID format: id1='{str_id1}', id2='{str_id2}'")
                    continue

                item[expected_id_key1] = int(str_id1)
                item[expected_id_key2] = int(str_id2)
                validated_matches.append(item)
                log.debug(f"  Item {item_idx} successfully validated and added.")
            except ValueError as ve:
                log.warning(f"Skipping item {item_idx} due to ValueError during ID conversion (id1='{str_id1}', id2='{str_id2}'): {ve}")
                continue
            except Exception as e:
                log.error(f"Skipping item {item_idx} due to unexpected error: {e}. Item data: {item}")
                continue
        
        if validated_matches: # If this block yielded any valid matches, we are done.
            log.info(f"Finished processing block {i+1}. Found {len(validated_matches)} validated matches.")
            break # Exit the loop over json_blocks
        else:
            log.warning(f"Block {i+1} parsed correctly but yielded no validated matches after item-level validation.")
            # Continue to the next block if any

    if not parsed_successfully and not validated_matches:
        # Fallback 1: If no ```json ``` block worked, try to parse the whole response as JSON (if it's very simple)
        log.info("No valid JSON found in ```json ... ``` blocks, or blocks yielded no valid items. Attempting fallback 1: parse entire response if it's a simple JSON list.")
        if len(cleaned_text.strip()) > 0 and (cleaned_text.strip().startswith('[') and cleaned_text.strip().endswith(']')):
            try:
                parsed_llm_output = json.loads(cleaned_text.strip()) # Try parsing the whole thing
                if isinstance(parsed_llm_output, list) and parsed_llm_output:
                    log.info(f"Fallback 1: Successfully parsed entire response as a JSON list of {len(parsed_llm_output)} items. Validating items...")
                    for item_idx, item in enumerate(parsed_llm_output):
                        is_valid_item, id1_val_str, id2_val_str = _validate_llm_item(item, item_idx, expected_id_key1, expected_id_key2, "fallback1_block")
                        if is_valid_item:
                            validated_matches.append({expected_id_key1: id1_val_str, expected_id_key2: id2_val_str})
                    if validated_matches:
                        log.info(f"Fallback 1 yielded {len(validated_matches)} validated matches.")
                        # No break here, as we want to prefer ```json``` blocks if they ever work.
                        # This path is taken only if validated_matches is still empty.
                else:
                    log.warning("Fallback 1: Entire response parsed but was not a non-empty list.")
            except json.JSONDecodeError as e:
                log.warning(f"Fallback 1: Could not parse entire response as JSON. Error: {e}")
        else:
            log.info("Fallback 1: Entire response does not appear to be a simple JSON list, skipping.")

    # Fallback 2: If still no matches, try to find a bare JSON list pattern (more robust than Fallback 1)
    if not validated_matches: # Check again, Fallback 1 might have populated it
        log.info("Still no validated matches. Attempting Fallback 2: find bare JSON list pattern.")
        # This regex looks for a pattern starting with '[', containing at least one '{...}', and ending with ']'
        # It captures the outermost list.
        bare_json_regex = r"(\[\s*(?:\{[\s\S]*?\}\s*,\s*)*\{[\s\S]*?\}\s*\])"
        bare_json_finds = list(re.finditer(bare_json_regex, cleaned_text))
        
        if bare_json_finds:
            # Take the last found bare JSON array, as LLMs often put preamble first.
            json_str_candidate = bare_json_finds[-1].group(1)
            candidate_snippet_for_log = json_str_candidate[:150].replace('\n', ' ')
            log.info(f"Fallback 2: Found potential bare JSON list. Attempting to parse candidate: {candidate_snippet_for_log}...")
            try:
                parsed_llm_output = json.loads(json_str_candidate)
                if isinstance(parsed_llm_output, list) and parsed_llm_output:
                    log.info(f"Fallback 2: Successfully parsed bare JSON candidate into a list of {len(parsed_llm_output)} items. Validating items...")
                    for item_idx, item in enumerate(parsed_llm_output):
                        is_valid_item, id1_val_str, id2_val_str = _validate_llm_item(item, item_idx, expected_id_key1, expected_id_key2, "fallback2_block")
                        if is_valid_item:
                            validated_matches.append({expected_id_key1: id1_val_str, expected_id_key2: id2_val_str})
                    if validated_matches:
                         log.info(f"Fallback 2 yielded {len(validated_matches)} validated matches.")
                else:
                    log.warning("Fallback 2: Bare JSON candidate parsed but was not a non-empty list.")
            except json.JSONDecodeError as e:
                log.warning(f"Fallback 2: Could not parse bare JSON candidate. Error: {e}. Candidate snippet: {json_str_candidate[:200]}...")
        else:
            log.warning("Fallback 2: Could not find any bare JSON list pattern in the response.")

    if not validated_matches:
        log.warning("LLM returned an empty list of matches or parsing failed for all attempts (including fallbacks).")
        # AI: Adding more context to the log when no matches are found
        log.debug(f"Full raw LLM output when no matches found:\n{cleaned_text}")

    return validated_matches


def _validate_llm_item(item: Any, item_idx: int, id1_key: str, id2_key: str, block_info: str) -> tuple[bool, str | None, str | None]:
    """Validates a single item from the parsed LLM JSON output."""
    log.debug(f"Processing item {item_idx} from {block_info}: {item}")
    if not isinstance(item, dict):
        log.warning(f"  Item {item_idx} from {block_info} is not a dictionary. Skipping.")
        return False, None, None

    id1_val = item.get(id1_key)
    id2_val = item.get(id2_key)

    log.debug(f"  Item {item_idx} from {block_info}: id1_val='{id1_val}' (type: {type(id1_val)}), id2_val='{id2_val}' (type: {type(id2_val)})")

    # Convert IDs to string and perform basic validation
    try:
        # Allow integers or strings that can be cast to string
        str_id1 = str(id1_val) if id1_val is not None else ""
        str_id2 = str(id2_val) if id2_val is not None else ""
    except Exception as e:
        log.warning(f"  Item {item_idx} from {block_info}: Error converting IDs to string ('{id1_val}', '{id2_val}'): {e}. Skipping.")
        return False, None, None
    
    log.debug(f"  Item {item_idx} from {block_info}: str_id1='{str_id1}', str_id2='{str_id2}'")

    # Validation checks (as originally designed)
    non_empty1 = bool(str_id1)
    non_empty2 = bool(str_id2)
    # Allow only digits, hyphens, or specific alphanumeric patterns if necessary in future
    # For now, simple digit check for basic numeric IDs from range(), but allow general strings
    # As Conclavoscope IDs are numeric (0-indexed) and elector_id are also numeric
    chars1 = str_id1.isalnum() or all(c.isdigit() or c == '-' for c in str_id1) if str_id1 else False 
    chars2 = str_id2.isalnum() or all(c.isdigit() or c == '-' for c in str_id2) if str_id2 else False
    # Max length check (e.g., up to 10 chars, adjust if IDs can be longer)
    len1 = len(str_id1) <= 10
    len2 = len(str_id2) <= 10
    # Check for disallowed characters or patterns (e.g. leading/trailing hyphens if problematic)
    hyphens1 = not (str_id1.startswith('-') or str_id1.endswith('-')) if '-' in str_id1 else True
    hyphens2 = not (str_id2.startswith('-') or str_id2.endswith('-')) if '-' in str_id2 else True
    # Check for leading zeros if IDs are purely numeric and this is significant (e.g. '01' vs '1')
    # For now, assume string comparison handles this, but can be more specific:
    leading_zeros1 = not (str_id1.startswith('0') and len(str_id1) > 1 and str_id1.isdigit()) if str_id1 else True
    leading_zeros2 = not (str_id2.startswith('0') and len(str_id2) > 1 and str_id2.isdigit()) if str_id2 else True

    all_validations = {
        "non_empty1": non_empty1,
        "non_empty2": non_empty2,
        "chars1": chars1,
        "chars2": chars2,
        "len1": len1,
        "len2": len2,
        "hyphens1": hyphens1,
        "hyphens2": hyphens2,
        "leading_zeros1": leading_zeros1,
        "leading_zeros2": leading_zeros2
    }
    log.debug(f"  Item {item_idx} from {block_info} Validations: {all_validations}")

    if all(all_validations.values()):
        return True, str_id1, str_id2
    else:
        failed_checks = {k:v for k,v in all_validations.items() if not v}
        log.warning(f"  Item {item_idx} from {block_info} failed validation checks: {failed_checks}. Values: id1='{str_id1}', id2='{str_id2}'. Skipping.")
        return False, None, None

def call_gemini_api(prompt: str, model_name: str = GEMINI_MODEL_NAME, expected_id_key1: str = "gc_id", expected_id_key2: str = "ch_id") -> Optional[List[Dict[str, Any]]]:
    """Sends the prompt to the Gemini model and returns the text response.

    Handles potential API errors and provides limited response info on failure.

    Args:
        prompt: The prompt string to send to the LLM.
        model_name: The name of the Gemini model to use.
        expected_id_key1: The expected key for the first ID in the JSON response.
        expected_id_key2: The expected key for the second ID in the JSON response.

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
            print("Warning: Prompt was blocked. Reason:", response.prompt_feedback.block_reason, file=sys.stderr)
            return None
        if not response.candidates:
             print("Warning: No candidates returned by the API.", file=sys.stderr)
             return None

        actual_finish_reason = response.candidates[0].finish_reason
        
        # Dynamically get the FinishReason enum type from an instance
        FinishReasonEnum = type(actual_finish_reason)
        expected_stop_reason = FinishReasonEnum.STOP

        if actual_finish_reason != expected_stop_reason:
            print(f"Warning: Generation did not finish with STOP. Reason: {actual_finish_reason.name if hasattr(actual_finish_reason, 'name') else actual_finish_reason}", file=sys.stderr)
            # Try to extract text even if finish reason is not STOP, parts might exist
            if response.candidates[0].content and response.candidates[0].content.parts:
                print("Attempting to extract partial text...")
                try:
                    # Concatenate text from all parts that have a text attribute
                    extracted_text = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text'))
                    if extracted_text:
                        return extracted_text
                    else:
                        print("No text found in content parts despite unexpected finish reason.", file=sys.stderr)
                        return None
                except Exception as e_text_extract:
                    print(f"Error extracting partial text: {e_text_extract}", file=sys.stderr)
                    return None
            else:
                print("No content or parts available for unexpected finish reason.", file=sys.stderr)
                return None

        print(f"Received response from Gemini (finish reason: {actual_finish_reason.name if hasattr(actual_finish_reason, 'name') else actual_finish_reason}).")
        # If finish_reason is STOP (or if we fell through), use response.text for robust text extraction.
        # The response.text property will raise ValueError if text can't be extracted.
        raw_llm_output = response.text

        parsed_matches = extract_json_from_llm_response(raw_llm_output, expected_id_key1, expected_id_key2)

        if parsed_matches is not None: # Check for None first (parsing didn't fail outright)
            print(f"Successfully parsed {len(parsed_matches)} matches from LLM response.")
            if not parsed_matches: # If it's an empty list
                # AI: Log full raw output if an empty list is parsed, to diagnose prompt/model behavior
                print(f"LLM returned an empty list of matches. Full raw LLM output:\n{raw_llm_output}", file=sys.stderr)
            return parsed_matches
        else: # parsed_matches is None (parsing failed)
            print("LLM response parsing failed or yielded no matches (parser returned None).", file=sys.stderr)
            print(f"Full raw LLM output on parsing failure (parser returned None):\n{raw_llm_output}", file=sys.stderr)
            return None

    except AttributeError as ae:
         print("Error: Potentially invalid response structure from API:", ae, file=sys.stderr)
         # Log the raw response if possible without causing further errors
         try:
            print("Raw response object type:", type(response), file=sys.stderr)
            # print(f"Raw response content: {response}", file=sys.stderr) # Avoid printing potentially huge objects
         except Exception as log_e:
             print("Could not log raw response details:", log_e, file=sys.stderr)
         return None
    except Exception as e:
        print("Error interacting with Gemini API:", e, file=sys.stderr)
        # Attempt to log more details if available in the exception
        if hasattr(e, 'response'):
            print("API Error Response:", e.response, file=sys.stderr)
        raise # Changed from: return None

def generate_matching_prompt(gc_list: List[Dict[str, Any]], ch_list: List[Dict[str, Any]], id1_key: str = "gc_id", id2_key: str = "ch_id") -> str:
    """Creates the prompt for the Gemini LLM to match names.

    Args:
        gc_list: Preprocessed GCatholic data as a list of dictionaries.
        ch_list: Preprocessed Catholic Hierarchy data as a list of dictionaries.
        id1_key: The key name for IDs from the first list (gc_list) to be used in the prompt output rules.
        id2_key: The key name for IDs from the second list (ch_list) to be used in the prompt output rules.

    Returns:
        The complete prompt string for the LLM.
    """
    log.info("Generating LLM prompt...")
    # AI: Updated prompt_header to use f-string and dynamic id1_key, id2_key. Note {{}} for literal curly braces in f-string.
    prompt_header = f"""
You are an expert data analyst specializing in Catholic Church hierarchy. Your task is to match individuals between two lists of cardinals based on their names and potentially other provided details (like age or birthdate if available and consistent).

List 1 contains descriptions like: 'LastName, FirstName, Suffix.(Age)...Title...'
List 2 contains names like: 'FirstName [MiddleName] CardinalLastName [Suffix]'

Please compare the two lists below and identify pairs of individuals who are likely the same person. Consider variations in name format, middle names/initials, titles (like C.SS.R., O.F.M.), and potential minor discrepancies in age or birthdate.

Provide your matches **only** as a valid JSON list of dictionaries, where each dictionary represents a matched pair and has the keys '{id1_key}' and '{id2_key}' corresponding to the IDs provided in the input lists. Do not include any text before or after the JSON list. Ensure the JSON is perfectly formatted.

Example Output Format:
[{{{{"{id1_key}": 10, "{id2_key}": 5}}}}, {{{{"{id1_key}": 25, "{id2_key}": 112}}}}] 

If you are uncertain about a match, do not include it. Only include high-confidence matches.

--- List 1 ---
"""
    # AI: Pass the dynamic id keys to format_list_for_prompt
    gc_prompt_lines = format_list_for_prompt(gc_list, id1_key)
    ch_prompt_lines = format_list_for_prompt(ch_list, id2_key)

    prompt = (
        prompt_header +
        "\n".join(gc_prompt_lines) +
        "\n\n--- List 2 ---\n" +
        "\n".join(ch_prompt_lines) +
        "\n\n--- Matched Pairs (JSON List Only) ---\n[" # Prime the LLM for JSON list output
    )
    print(f"Prompt generated. Length: {len(prompt)} characters.")
    return prompt

def prepare_dataframe_for_prompt(
    df: pd.DataFrame,
    name_col: str,
    id_col_original_name: str,
    id_col_for_prompt: str
) -> List[Dict[str, Any]]:
    """Prepares a DataFrame for prompt generation by creating a list of dictionaries.
    Each dictionary contains the original ID (with its key set to id_col_for_prompt) 
    and a name/description field.
    """
    output_list = []
    for _, row in df.iterrows():
        name_desc_val = str(row[name_col]) # Make sure it's a string
        # Ensure the ID is taken from its original column
        id_val = row[id_col_original_name]
        # The key for the ID in the output dictionary is id_col_for_prompt
        output_list.append({
            id_col_for_prompt: id_val, 
            'name_desc': name_desc_val # Generic key for name/description
        })
    return output_list

def match_datasets_llm(
    df1: pd.DataFrame,
    df1_name_col: str,
    df1_id_col: str,
    df2: pd.DataFrame,
    df2_name_col: str,
    df2_id_col: str,
    output_path: Path, # Added output_path for consistency, though not directly used by LLM part here
    dataset1_label: str,
    dataset2_label: str,
) -> List[Dict[str, Any]]:
    """Matches records between two DataFrames using an LLM.

    Args:
        df1: First DataFrame.
        df1_name_col: Name column in df1.
        df1_id_col: ID column in df1.
        df2: Second DataFrame.
        df2_name_col: Name column in df2.
        df2_id_col: ID column in df2.
        output_path: Path to save matched pairs (used by calling functions, not directly here).
        dataset1_label: Label for the first dataset (e.g., 'GCatholic').
        dataset2_label: Label for the second dataset (e.g., 'Catholic Hierarchy').

    Returns:
        A list of dictionaries, where each dictionary represents a matched pair
        with keys corresponding to df1_id_col and df2_id_col.
    """
    log.info(f"Starting LLM matching for {dataset1_label} ({df1_id_col}) and {dataset2_label} ({df2_id_col}).")
    log.info(f"Dataset 1 ({dataset1_label}) has {len(df1)} records. Name col: {df1_name_col}, ID col: {df1_id_col}")
    log.info(f"Dataset 2 ({dataset2_label}) has {len(df2)} records. Name col: {df2_name_col}, ID col: {df2_id_col}")

    # Select relevant columns and handle potential missing columns gracefully
    df1_cols_to_select = [df1_name_col, df1_id_col] + [col for col in ['Age', 'BirthDate'] if col in df1.columns]
    df1_subset = df1[df1_cols_to_select].copy()
    df2_cols_to_select = [df2_name_col, df2_id_col] + [col for col in ['Age', 'BirthDate'] if col in df2.columns]
    df2_subset = df2[df2_cols_to_select].copy()

    # AI: Use prepare_dataframe_for_prompt, passing dfX_id_col as the id_col_for_prompt.
    # This ensures the dictionaries in dfX_prompt_data use dfX_id_col as their ID key.
    df1_prompt_data = prepare_dataframe_for_prompt(df1_subset, df1_name_col, df1_id_col, df1_id_col)
    df2_prompt_data = prepare_dataframe_for_prompt(df2_subset, df2_name_col, df2_id_col, df2_id_col)

    # AI: Call generate_matching_prompt with id1_key=df1_id_col and id2_key=df2_id_col.
    # This tells generate_matching_prompt to instruct the LLM to use these keys in the JSON output,
    # and format_list_for_prompt (called by generate_matching_prompt) will use these keys to read from dfX_prompt_data.
    prompt = generate_matching_prompt(df1_prompt_data, df2_prompt_data, id1_key=df1_id_col, id2_key=df2_id_col)

    # AI: Call call_gemini_api with expected_id_key1=df1_id_col and expected_id_key2=df2_id_col.
    # This ensures extract_json_from_llm_response (called by call_gemini_api) validates against these dynamic keys.
    raw_response_matches = call_gemini_api(prompt, GEMINI_MODEL_NAME, expected_id_key1=df1_id_col, expected_id_key2=df2_id_col)

    if raw_response_matches is None:
        log.warning(f"LLM call returned None for {dataset1_label} and {dataset2_label}. Returning empty list.")
        return []

    log.info(f"Successfully received {len(raw_response_matches)} matches from LLM for {dataset1_label} vs {dataset2_label}.")
    
    # Validate that the IDs are integers and keys match dfX_id_col
    validated_matches = []
    for match in raw_response_matches:
        try:
            id1_val = int(match[df1_id_col])
            id2_val = int(match[df2_id_col])
            validated_matches.append({df1_id_col: id1_val, df2_id_col: id2_val})
        except (ValueError, KeyError, TypeError) as e:
            log.error(f"Skipping invalid match: {match}. Error: {e}")
            continue

    return validated_matches

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
            matches = extract_json_from_llm_response(response_text, "gc_id", "ch_id")
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
                matches = extract_json_from_llm_response(response_text, "gc_id", "ch_id")
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