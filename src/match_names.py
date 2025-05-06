# AI: src/match_names.py
# Purpose: Use Gemini LLM to match cardinal names between GCatholic and Catholic Hierarchy datasets.

import os
import pandas as pd
import google.generativeai as genai
import json
import time

# --- Configuration --- 
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
GC_RAW_PATH = os.path.join(DATA_DIR, 'scraped_gcatholic_raw.csv')
CH_RAW_PATH = os.path.join(DATA_DIR, 'scraped_ch_raw.csv')
MATCH_OUTPUT_PATH = os.path.join(DATA_DIR, 'llm_matched_pairs.json')

# --- Load API Key --- 
# IMPORTANT: Set the GOOGLE_API_KEY environment variable before running.
try:
    GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']
    genai.configure(api_key=GOOGLE_API_KEY)
    print("Google AI API Key loaded.")
except KeyError:
    print("Error: GOOGLE_API_KEY environment variable not set.")
    print("Please set the GOOGLE_API_KEY environment variable and try again.")
    exit(1)
except Exception as e:
    print(f"Error configuring Google AI SDK: {e}")
    exit(1)

# --- Helper Functions --- 

def load_data(gc_path: str, ch_path: str) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Loads the raw scraped data.

    Args:
        gc_path: Path to the raw GCatholic CSV.
        ch_path: Path to the raw Catholic Hierarchy CSV.

    Returns:
        A tuple containing the GCatholic DataFrame and Catholic Hierarchy DataFrame, or None if loading fails.
    """
    df_gc, df_ch = None, None
    try:
        df_gc = pd.read_csv(gc_path)
        print(f"Loaded GC Raw: {df_gc.shape}")
    except FileNotFoundError:
        print(f"Error: GCatholic raw file not found at {gc_path}")
    except Exception as e:
        print(f"Error loading GCatholic raw file: {e}")
        
    try:
        df_ch = pd.read_csv(ch_path)
        print(f"Loaded CH Raw: {df_ch.shape}")
    except FileNotFoundError:
        print(f"Error: Catholic Hierarchy raw file not found at {ch_path}")
    except Exception as e:
        print(f"Error loading Catholic Hierarchy raw file: {e}")

    return df_gc, df_ch

def preprocess_gcatholic(df: pd.DataFrame) -> pd.DataFrame:
    """Performs minimal preprocessing on GCatholic data (e.g., age filtering)."""
    print("Preprocessing GCatholic data...")
    # Rename for clarity
    df = df.rename(columns={'Cardinal Electors(135)': 'gc_description', 'Age': 'gc_age', 'Date ofBirth': 'gc_birthdate'})
    # Convert age
    df['gc_age'] = pd.to_numeric(df['gc_age'], errors='coerce')
    # Filter electors (age < 80)
    df_filtered = df[df['gc_age'] < 80].copy()
    print(f"GCatholic Electors (Age < 80): {len(df_filtered)}")
    # Add a unique identifier based on original index
    df_filtered['gc_id'] = df_filtered.index
    return df_filtered[['gc_id', 'gc_description', 'gc_age', 'gc_birthdate']]

def preprocess_catholic_hierarchy(df: pd.DataFrame) -> pd.DataFrame:
    """Performs minimal preprocessing on CH data (e.g., add id)."""
    print("Preprocessing Catholic Hierarchy data...")
    # Rename for clarity
    df = df.rename(columns={'Name': 'ch_name', 'Age': 'ch_age', 'Birthdate': 'ch_birthdate', 'Elevated': 'ch_elevated_date', 'Current Title': 'ch_title'})
    # Add a unique identifier based on original index
    df['ch_id'] = df.index
    return df[['ch_id', 'ch_name', 'ch_age', 'ch_birthdate', 'ch_elevated_date', 'ch_title']]

def generate_matching_prompt(gc_list: list[dict], ch_list: list[dict]) -> str:
    """Creates the prompt for the Gemini LLM to match names."""
    
    # AI: Correctly construct the multi-line prompt string
    prompt_lines = [
        """
    You are an expert data analyst specializing in Catholic Church hierarchy. Your task is to match individuals between two lists of cardinals based on their names and potentially other provided details (like age or birthdate if available and consistent).

    List 1 (GCatholic) contains descriptions like: 'LastName, FirstName, Suffix.(Age)...Title...'
    List 2 (Catholic Hierarchy) contains names like: 'FirstName [MiddleName] CardinalLastName [Suffix]'

    Please compare the two lists below and identify pairs of individuals who are likely the same person. Consider variations in name format, middle names/initials, titles (like C.SS.R., O.F.M.), and potential minor discrepancies in age or birthdate.

    Provide your matches as a JSON list of dictionaries, where each dictionary represents a matched pair and has the keys 'gc_id' and 'ch_id' corresponding to the IDs provided in the input lists.

    Example Output Format: [{'gc_id': 10, 'ch_id': 5}, {'gc_id': 25, 'ch_id': 112}]

    If you are uncertain about a match, do not include it.

    --- List 1 (GCatholic) ---
        """
    ]
    
    # Add GCatholic items
    for item in gc_list:
        prompt_lines.append(f"gc_id: {item['gc_id']}, description: {item['gc_description']}, age: {item.get('gc_age', 'N/A')}, birthdate: {item.get('gc_birthdate', 'N/A')}")
    
    prompt_lines.append("\n--- List 2 (Catholic Hierarchy) ---")
    
    # Add Catholic Hierarchy items
    for item in ch_list:
        prompt_lines.append(f"ch_id: {item['ch_id']}, name: {item['ch_name']}, age: {item.get('ch_age', 'N/A')}, birthdate: {item.get('ch_birthdate', 'N/A')}, title: {item.get('ch_title', 'N/A')}")
        
    prompt_lines.append("\n--- Matched Pairs (JSON List) ---")
    # Add a placeholder for the LLM to start generating JSON
    prompt_lines.append("[") 
    
    # Join all lines into the final prompt string
    prompt = "\n".join(prompt_lines)
    
    return prompt

def get_llm_matches(prompt: str, model_name: str = "gemini-2.5-flash-preview-04-17") -> list[dict]:
    """Sends the prompt to the Gemini model and parses the JSON response."""
    print(f"Sending prompt to Gemini model ({model_name})... Length: {len(prompt)} chars")
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        
        # Clean the response: remove markdown backticks and 'json' identifier
        cleaned_response_text = response.text.strip().replace('```json', '').replace('```', '').strip()
        print(f"LLM Raw Response Snippet: {cleaned_response_text[:200]}...") # Print snippet

        # Parse the JSON response
        matches = json.loads(cleaned_response_text)
        if isinstance(matches, list) and all(isinstance(item, dict) and 'gc_id' in item and 'ch_id' in item for item in matches):
            print(f"Successfully parsed {len(matches)} matches from LLM response.")
            return matches
        else:
            print("Error: LLM response was not in the expected JSON format.")
            print(f"Full Response: {cleaned_response_text}")
            return []
            
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from LLM response: {e}")
        print(f"Full Response: {cleaned_response_text}")
        return []
    except Exception as e:
        print(f"Error interacting with Gemini API: {e}")
        # Fallback: Check if parts exist (for safety/content filtering issues)
        try:
             print(f"Response parts: {response.parts}")
        except Exception:
             pass
        return []

# --- Main Execution --- 
if __name__ == "__main__":
    print("--- Starting LLM Name Matching Script ---")
    
    df_gc_raw, df_ch_raw = load_data(GC_RAW_PATH, CH_RAW_PATH)
    
    if df_gc_raw is None or df_ch_raw is None:
        print("Error: Failed to load one or both raw datasets. Exiting.")
        exit(1)
        
    # Preprocess
    df_gc_processed = preprocess_gcatholic(df_gc_raw)
    df_ch_processed = preprocess_catholic_hierarchy(df_ch_raw)
    
    # Convert DataFrames to lists of dictionaries for the prompt
    gc_list_for_prompt = df_gc_processed.to_dict('records')
    ch_list_for_prompt = df_ch_processed.to_dict('records')
    
    # Generate Prompt
    prompt = generate_matching_prompt(gc_list_for_prompt, ch_list_for_prompt)
    # print("\n--- Generated Prompt ---")
    # print(prompt[:1000] + "...") # Print beginning of prompt for verification
    # print("-" * 20)
    
    # Get Matches from LLM
    # Adding retry logic in case of transient API issues
    max_retries = 3
    matches = []
    for attempt in range(max_retries):
        matches = get_llm_matches(prompt)
        if matches: # If we got a non-empty list of matches, break
             break
        elif attempt < max_retries - 1: # Don't sleep on last attempt
             print(f"Retrying LLM call (attempt {attempt + 2}/{max_retries})...")
             time.sleep(5 * (attempt + 1)) # Exponential backoff
             
    if not matches:
        print("Failed to get matches from LLM after multiple attempts.")
        # Optionally save the prompt for debugging
        # with open("failed_prompt.txt", "w") as f:
        #     f.write(prompt)
        # print("Saved failed prompt to failed_prompt.txt")
        exit(1)

    # Save Matches
    try:
        with open(MATCH_OUTPUT_PATH, 'w') as f:
            json.dump(matches, f, indent=4)
        print(f"Successfully saved {len(matches)} matched pairs to {MATCH_OUTPUT_PATH}")
    except Exception as e:
        print(f"Error saving matched pairs: {e}")

    print("--- LLM Name Matching Script Finished ---")
