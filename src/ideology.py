import pandas as pd
import logging
from typing import Dict, List, Optional
import os
import json
import time

# AI: Check if the library is available
try:
    import google.generativeai as genai
except ImportError:
    genai = None
    logging.warning("google-generativeai package not found. LLM scoring will be skipped.")

log = logging.getLogger(__name__)


# --- Papacy Score Calculation ---

PAPACY_SCORE_MAP: Dict[str, int] = {
    'Francis': 1,
    'Benedict XVI': -1,
    'John Paul II': -1,
}

def calculate_papacy_score(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates the papacy_score based on the appointing_pope column.

    Assigns +1 for Francis appointees, -1 for Benedict XVI or John Paul II
    appointees, and 0 for others or missing values.

    Args:
        df: DataFrame containing elector data with an 'appointing_pope' column.

    Returns:
        DataFrame with an added 'papacy_score' column.

    Raises:
        KeyError: If the 'appointing_pope' column is missing.
    """
    if 'appointing_pope' not in df.columns:
        raise KeyError("DataFrame must contain the 'appointing_pope' column to calculate papacy score.")

    log.info("Calculating papacy score based on appointing pope...")

    # Map pope names to scores, default to 0 if not found or NaN
    df['papacy_score'] = df['appointing_pope'].map(PAPACY_SCORE_MAP).fillna(0).astype(int)

    log.debug(f"Papacy score distribution:\n{df['papacy_score'].value_counts()}")
    log.info("Papacy score calculated successfully.")

    return df


# --- LLM Score Calculation ---

# AI: Define Few-Shot Examples and System Prompt (as text above)
FEW_SHOT_EXAMPLES = """
Example 1:
Input: Cardinal Luis Antonio Tagle, Pro-Prefect for the Dicastery for Evangelization, Appointed by Francis.
Explanation: Cardinal Tagle is often seen as a prominent figure aligned with Pope Francis's pastoral vision, emphasizing mercy and outreach. His background and focus generally place him on the progressive side of the spectrum within the College.
score: 0.7

Example 2:
Input: Cardinal Raymond Burke, Patron of the Sovereign Military Order of Malta (until 2023), Appointed by Benedict XVI.
Explanation: Cardinal Burke is a leading voice for traditional liturgy and doctrine, often expressing concerns about modern interpretations. His public stances and alignment with more conservative groups position him clearly on the conservative end.
score: -0.9

Example 3:
Input: Cardinal Pietro Parolin, Secretary of State, Appointed by Francis.
Explanation: As Secretary of State, Cardinal Parolin navigates complex diplomatic and administrative roles. While appointed by Francis, his position often requires a more measured, centrist approach, balancing various factions within the Church.
score: 0.1
"""

SYSTEM_PROMPT = f"""
You are an expert analyst of Vatican affairs. Your task is to assign an ideology score to Catholic cardinals based on available information.
Use the following scale:
-1.0 = Highly Conservative (emphasizes tradition, doctrinal clarity, liturgical discipline)
 0.0 = Centrist (balances different perspectives, focuses on administration or diplomacy)
+1.0 = Highly Progressive (emphasizes pastoral flexibility, social justice, synodality)

Analyze the provided information for the cardinal (name, title, appointing pope).
Provide a brief 2-3 sentence explanation justifying your score, focusing on how the available information relates to the ideological scale.
Finally, output the score as a JSON object with keys "llm_score" (float) and "explanation" (string).

Here are some examples to guide your scoring:
{FEW_SHOT_EXAMPLES}
"""

MODEL_NAME = "gemini-2.5-flash-preview-04-17"

def calculate_llm_score(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates ideology score using an LLM based on cardinal metadata.

    Uses the Google Generative AI API (Gemini Flash) to score cardinals based
    on their name, title, and appointing pope.

    Requires the GOOGLE_API_KEY environment variable to be set.

    Args:
        df: DataFrame containing elector data with 'elector_id', 'name_clean',
            'ch_title', and 'appointing_pope' columns.

    Returns:
        DataFrame with added 'llm_score' (float) and 'llm_explanation' (str) columns.
    """
    if genai is None:
        log.warning("Skipping LLM scoring because google-generativeai package is not installed.")
        df['llm_score'] = None
        df['llm_explanation'] = None
        return df

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        log.error("GOOGLE_API_KEY environment variable not set. Cannot proceed with LLM scoring.")
        # Optionally, raise an error or return df with None values
        df['llm_score'] = None
        df['llm_explanation'] = None
        return df

    genai.configure(api_key=api_key)

    # Check for required input columns
    required_cols = ['elector_id', 'name_clean', 'ch_title', 'appointing_pope']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        log.error(f"Missing required columns for LLM scoring: {missing_cols}")
        df['llm_score'] = None
        df['llm_explanation'] = None
        return df

    log.info(f"Starting LLM ideology scoring for {len(df)} electors using {MODEL_NAME}...")

    model = genai.GenerativeModel(
        MODEL_NAME,
        system_instruction=SYSTEM_PROMPT,
        generation_config=genai.GenerationConfig(temperature=0.0)
    )

    scores: List[Optional[float]] = []
    explanations: List[Optional[str]] = []
    request_count = 0
    start_time = time.time()

    for index, row in df.iterrows():
        elector_id = row['elector_id']
        name = row['name_clean']
        title = row['ch_title']
        pope = row.get('appointing_pope', 'Unknown') # Handle missing pope

        prompt = f"Input: Cardinal {name}, {title}, Appointed by {pope}."

        try:
            log.debug(f"Sending request for elector {elector_id} ({name})...")
            response = model.generate_content(prompt)
            request_count += 1

            # Attempt to parse the JSON response
            try:
                # Gemini API might return text containing ```json ... ```
                json_text = response.text.strip()
                if json_text.startswith('```json'):
                    json_text = json_text[7:]
                if json_text.endswith('```'):
                    json_text = json_text[:-3]
                json_text = json_text.strip()

                result = json.loads(json_text)
                score = result.get('llm_score')
                explanation = result.get('explanation')

                if isinstance(score, (float, int)):
                    scores.append(float(score))
                    # Basic sanity check
                    if not (-1.0 <= score <= 1.0):
                        log.warning(f"LLM returned score outside range [-1, 1] for {elector_id}: {score}")
                else:
                    log.warning(f"LLM response for {elector_id} had invalid score type: {score}. Setting to None.")
                    scores.append(None)

                explanations.append(explanation if isinstance(explanation, str) else None)
                log.debug(f"Received score {score} for elector {elector_id}.")

            except (json.JSONDecodeError, AttributeError, KeyError, TypeError) as json_e:
                log.error(f"Failed to parse LLM JSON response for elector {elector_id} ({name}): {json_e}\nResponse text: '{response.text}'")
                scores.append(None)
                explanations.append(f"Error parsing response: {response.text}")

        except Exception as e:
            log.error(f"Error during LLM API call for elector {elector_id} ({name}): {e}")
            scores.append(None)
            explanations.append(f"API Error: {e}")

        # Simple rate limiting to avoid overwhelming API (adjust as needed)
        if request_count % 10 == 0:
             time.sleep(1) # Sleep 1 second every 10 requests

    end_time = time.time()
    log.info(f"LLM scoring completed in {end_time - start_time:.2f} seconds for {request_count} requests.")

    df['llm_score'] = scores
    df['llm_explanation'] = explanations

    # Log stats about scores
    valid_scores = [s for s in scores if s is not None]
    if valid_scores:
        log.info(f"Generated {len(valid_scores)}/{len(df)} valid LLM scores.")
        log.info(f"LLM Score Stats: Min={min(valid_scores):.2f}, Max={max(valid_scores):.2f}, Avg={sum(valid_scores)/len(valid_scores):.2f}")
    else:
        log.warning("No valid LLM scores were generated.")

    return df


# --- Placeholder for other score calculations (from product plan) ---
# def calculate_statement_score(df: pd.DataFrame) -> pd.DataFrame:
#     # ... implementation ...
#     return df

# def calculate_editorial_score(df: pd.DataFrame) -> pd.DataFrame:
#     # ... implementation ...
#     return df

# def calculate_map_score(df: pd.DataFrame) -> pd.DataFrame:
#     # ... implementation ...
#     return df

# def calculate_thematic_score(df: pd.DataFrame) -> pd.DataFrame:
#     # ... implementation ...
#     return df

# def calculate_composite_ideology_score(df: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
#     # ... implementation using component scores ...
#     # Ensure final score is normalized [-1, 1]
#     return df
