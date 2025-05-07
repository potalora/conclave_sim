import sys
import json
import time
import logging
import argparse
from pathlib import Path
from bs4 import BeautifulSoup
import re
from typing import List, Dict, Any
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

# Assuming setup_logging is in src.utils
try:
    from .utils import setup_logging
except ImportError:
    # Allow running as a script for testing
    setup_logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

log = logging.getLogger(__name__)

CONCLAVOSCOPE_URL = "https://conclavoscope.com/"

def extract_cardinal_data_from_table(soup: BeautifulSoup) -> List[Dict[str, Any]]:
    """Extracts cardinal data from the HTML table.

    Args:
        soup: BeautifulSoup object of the rendered page HTML.

    Returns:
        A list of dictionaries, each representing a cardinal.
    """
    cardinal_data = []
    # Find the table - Selector based on inspection of conclavoscope.com (may need updates)
    table = soup.find('table', id='cardinalsTable') # Or use class name if no ID

    if not table:
        log.error("Could not find the cardinal table.")
        raise ValueError("Could not find the cardinal table.")

    tbody = table.find('tbody')
    if not tbody:
        log.error("Could not find tbody within the table.")
        raise ValueError("Could not find tbody within the cardinal table.")

    rows = tbody.find_all('tr')
    log.info(f"Found {len(rows)} rows in the table body.")

    # --- Flexible Header Detection --- # Needs adjustment based on table structure
    header_row = table.find('thead').find('tr')
    headers = [th.get_text(strip=True).lower() for th in header_row.find_all('th')]
    log.info(f"Detected headers: {headers}")

    # Attempt to find indices for required columns (adjust names as needed)
    try:
        # Look for common variations of column names
        name_col_idx = headers.index('name')
        score_col_idx = headers.index('total position') # Adjusted to 'total position'

        # These might not be present, handle gracefully if needed or remove if not essential now
        papabile_col_idx = headers.index('papabile score') if 'papabile score' in headers else -1 
        alignment_col_idx = headers.index('alignment score') if 'alignment score' in headers else -1

    except ValueError as e:
        log.error(f"Missing required column in table headers: {e}. Headers found: {headers}")
        raise ValueError("Missing required column in table headers.")

    for row in rows:
        td_content = [td.get_text(strip=True) for td in row.find_all('td')]

        full_name_str = td_content[name_col_idx]
        total_score_str = td_content[score_col_idx]

        # Regex v4: 1:Name(non-greedy), 2:Age, 3:Papabile, 4:Country with dash, 5:Appended (CapWords)
        name_age_papabile_country_match = re.match(
            r"^(.+?)"  # Group 1: Name (non-greedy)
            r"(?:\s\((\d+)\))?"  # Group 2: Optional Age
            r"(?:\s*(Papabile))?"  # Group 3: Optional Papabile status
            r"(?:" # Start of country alternatives (optional group)
            r"\s*-\s*(.+)"  # Group 4: Country with ' - ' separator
            r"|" # OR
            # Group 5: Appended country (Specific: Cap Word, then optional space + Cap Word...)
            r"(?:(?:\s+)|(?<=[a-zA-Z]))(([A-Z][a-z]+(?:\s[A-Z][a-z]+)*))" # Added space constraint for multi-word
            r")?$", # End of country alternatives, make it optional, end of string
            full_name_str
        )

        if name_age_papabile_country_match:
            cardinal_name = name_age_papabile_country_match.group(1).strip()
            # age_str = name_age_papabile_country_match.group(2) # Available if needed
            # is_papabile = bool(name_age_papabile_country_match.group(3)) # Available if needed

            country_with_dash_sep = name_age_papabile_country_match.group(4)
            country_appended = name_age_papabile_country_match.group(5)

            if country_with_dash_sep:
                country = country_with_dash_sep.strip()
            elif country_appended:
                country = country_appended.strip()
            else:
                country = "Unknown"
        else:
            log.warning(f"Primary regex failed to parse name/country/age from: '{full_name_str}'. Using full string as name.")
            cardinal_name = full_name_str # Fallback
            country = "Unknown"

        # Parse total_score (e.g., "64/100" or "64")
        score_val = None
        score_match = re.match(r"(\d+)(?:/\d+)?", total_score_str) # Made /100 part optional
        if score_match:
            score_val = int(score_match.group(1))
        else:
            log.warning(f"Could not parse score from: {total_score_str}")

        # Get other scores if columns exist
        papabile_score = td_content[papabile_col_idx] if papabile_col_idx != -1 else None
        alignment_score = td_content[alignment_col_idx] if alignment_col_idx != -1 else None

        cardinal_data.append({
            'id': None,  # cs_id is not available in this HTML table
            'name': cardinal_name,
            'cs_country': country,
            'total_score': score_val,
            'papabile_score': papabile_score, # Keep these if they are useful, format as needed
            'alignment_score': alignment_score
        })

    return cardinal_data


def fetch_html_with_playwright(url: str, wait_selector: str = 'table#cardinalsTable', wait_timeout: int = 10000) -> str:
    """Fetches the fully rendered HTML of a page using Playwright.

    Args:
        url: The URL to scrape.
        wait_selector: The CSS selector to wait for before grabbing HTML.
        wait_timeout: Maximum time in milliseconds to wait.

    Returns:
        The HTML content of the page as a string.
    """
    html_content = ""
    log.info(f"Launching Playwright to fetch {url}")
    try:
        with sync_playwright() as p:
            # Try chromium first, add fallbacks if needed
            browser = p.chromium.launch() # Add headless=False for debugging
            page = browser.new_page()
            log.info(f"Navigating to {url}...")
            page.goto(url, timeout=60000) # Increased navigation timeout
            log.info(f"Waiting for selector '{wait_selector}' (timeout: {wait_timeout}ms)...")
            page.wait_for_selector(wait_selector, timeout=wait_timeout)
            log.info("Selector found. Getting page content...")
            html_content = page.content()
            log.info(f"Successfully retrieved HTML content (length: {len(html_content)}).")
            browser.close()
    except PlaywrightTimeoutError:
        log.error(f"Playwright timed out waiting for selector '{wait_selector}' at {url}")
        # Optionally, try to get content anyway or raise
        # html_content = page.content() # Attempt to get content even on timeout?
        raise
    except Exception as e:
        log.error(f"An error occurred during Playwright execution: {e}", exc_info=True)
        # Ensure browser is closed if error occurs after launch
        if 'browser' in locals() and browser.is_connected():
             browser.close()
        raise # Re-raise the exception

    if not html_content:
        raise ValueError("Playwright failed to retrieve HTML content.")

    return html_content


def scrape_conclavoscope(url: str, output_path: Path):
    """Scrapes cardinal data from Conclavoscope using Playwright and saves it to JSON.

    Args:
        url: The URL to scrape.
        output_path: Path to save the extracted JSON data.
    """
    try:
        # Fetch HTML using Playwright
        html = fetch_html_with_playwright(url, wait_selector='table#cardinalsTable')

        # Parse the fetched HTML
        soup = BeautifulSoup(html, 'lxml')

        # Extract data from the table
        cardinal_data = extract_cardinal_data_from_table(soup)

        if not cardinal_data:
             log.error("No cardinal data extracted from the table.")
             sys.exit(1)

        log.info(f"Successfully extracted data for {len(cardinal_data)} cardinals.")

        # --- Save data to JSON --- #
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(cardinal_data, f, indent=4, ensure_ascii=False)
        log.info(f"Cardinal data successfully saved to {output_path}")

    except (ValueError, PlaywrightTimeoutError) as e:
         log.error(f"Failed during Playwright fetch or table parsing: {e}", exc_info=True)
         sys.exit(1)
    except Exception as e:
        log.error(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape cardinal data from Conclavoscope.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "conclavoscope_parsed.json",
        help="Path to save the parsed JSON data."
    )
    args = parser.parse_args()

    # Setup logging if run as script
    if isinstance(setup_logging, type(logging)):
         setup_logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    scrape_conclavoscope(CONCLAVOSCOPE_URL, args.output)
