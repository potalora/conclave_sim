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
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    table = soup.find('table', id='cardinalsTable')

    if not table:
        log.error("Could not find the cardinal table.")
        raise ValueError("Could not find the cardinal table.")

    tbody = table.find('tbody')
    if not tbody:
        log.error("Could not find tbody within the table.")
        raise ValueError("Could not find tbody within the cardinal table.")

    rows = tbody.find_all('tr')
    log.info(f"Found {len(rows)} rows in the table body.")

    header_row = table.find('thead').find('tr')
    headers = [th.get_text(strip=True).lower() for th in header_row.find_all('th')]
    log.info(f"Detected headers: {headers}")

    try:
        name_col_idx = headers.index('name')
        score_col_idx = headers.index('total position')
        papabile_col_idx = headers.index('papabile score') if 'papabile score' in headers else -1
        alignment_col_idx = headers.index('alignment score') if 'alignment score' in headers else -1

    except ValueError as e:
        log.error(f"Missing required column in table headers: {e}. Headers found: {headers}")
        raise ValueError("Missing required column in table headers.")

    for i, row in enumerate(rows):
        cells = row.find_all('td')
        if len(cells) <= max(name_col_idx, score_col_idx): 
            log.warning(f"Row {i+1} has too few cells ({len(cells)}), skipping.")
            continue

        name_cell_html = cells[name_col_idx]
        cell_texts = [s.strip() for s in name_cell_html.stripped_strings if s.strip()]
        
        cardinal_name = "Unknown" # Default
        actual_cs_country_raw = None # Default
        is_papabile_from_name_cell_raw = False # Default

        if not cell_texts:
            log.warning(f"Row {i+1}, Name cell at index {name_col_idx} is empty or contains only whitespace. Cardinal name will be 'Unknown'.")
        else:
            cardinal_name = cell_texts[0] # First part is always the name

            if len(cell_texts) > 1:
                # Second part (cell_texts[1]) could be country or papabile status
                second_part_text = cell_texts[1]
                processed_second_part = second_part_text # Already stripped from list comprehension

                # Check for parentheses, e.g., "(Papabile)"
                if processed_second_part.startswith("(") and processed_second_part.endswith(")"):
                    processed_second_part = processed_second_part[1:-1]

                if processed_second_part.lower() == "papabile":
                    is_papabile_from_name_cell_raw = True
                    # If second part was papabile, country might be in third part (cell_texts[2])
                    if len(cell_texts) > 2:
                        actual_cs_country_raw = cell_texts[2] # Already stripped
                        # Optional: Further process country if it can also have parentheses
                        # if actual_cs_country_raw.startswith("(") and actual_cs_country_raw.endswith(")"):
                        #     actual_cs_country_raw = actual_cs_country_raw[1:-1]
                else:
                    # Second part was not "papabile", so it's the country.
                    # Use the processed_second_part which has parentheses removed (if any).
                    actual_cs_country_raw = processed_second_part
            # If len(cell_texts) == 1, only name is extracted. Country and papabile status remain default.

        log.debug(f"Parsed Name: '{cardinal_name}', Raw parts from name cell: {cell_texts}, Deduced Country: '{actual_cs_country_raw}', Deduced Papabile from name cell: {is_papabile_from_name_cell_raw}")

        total_score_str = cells[score_col_idx].get_text(strip=True)
        score_val = None
        score_match = re.match(r"(\d+)(?:/\d+)?", total_score_str)
        if score_match:
            score_val = int(score_match.group(1))
        else:
            log.warning(f"Could not parse score from: {total_score_str} for {cardinal_name}")

        papabile_score = cells[papabile_col_idx].get_text(strip=True) if papabile_col_idx != -1 and len(cells) > papabile_col_idx else None
        alignment_score = cells[alignment_col_idx].get_text(strip=True) if alignment_col_idx != -1 and len(cells) > alignment_col_idx else None

        cardinal_data.append({
            'id': None, 
            'name': cardinal_name,
            'cs_country_raw': actual_cs_country_raw,
            'is_papabile_from_name_cell_raw': is_papabile_from_name_cell_raw,
            'total_score': score_val,
            'papabile_score': papabile_score, 
            'alignment_score': alignment_score
        })

    return cardinal_data


def _try_hide_modal(page, modal_selector="#finderInviteModal", context_msg=""):
    """Attempts to hide or remove the specified modal and its backdrops using JavaScript."""
    modal_is_present_initially = False
    # Check if modal is visible
    # Using is_visible for elements that might exist but be display:none initially
    # and query_selector for style checks if it does exist.
    modal_element = page.query_selector(modal_selector)
    if modal_element and (page.is_visible(f"{modal_selector}.show") or \
                         page.is_visible(f"{modal_selector}.in") or \
                         "display: block" in (modal_element.get_attribute("style") or "")):
        
        log.warning(f"Modal '{modal_selector}' detected {context_msg}. Attempting to hide/remove it with JS.")
        modal_is_present_initially = True
        try:
            # Attempt 1: Set display to none
            js_hide_script = f"""
                (() => {{ // Wrap in IIFE
                    let modal_hidden_by_style = false;
                    const modal = document.querySelector('{modal_selector}');
                    if (modal) {{
                        modal.style.display = 'none'; 
                        modal.classList.remove('show', 'in'); 
                        modal_hidden_by_style = (modal.style.display === 'none');
                    }}
                    const backdrops = document.querySelectorAll('.modal-backdrop');
                    backdrops.forEach(backdrop => {{ backdrop.style.display = 'none'; }});
                    return modal_hidden_by_style;
                }})(); // End IIFE
            """
            hidden_by_style_attempt = page.evaluate(js_hide_script)
            page.wait_for_timeout(300) # Brief pause
            
            # Check again. If still visible and an attempt was made to hide by style, try removing it.
            modal_still_visible = False
            modal_element_after_hide = page.query_selector(modal_selector) # Re-query, it might have been removed by other scripts
            if modal_element_after_hide and (page.is_visible(f"{modal_selector}.show") or \
                                      page.is_visible(f"{modal_selector}.in") or \
                                      "display: block" in (modal_element_after_hide.get_attribute("style") or "")):
                modal_still_visible = True

            if modal_is_present_initially and modal_still_visible:
                log.warning(f"Modal '{modal_selector}' {context_msg} still appears visible after 'display:none'. Attempting to remove element.")
                js_remove_script = f"""
                    const modal_el = document.querySelector('{modal_selector}');
                    if (modal_el) {{ modal_el.remove(); }}
                    document.querySelectorAll('.modal-backdrop').forEach(b => b.remove());
                """
                page.evaluate(js_remove_script)
                page.wait_for_timeout(300) # Brief pause

            # Final check if element is gone or truly hidden (not just display:none on a persistent element)
            final_modal_check = page.query_selector(modal_selector)
            if not final_modal_check or not final_modal_check.is_visible():
                 log.info(f"Modal '{modal_selector}' {context_msg} successfully hidden/removed via JS.")
            elif modal_is_present_initially: # If it was there and we tried, but it's still there and visible
                 log.warning(f"Modal '{modal_selector}' {context_msg} may STILL be visible/present after JS hide/remove attempts.")

        except Exception as e_js_modal:
            log.error(f"Error during JS modal hide/remove for '{modal_selector}' {context_msg}: {e_js_modal}")
    return modal_is_present_initially


def fetch_html_with_playwright(url: str, wait_selector: str = 'table#cardinalsTable', wait_timeout: int = 20000) -> List[Dict[str, Any]]:
    """Fetches data from all pages of the cardinal table using Playwright.

    Args:
        url: The URL to scrape.
        wait_selector: The CSS selector for the main table to ensure it's loaded.
        wait_timeout: Maximum time in milliseconds to wait for elements.

    Returns:
        A list of dictionaries, each representing a cardinal, from all pages.
    """
    all_cardinals_data = []
    log.info(f"Launching Playwright to fetch {url}")
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            log.info(f"Navigating to {url}...")
            page.goto(url, wait_until='networkidle', timeout=60000)
            log.info(f"Navigated to {url}. Waiting for page to settle and initial content to load...")
            page.wait_for_timeout(3000) # Extra wait for any dynamic content or scripts

            # AI: Use helper for initial modal check
            modal_detected_on_load = _try_hide_modal(page, context_msg="on page load")

            # --- Select number of entries to display --- #
            length_dropdown_selector = "select[name='cardinalsTable_length']"
            dropdown_interaction_skipped = False

            if modal_detected_on_load:
                log.info("Skipping 'Show entries' dropdown interaction due to modal detection ON PAGE LOAD.")
                dropdown_interaction_skipped = True
            elif not page.query_selector(length_dropdown_selector):
                log.warning(f"Entries dropdown ('{length_dropdown_selector}') not found. Proceeding with default page size.")
                dropdown_interaction_skipped = True
            else: # Modal not detected on load, and dropdown exists
                log.info(f"Found entries dropdown: {length_dropdown_selector}. Attempting to set entries per page.")
                
                _try_hide_modal(page, context_msg="before 'Show entries' dropdown interaction") # AI: Call before dropdown

                try:
                    page.select_option(length_dropdown_selector, value='100', timeout=10000) # Reduced timeout
                    log.info("Selected '100' entries per page.")
                except PlaywrightTimeoutError: 
                    log.warning("Failed to select '100' entries. Trying to hide modal again and then try '-1' (All).")
                    _try_hide_modal(page, context_msg="after failing '100' and before trying '-1'")
                    try:
                        page.select_option(length_dropdown_selector, value='-1', timeout=10000) # Reduced timeout
                        log.info("Selected '-1' (All) entries per page.")
                    except PlaywrightTimeoutError:
                        log.warning("Failed to select '-1' (All) entries. Modal might be interfering. Proceeding with default page size.")
                        dropdown_interaction_skipped = True 
                
                if not dropdown_interaction_skipped:
                    page.wait_for_timeout(3000) 
            
            if dropdown_interaction_skipped:
                 log.info("Proceeding with default page size for entries.")

            page_num = 1
            while True:
                # Wait for the table to be present before extracting data
                try:
                    page.wait_for_selector(wait_selector, timeout=wait_timeout, state='visible')
                except PlaywrightTimeoutError:
                    log.error(f"Page {page_num}: Table selector '{wait_selector}' not found or not visible after timeout. Ending pagination.")
                    break
                
                current_html = page.content()
                soup = BeautifulSoup(current_html, 'html.parser')
                data = extract_cardinal_data_from_table(soup)
                all_cardinals_data.extend(data)
                log.info(f"Page {page_num}: Extracted {len(data)} cardinals. Total: {len(all_cardinals_data)}")

                _try_hide_modal(page, context_msg=f"before checking/clicking 'Next' on page {page_num}")
                page.wait_for_timeout(500) # Allow page to settle after modal operations

                next_button_locator = page.locator("#cardinalsTable_paginate a:has-text('Next')")
                
                if next_button_locator.count() > 0:
                    is_disabled = "disabled" in (next_button_locator.first.get_attribute("class") or "").lower()
                    if not is_disabled:
                        log.info(f"Page {page_num}: Active 'Next' button found via locator. Clicking to go to page {page_num + 1}.")
                        try:
                            next_button_locator.first.click(timeout=10000)
                            log.info("Waiting for table content of the next page...")
                            # This wait_for_selector is for the *new* page's table.
                            # If it times out, we assume we've gone past the last actual page.
                            page.wait_for_selector(wait_selector, timeout=wait_timeout, state='visible') 
                            page.wait_for_timeout(2000) # Additional settle time for data after table appears
                            log.info(f"Content for page {page_num + 1} appears loaded.")
                        except PlaywrightTimeoutError:
                            log.info(f"Page {page_num}: Timed out waiting for table content after clicking 'Next'. This likely means it was the last page. Ending pagination.")
                            break 
                        except Exception as e_click_general:
                            log.error(f"Page {page_num}: An unexpected error occurred clicking 'Next' or waiting for new content: {e_click_general}. Ending pagination.")
                            break
                        page_num += 1
                    else:
                        log.info(f"Page {page_num}: 'Next' button found via locator but is disabled. End of pages.")
                        break
                else:
                    log.info(f"Page {page_num}: 'Next' button (using locator #cardinalsTable_paginate a:has-text('Next')) not found. End of pages.")
                    break
            
            browser.close()

    except PlaywrightTimeoutError:
        log.error(f"Playwright timed out during pagination or waiting for selector '{wait_selector}' at {url}", exc_info=True)
        if 'browser' in locals() and browser.is_connected():
            browser.close()
        # Depending on policy, might want to return partial data or raise
        if not all_cardinals_data:
             raise # Re-raise if no data was collected at all
        else:
             log.warning("Timeout occurred, returning partially collected data.")
    except Exception as e:
        log.error(f"An error occurred during Playwright execution: {e}", exc_info=True)
        if 'browser' in locals() and browser.is_connected():
            browser.close()
        if not all_cardinals_data:
             raise
        else:
             log.warning("Error occurred, returning partially collected data.")

    if not all_cardinals_data:
        # This check might be redundant if exceptions above re-raise when no data, 
        # but good for safety if those are changed.
        raise ValueError("Playwright failed to retrieve any cardinal data.")

    return all_cardinals_data


def scrape_conclavoscope(url: str, output_path: Path):
    """Scrapes cardinal data from Conclavoscope using Playwright and saves it to JSON.

    Args:
        url: The URL to scrape.
        output_path: Path to save the extracted JSON data.
    """
    try:
        # Fetch HTML using Playwright - now returns list of dicts from all pages
        all_cardinal_data = fetch_html_with_playwright(url, wait_selector='table#cardinalsTable')

        # Data is already parsed by fetch_html_with_playwright
        # soup = BeautifulSoup(html, 'lxml')
        # cardinal_data = extract_cardinal_data_from_table(soup)

        if not all_cardinal_data:
             log.error("No cardinal data extracted from the website.") # Updated message
             sys.exit(1)

        log.info(f"Successfully extracted data for {len(all_cardinal_data)} cardinals from all pages.")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_cardinal_data, f, indent=4, ensure_ascii=False)
        log.info(f"Cardinal data successfully saved to {output_path}")

    except ValueError as ve:
        log.error(f"A value error occurred during scraping: {ve}")
        sys.exit(1)
    except PlaywrightTimeoutError as pte:
        log.error(f"Playwright timed out during the scraping process: {pte}")
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

    scrape_conclavoscope(CONCLAVOSCOPE_URL, args.output)
