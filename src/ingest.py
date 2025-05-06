import pandas as pd
from typing import Optional, Dict, List
import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin # AI: Added import for urljoin

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


# AI: Added function to scrape GCatholic
def scrape_gcatholic_roster(url: str = "https://gcatholic.org/hierarchy/cardinals-alive-age.htm") -> Optional[List[Dict[str, str]]]:
    """Scrapes the 'Living Cardinals' table from GCatholic.org.

    Fetches the page content, parses the HTML, and attempts to find the main
    cardinals table.

    Args:
        url: The URL of the GCatholic living cardinals page.

    Returns:
        A list of dictionaries, where each dictionary represents a row
        (a cardinal) with column headers as keys, or None if scraping fails.

    Raises:
        requests.exceptions.RequestException: If the web request fails.
    """
    print(f"Attempting to scrape GCatholic roster from: {url}")
    try:
        response = requests.get(url, timeout=30) # Add timeout
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL {url}: {e}")
        raise # Re-raise the exception after logging

    soup = BeautifulSoup(response.content, 'lxml') # Use lxml parser

    # Find the table (based on product plan: class='table-striped')
    # Note: Website inspection might be needed if this class name is inaccurate
    # or if there are multiple tables with this class.
    # AI: Updated selector based on raw HTML inspection
    table = soup.find("table", class_="tb")

    if not table:
        # AI: Updated error message to reflect the new selector
        print(f"Error: Could not find the expected table (class='tb') on {url}")
        return None

    print("Successfully found the cardinals table.")

    # AI: Implement table data extraction logic
    cardinals_data: List[Dict[str, str]] = []
    headers = []

    # Extract headers (usually in <thead> or the first <tr>)
    header_row = table.find('thead')
    if not header_row:
        header_row = table.find('tr') # Fallback to first row if no thead

    if header_row:
        # Extract text from th or td elements within the header row
        headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]
        print(f"Found headers: {headers}")
    else:
        print("Warning: Could not find table headers.")
        # Optionally define default headers if structure is known but lacks th/thead

    # Extract data rows (usually in <tbody> or subsequent <tr>)
    body = table.find('tbody')
    if not body:
        data_rows = table.find_all('tr')[1:] # Skip header row if no tbody
    else:
        data_rows = body.find_all('tr')

    print(f"Found {len(data_rows)} data rows.")

    for row in data_rows:
        cells = row.find_all('td')
        if headers:
            # Create dict only if headers were found and cell count matches
            if len(cells) == len(headers):
                row_data = {headers[i]: cells[i].get_text(strip=True) for i in range(len(cells))}
                cardinals_data.append(row_data)
            else:
                print(f"Warning: Row skipped due to mismatch between cell count ({len(cells)}) and header count ({len(headers)}). Row content: {[c.get_text(strip=True) for c in cells]}")
        else:
            # If no headers, store as list of strings (less ideal)
            row_data_list = [cell.get_text(strip=True) for cell in cells]
            # Representing as dict with generic keys if no headers
            cardinals_data.append({f"col_{i}": data for i, data in enumerate(row_data_list)})

    # AI: Removed placeholder print statement
    # print("Table data extraction not yet implemented.")

    if not cardinals_data:
        print("Warning: Data extraction resulted in an empty list.")

    return cardinals_data


# AI: Added function to scrape Catholic Hierarchy
# AI: Updated default URL again to the scardc3 page as requested by user
def scrape_catholic_hierarchy_roster(url: str = "https://www.catholic-hierarchy.org/bishop/scardc3.html") -> Optional[List[Dict[str, str]]]:
    """Scrapes the list from Catholic-Hierarchy.org.

    Args:
        url: The URL of the Catholic Hierarchy page (default updated).
    
    Returns:
        A list of dictionaries representing cardinals, or None if scraping fails.

    Raises:
        requests.exceptions.RequestException: If the web request fails.
    """ # AI: Ensure docstring is properly closed
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

    return cardinals_data


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

    # AI: Add test for scraper
    print("\n--- Testing scrape_gcatholic_roster --- ")
    try:
        gcatholic_data = scrape_gcatholic_roster()
        if gcatholic_data is not None:
            print(f"Successfully scraped {len(gcatholic_data)} records.")
            if gcatholic_data:
                print("First 2 records:")
                print(gcatholic_data[:2]) # Print first 2 records
        else:
            print("Scraping failed or table not found.")
    except requests.exceptions.RequestException as e:
        print(f"Scraping failed due to network/HTTP error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during scraping: {e}")

    # AI: Add test for Catholic Hierarchy scraper
    print("\n--- Testing scrape_catholic_hierarchy_roster --- ")
    try:
        ch_data = scrape_catholic_hierarchy_roster()
        if ch_data is not None:
            print(f"Successfully scraped {len(ch_data)} records.")
            if ch_data:
                print("First 2 records (potential):")
                print(ch_data[:2])
        else:
            # AI: Updated message
            print("Scraping failed or table not found/identified.")
    except requests.exceptions.RequestException as e:
        print(f"Scraping failed due to network/HTTP error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during scraping: {e}")

    print("--- Test complete ---")
