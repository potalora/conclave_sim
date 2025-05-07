# Temporary script to parse GCatholic Markdown scrape from a file
import pandas as pd
import re
import sys
from pathlib import Path
from datetime import datetime

RAW_MD_PATH = Path(__file__).parent / 'data' / 'gcatholic_scrape_raw.md'
OUTPUT_CSV_PATH = Path(__file__).parent / 'data' / 'gcatholic_cardinals_raw.csv'

def parse_markdown_file(md_file_path: Path) -> pd.DataFrame:
    """Parses the GCatholic raw markdown file to extract elector data.

    Args:
        md_file_path: Path to the raw markdown file.

    Returns:
        A pandas DataFrame containing the parsed elector data.
    """
    electors = []
    # AI: Regex attempt 12: Reverted to Attempt 10 structure, modified sub-pattern for '.Sp' case.
    # Groups: 1=Name, 2=Age(paren), 3=Description, 4=Age(col), 5=DOB
    pattern = re.compile(r".*?Cardinal\s*([\w\s.'\-áéíóúüñçÁÉÍÓÚÜÑÇ]+(?:\s*,?\s*[A-Z](?:\.[A-Z]+|\.[A-Z][a-z]+)*\.?(?:\s+\w+\.?)?)?)\s*\((\d+)\)([^|]*?)\s*\|\s*(\d+)\s*\|\s*(\d{4}\.\d{2}\.\d{2})\s*")

    try:
        with open(md_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if len(lines) < 7:
            print(f"Warning: File '{md_file_path}' has fewer than 7 lines. Cannot find elector data line.")
            return pd.DataFrame(electors, columns=['Name', 'Age', 'Date of Birth'])

        # AI: Read the 6th line (index 5) which might contain all elector data
        data_line = lines[5].strip()

        # AI: Find start and end markers for elector data within the line
        start_marker = "Cardinal Electors"
        end_marker = "Cardinal Non-Electors"
        start_index = data_line.find(start_marker)
        end_index = data_line.find(end_marker)

        if start_index == -1:
            print(f"Error: Could not find start marker '{start_marker}' on line 6.")
            return pd.DataFrame(electors, columns=['Name', 'Age', 'Date of Birth'])

        # AI: Adjust start index to be after the marker and any initial header like '(135) | AgeAscending| Date ofBirth | |'
        header_end_marker = "| |"
        start_data_index = data_line.find(header_end_marker, start_index)
        if start_data_index != -1:
            start_data_index += len(header_end_marker) # Start after the marker
        else:
            # Fallback if '| |' isn't found right after header - might indicate format change
            print(f"Warning: Could not find expected '{header_end_marker}' after '{start_marker}'. Attempting to proceed, but parsing might be inaccurate.")
            # Try to find the end of the first parenthesis part as a rough start
            temp_start = data_line.find(')', start_index)
            if temp_start != -1:
                start_data_index = temp_start + 1 # Guess start after parenthesis
            else:
                start_data_index = start_index + len(start_marker) # Last resort


        # AI: Extract the relevant substring containing only elector data
        if end_index != -1:
            elector_data_substring = data_line[start_data_index:end_index].strip()
        else:
            # If end marker not found, assume electors run to the end of the line
            print(f"Warning: Could not find end marker '{end_marker}'. Parsing data until the end of the line.")
            elector_data_substring = data_line[start_data_index:].strip()

        # AI: Split the substring into segments based on the '| |' delimiter
        segments = elector_data_substring.split('| |')

        for segment in segments:
            segment = segment.strip()
            if not segment: # Skip empty segments resulting from split
                continue

            match = pattern.search(segment)
            if match:
                # Clean name: remove potential double spaces sometimes left by suffixes
                name = re.sub(r'\s+', ' ', match.group(1).strip()).strip()
                age_in_paren = int(match.group(2))
                description = match.group(3)
                age = int(match.group(4)) # Use the age from the column (group 4)
                dob_str = match.group(5)

                try:
                    dob = datetime.strptime(dob_str, '%Y.%m.%d').date()
                    electors.append({'Name': name, 'Age': age, 'Date of Birth': dob})
                except ValueError:
                    print(f"Warning: Could not parse date '{dob_str}' for {name}. Skipping entry.")
            else:
                # Debugging for non-matching segments
                 if segment:
                     print(f"Debug: No match found in segment: '{segment[:150]}...'" )

    except FileNotFoundError:
        print(f"Error: File not found at {md_file_path}")
        return pd.DataFrame(columns=['Name', 'Age', 'Date of Birth'])
    except Exception as e:
        print(f"An unexpected error occurred during parsing: {e}")
        # Optionally re-raise or log traceback for more detail
        # import traceback
        # print(traceback.format_exc())
        return pd.DataFrame(columns=['Name', 'Age', 'Date of Birth'])

    if not electors:
        print("Warning: No elector data could be parsed from the located substring.")

    return pd.DataFrame(electors, columns=['Name', 'Age', 'Date of Birth'])


def main():
    print(f"Attempting to read raw scrape data from: {RAW_MD_PATH}")
    try:
        with open(RAW_MD_PATH, 'r', encoding='utf-8') as f:
            raw_markdown = f.read()
        print(f"Successfully read {len(raw_markdown)} characters from raw scrape file.")
    except FileNotFoundError:
        print(f"Error: Raw scrape file not found at {RAW_MD_PATH}", file=sys.stderr)
        print("Please ensure you have copied the full scrape output into this file.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading raw scrape file: {e}", file=sys.stderr)
        sys.exit(1)

    df_electors = parse_markdown_file(RAW_MD_PATH)

    if not df_electors.empty:
        print(f"Successfully parsed {len(df_electors)} electors.")
        # Ensure the 'data' directory exists
        OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        # Save to CSV
        df_electors.to_csv(OUTPUT_CSV_PATH, index=False)
        print(f"Saved parsed data to: {OUTPUT_CSV_PATH}")
    else:
        print("Parsing failed or resulted in empty data. CSV file not created.")

if __name__ == "__main__":
    main()
