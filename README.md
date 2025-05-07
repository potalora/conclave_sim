# Windsurf Conclave Simulation

## Background

On May 4, 2025, Pope Francis passed away, initiating the need for a conclave of 135 cardinal electors to choose his successor.

## Project Goal

Build a robust, transparent Monte Carlo simulation using Windsurf to forecast likely conclave outcomes and timelines under real-world rules and scenarios.

## Vision & Objectives

* **Vision:** Leverage an AI-native IDE to rapidly prototype, iterate, and maintain a conclave simulation engine that produces data-driven forecasts for an imminent papal election.
* **Objectives:**
  * Model the full multi-round, two-thirds-majority voting process among 135 electors.
  * Enable scenario analyses (e.g., shifting blocs, elector absences, rule variations).
  * Deliver clear visual and statistical outputs to inform analysts and stakeholders.
  * Maintain code quality and collaboration via Windsurf’s AI features and CI/CD.

## Repository Structure

(Refer to `.windsurfrules` for details)

## Data Ingestion Pipeline

The simulation relies on accurate data for the 135 cardinal electors. This data is compiled and processed using scripts in the `src/` directory, primarily `src/ingest.py` and `src/match_names.py`.

The pipeline performs the following steps:

1. **Scraping:** Retrieves lists of living cardinals/electors from [GCatholic.org](https://gcatholic.org/hierarchy/cardinals-alive-age.htm) and [Catholic-Hierarchy.org](https://www.catholic-hierarchy.org/bishop/scardc3.html). Raw data is saved to `data/scraped_*.csv`.
2. **Name Matching (LLM):** If not already present, uses the Google Gemini API (`gemini-2.5-flash-preview-04-17`) via `src/match_names.py` to identify corresponding records between the two scraped datasets based on name similarity. Matches are saved to `data/llm_matched_pairs.json`. *Note: Requires a `GOOGLE_API_KEY` environment variable.*
3. **Merging:** Combines the two datasets based on the LLM matches, keeping only electors eligible for the conclave (age < 80).
4. **Standardization:** Cleans and standardizes cardinal names.
5. **Output:** Saves the final merged and processed elector list to `data/merged_electors.csv`.

**Running the Pipeline:**

The entire pipeline can be executed by running the `ingest` module from the project root:

```bash
python -m src.ingest
```

The script is designed to be idempotent. If the raw scraped files (`scraped_*.csv`) or the LLM matches file (`llm_matched_pairs.json`) already exist, the corresponding steps (scraping, LLM matching) will be skipped to save time and API calls. The IDs (`gc_id`, `ch_id`) are added and saved back into the raw CSVs during the first run of `match_names.py` or the first run of `ingest.py` that triggers matching.

---

## Getting Started

(To be detailed in `docs/onboarding.md`)
