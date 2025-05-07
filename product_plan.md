## Windsurf-Based Conclave Simulation Product Plan

### Background

* **Papal Vacancy Triggered:** On May 4, 2025, Pope Francis passed away, initiating the need for a conclave of 135 cardinal electors to choose his successor.
* **Project Goal:** Build a robust, transparent Monte Carlo simulation to forecast likely conclave outcomes and timelines under real-world rules and scenarios.

### 1. Vision & Objectives

* **Vision:** Leverage an AI-native IDE to rapidly prototype, iterate, and maintain a conclave simulation engine that produces data-driven forecasts for an imminent papal election.
* **Objectives:**

  * Model the full multi-round, two-thirds-majority voting process among 135 electors.
  * Enable scenario analyses (e.g., shifting blocs, elector absences, rule variations).
  * Deliver clear visual and statistical outputs to inform analysts and stakeholders.
  * Maintain code quality and collaboration via Windsurf’s AI features and CI/CD.

### 2. Target Users & Use Cases

* **Primary Users:** Data scientists, political analysts, religious scholars.
* **Use Cases:**

  * Providing near-real-time probability updates during the conclave.
  * Educating audiences on cardinal voting dynamics.
  * Extending the platform to similar multi-round elections (e.g., party leadership contests).

### 3. Key Features & Prioritization

| Priority | Feature                                  | Description                                                                      |
| -------- | ---------------------------------------- | -------------------------------------------------------------------------------- |
| P0       | Base Simulation Engine                   | Core Monte Carlo loop, ballot rules, transition matrix, result aggregation.      |
| P1       | Elector Roster Ingestion & Profiles      | CSV/JSON import, schema validation, interactive editing via Windsurf.            |
| P1       | Preference & Transition Parameter Module | Configurable β-weights, distance metrics, transition-matrix generators.          |
| P2       | Diagnostics & Visualization              | Kernel-density, histogram, violin plots; interactive dashboards in notebooks.    |
| P2       | Scenario Runner                          | CLI/API to run "what-if" scenarios: remove electors, alter rules, adjust priors. |
| P3       | CI/CD Integration                        | Automated unit tests (pytest), linting (flake8), and Git hooks.                  |
| P3       | Documentation & Examples                 | Markdown docs, Jupyter notebooks, and Windsurf Cascade flows.                    |

### 4. Roadmap & Timeline

| Phase       | Duration | Deliverables                          |
| ----------- | -------- | ------------------------------------- |
| **Phase 0** | 1 week   | Repo + Windsurf project setup; P0 MVP |
| **Phase 1** | 2 weeks  | Dataset Creation & Initial Modeling  |
| **Phase 2** | 2 weeks  | Diagnostics; basic visualizations     |
| **Phase 3** | 2 weeks  | Scenario runner; extended tests       |
| **Phase 4** | 1 week   | Full docs; example notebooks          |

### 5. Metrics & Success Criteria

* **Code Velocity:** Time to create/refactor core modules (<30 min per feature).
* **Accuracy:** Simulated 2013 conclave win probability within ±5 % of historical estimates.
* **Adoption:** ≥5 external contributors in first month.
* **Performance:** 20 000 simulations in <1 min on standard laptop CPU.

### 6. Risks & Mitigations

* **Dependency Drift:** Pin versions; use conda-lock.
* **Compute Limitations:** Leverage local GPU or offload heavy batches to Colab.
* **Model Calibration:** Validate against 2005/2013 data; embed POC notebooks.

---

## Dataset Creation Plan

**Summary:** You’ll (1) identify and catalogue authoritative cardinal rosters (GCatholic, Catholic-Hierarchy, Vatican press), (2) define a clear schema for each elector’s attributes, (3) write Python scrapers to fetch and parse HTML/CSV, (4) clean and normalize names/dates/locations, (5) validate record counts against official reports, (6) export raw and cleaned CSVs, and (7) document the full workflow in code and notebooks.

---

## 1. Identify Authoritative Sources

1. **GCatholic.org** – Maintains a “Living Cardinals” table with ages and birth dates (137 entries, including non-electors) ([GCatholic][1]).
2. **Catholic-Hierarchy.org** – Structured view of “Cardinal Electors” vs. non-voting cardinals ([Catholic Hierarchy][2]).
3. **Wikipedia – List of current cardinals** – Community-maintained roster (use cautiously, cross-check) ([Wikipedia][5]).
4. **Conclavescore**

---

## 2. Define Data Schema

Decide on these core fields for each elector:

* `elector_id` (int, unique identifier)
* `name` (string)
* `country` (string)
* `appointing_pope` (string: "John Paul II", "Benedict XVI")
* `role_type` (string: e.g., "Diocesan", "Curial", "Patriarch", "Emeritus")
* `date_of_birth` (string, YYYY-MM-DD) - *To be added*
* `date_elevated` (string, YYYY-MM-DD) - *To be added*
* `age_at_conclave` (int) - *Derived from date_of_birth and conclave date*
* `is_eligible` (boolean) - *Derived from age_at_conclave* (Age < 80 at *sede vacante* start)
* `ideology_score` (float, -1.0 to 1.0) - *To be added/derived*
* `ideology_basis` (string, optional notes on score derivation) - *To be added*

---

## 3. Acquisition & Extraction

1. **GCatholic Scraper**:

   ```python
   resp = requests.get("https://gcatholic.org/hierarchy/cardinals-alive-age.htm")
   soup = BeautifulSoup(resp.text, "html.parser")
   table = soup.find("table", {"class": "table-striped"})
   ```

   ([GCatholic][1])
2. **Catholic-Hierarchy Scraper**:

   ```python
   resp = requests.get("https://www.catholic-hierarchy.org/country/0c.html")
   ```

   ([Catholic Hierarchy][2])
3. **Vatican Press CSV/HTML**: Download the latest bulletin listing cardinals (e.g. December 2024) ([Vatican Press][3]).
4. **Acta Apostolicae Sedis**: If needed, fetch PDF/HTML volumes for creation dates ([Vatican][4]).

---

## 3.1 Transition Model Specification

The `TransitionModel` in `src/model.py` calculates the probability of each elector voting for each potential candidate in a given round. The probability is determined by a combination of ideological similarity and vote stickiness from the previous round.

**Inputs:**

* `elector_data` (pd.DataFrame): Indexed by `elector_id`, must contain the calculated `ideology_score` column.
* `current_votes` (dict, optional): Maps `elector_id` (voter) to `elector_id` (candidate voted for in the previous round). `None` for the first round.

**Parameters:**

* `beta_weight` (float): Controls the sensitivity to ideological distance. Higher values mean electors strongly prefer ideologically similar candidates.
* `stickiness_factor` (float, 0 to 1): Controls the tendency to repeat the previous vote. 0 means no stickiness, 1 means maximal stickiness (deterministic if combined with other factors).

**Calculation Steps:**

1. **Ideological Distance:** Calculate the pairwise absolute ideological distance matrix `D`, where `D[i, j] = |ideology_score[i] - ideology_score[j]|`.
2. **Base Attraction:** Calculate a base attraction matrix `A` based on distance: `A[i, j] = exp(-beta_weight * D[i, j])`. Note that `A[i, i]` (attraction to self) will be `exp(0) = 1`.
3. **Apply Stickiness (if `current_votes` is provided and `stickiness_factor > 0`):
    * Initialize `FinalScores = (1 - stickiness_factor) * A`.
    * For each voter `i` who previously voted for candidate `k`: Boost the score for that specific candidate: `FinalScores[i, k] += stickiness_factor`. *(Note: This simple additive boost assumes the baseline attraction score `A` is normalized or scaled appropriately before combining. The exact interaction might need calibration. An alternative approach is to combine probabilities directly: `P_final = stickiness * P_previous + (1-stickiness) * P_attraction`)*
4. **No Stickiness (if `current_votes` is `None` or `stickiness_factor == 0`):
    * `FinalScores = A`.
5. **Normalization:** Normalize the `FinalScores` matrix row-wise to get probabilities. For each row `i` (voter): `Probability[i, j] = FinalScores[i, j] / sum(FinalScores[i, :])`. Ensure the sum of probabilities for each voter equals 1.

**Handling Missing Scores:** Electors missing an `ideology_score` (due to missing `cs_alignment_score`) should be handled. Options include: (a) excluding them from the transition calculation, (b) imputing their score (e.g., with the mean/median), or (c) assigning a neutral transition probability. Method (b) or (c) is generally preferred to keep the elector count stable.

---

## 4. Parsing & Cleaning

* **Name normalization**: strip accents (e.g. `unidecode`), consistent casing.
* **Date parsing**: `datetime.strptime` → ISO 8601 strings.
* **Continent mapping**: use an ISO country-to-continent lookup (e.g. `pycountry` + custom map).
* **Elector filter**: calculate age on May 5, 2025 and set `elector_status = (age < 80)`.
* **Deduplication**: merge entries by exact `name` and `birth_date`.

---

## 5. Validation & QA

1. **Record count check**: ensure cleaned dataset has exactly 135 electors ([Wikipedia][6]).
2. **Cross-source sampling**: randomly verify 10 entries against Vatican press bulletins or GCatholic pages.
3. **Automated tests**: add `tests/test_dataset.py` to assert schema, record counts, no missing fields.

---

## 6. Output & Storage

* **Raw dump**: `data/raw/cardinals_raw.csv` (all fetched rows).
* **Clean profile**: `data/processed/cardinals_profiles.csv` (final schema, validated).
* **Versioning**: commit both CSVs to Git LFS or Data folder with date suffix.

---

## 7. Documentation & Examples

* **`docs/dataset.md`**: Outline sources, scraping commands, data schema.
* **`notebooks/dataset_creation.ipynb`**: Demo notebook showing fetch → clean → validate steps.

## 8. Ideology Score Calculation

The `ideology_score` for each elector is a primary driver of voting preference in the simulation. It aims to represent their position on a conservative-to-progressive spectrum.

### 8.1 Primary Data Source

* **Conclavoscope Alignment Score (`cs_alignment_score`):** The main input for the `ideology_score` will be the `cs_alignment_score` obtained from the merged dataset (`data/merged_electors.csv`). This score reflects an assessment of the cardinal's alignment based on Conclavoscope's methodology.

### 8.2 Calculation

1. **Handle Missing Values:** Address missing `cs_alignment_score` values in the input data. Common strategies include:
    * Imputation: Replace missing scores with the mean or median `cs_alignment_score` of all electors who have a score.
    * Default Value: Assign a neutral score (e.g., 0 if the final scale is -1 to +1).
2. **Normalization:** Normalize the (potentially imputed) `cs_alignment_score` to a consistent range, typically **[-1.0, +1.0]**, where -1.0 represents the most conservative and +1.0 represents the most progressive end of the scale relative to the dataset.
    * Min-Max Scaling is a suitable method:
        `norm_score = -1 + 2 * (score - min_score) / (max_score - min_score)`
3. **Assignment:** The resulting normalized score is assigned to the `ideology_score` column in the `elector_data` DataFrame used by the simulation models.

(Previous complex methods involving LLMs, external website scraping, and multi-factor weighting (Sections 8.2-8.3 in earlier versions) are deprecated in favor of this more direct approach based on available Conclavoscope data.)

### 8.3 Validation & Calibration (Simplified)

#### 8.3.1 Sanity Checks & Outlier Review

* Assert –1 ≤ `ideology_score` ≤ +1 after calculation.
* Review the distribution of the final `ideology_score` (e.g., histogram) to ensure it appears reasonable.
* Examine electors with extreme scores (close to -1 or +1) to ensure they align with qualitative expectations, if possible.

#### 8.3.2 Calibration Considerations

* While direct external benchmarking might be difficult without the previously planned data sources, the `beta_weight` parameter in the `TransitionModel` can be calibrated during simulation runs (e.g., by comparing simulation outcomes against historical conclaves or expert predictions) to tune the influence of the `ideology_score`.
* If simulation results consistently deviate from expectations, revisiting the normalization method or the handling of missing values for the `ideology_score` may be necessary.

#### 8.3.3 Integration into Simulation Model

* The final `ideology_score` is a key input for the `TransitionModel` in `src/model.py`, influencing voting probabilities between cardinals based on ideological distance.

## 9. Simulation Engine (`src/simulate.py`)

```python
# Simulation Engine
```

---

## Phase 1: Dataset Creation & Initial Modeling (Target: Week 1-2)

**Goal:** Establish a robust dataset of **current** cardinal electors and implement foundational preference and transition models.

**Key Tasks:**

1. **Elector Roster Ingestion (`src/ingest.py`):
    * Define a stable CSV schema for elector attributes (ID, Name, Region, DOB, Ordination Date, Ideology Score).
    * Implement `load_elector_data` function.
    * Implement web scraping (`requests`, `beautifulsoup4`) for:
        * GCatholic (Living cardinals): `scrape_gcatholic_roster`
        * Catholic-Hierarchy (Elector cardinals): `scrape_catholic_hierarchy_roster`
    * Focus: Create a consolidated dataset (`data/electors_current_consolidated.csv`) by merging scraped data with manual inputs (like ideology scores). Prioritize completeness and accuracy for *current* electors.
    * Defer historical elector lists (e.g., for 2013, 2005 conclaves) to later phase.

2. **Preference Model (`src/model.py`):
    * Implement `PreferenceModel` class.

**Backtesting:** Once the simulation for current electors is validated, create datasets for historical conclaves (e.g., 2005, 2013) and run the simulation to compare outcomes against reality.

**Geopolitical Factors:** Incorporate more nuanced regional dynamics beyond simple affinity.

**Papal Influence:** Model the potential impact of the previous Pope's appointments.

[1]: https://www.gcatholic.org/
[2]: https://www.catholic-hierarchy.org/
[3]: https://press.vatican.va/
[4]: https://www.vatican.va/archive/aas/index.htm
[5]: https://en.wikipedia.org/wiki/List_of_current_cardinals
[6]: https://en.wikipedia.org/wiki/Cardinals_created_by_Francis
