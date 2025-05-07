# Product Plan

## Background

* **Papal Vacancy Triggered:** In April 2025, Pope Francis passed away, initiating the need for a conclave of 135 cardinal electors to choose his successor.
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
| P1       | Papabile Weighting                       | Apply a configurable weight to papabile cardinals in the selection model.        |
| P2       | Diagnostics & Visualization              | Kernel-density, histogram, violin plots; interactive dashboards in notebooks.    |
| P2       | Scenario Runner                          | CLI/API to run "what-if" scenarios: remove electors, alter rules, adjust priors. |
| P2       | Regional Affinity                        | Model voting preferences based on shared geographic regions.                     |
| P3       | CI/CD Integration                        | Automated unit tests (pytest), linting (flake8), and Git hooks.                  |
| P3       | Documentation & Examples                 | Markdown docs, Jupyter notebooks, and Windsurf Cascade flows.                    |

### 3.2 Papabile Weighting

* **Rationale**: Increase the simulation's realism by giving a slight advantage to cardinals explicitly identified as "papabile" (pope-able) by external sources like Conclavoscope. This acknowledges that some candidates enter the conclave with higher perceived likelihoods of election.
* **Data Source**: `merged_electors.csv`, specifically the `is_papabile` boolean column derived from Conclavoscope's "Name" cell and "Papabile Score" column during the `src/ingest.py` process.
* **Implementation Steps**:
  * Modify `src/model.py`:
    * In the `PreferenceModel` or the part of `TransitionModel` that calculates initial candidate attractiveness:
      * If an elector has `is_papabile == True` in `elector_data`, apply a configurable weight (e.g., 1.5x, 2.0x, or a fixed addition) to their base selection probability or attraction score *before* normalization.
      * This weight should be a configurable parameter in the model or simulation setup (e.g., `papabile_weight_factor`).
  * Update `src/simulate.py` if necessary to pass this new configuration to the model.
  * Document this weight as a configurable parameter in `README.md` and any relevant notebooks.
* **Acceptance Criteria**:
  * The `is_papabile` column is correctly populated and utilized by `src/model.py`.
  * Simulations using the new model show a statistically significant (though potentially modest) increase in the selection likelihood of designated papabile cardinals, ceteris paribus, when the weight is active.
  * The previous issue of "Papabile" being listed as a country remains resolved, and the new `is_papabile` flag works correctly.
* **Considerations**:
  * The exact weighting factor needs to be configurable and potentially tuned through multiple simulation runs to achieve a realistic impact.
  * Ensure documentation (`README.md`, `docs/onboarding.md`) is updated to reflect the new model behavior and configuration options.

### 3.3 Regional Affinity

* **Rationale**: To model the observed tendency for cardinals to sometimes favor candidates from their own geographic region or those with similar cultural backgrounds, reflecting shared pastoral concerns or solidarity.
* **Data Source**:
  * `merged_electors.csv`, specifically the `country` column.
  * A new mapping file (e.g., `data/country_to_region_map.json` or hardcoded in `src/ingest.py` or `src/model.py`) that groups countries into broader geopolitical/cultural regions (e.g., "Europe_Mediterranean", "Europe_Eastern", "Africa_SubSaharan", "Asia_East", "Latin_America_South", "North_America").
* **Implementation Steps**:
  * **Data Preparation (`src/ingest.py` or a new utility script)**:
    * Create or integrate a mapping from individual countries to predefined regions.
    * Add a new `region` column to `merged_electors.csv` based on this mapping.
  * **Model Enhancement (`src/model.py`)**:
    * In the `PreferenceModel` or `TransitionModel` (specifically where initial attractions or base probabilities are calculated):
      * If voter `i` and candidate `j` are from the same `region`, apply a configurable positive weight/bonus to the attraction score `A[i, j]`.
      * Optionally, define affinities between different (but related) regions (e.g., a smaller bonus if voter is from "Europe_Mediterranean" and candidate is from "Europe_Eastern").
      * This regional affinity weight should be a configurable parameter (e.g., `regional_affinity_factor`, `intra_regional_bonus`).
  * **Documentation**: Update `README.md` and relevant notebooks to explain the regional mapping and new model parameters.
* **Acceptance Criteria**:
  * The `region` column is correctly populated in `merged_electors.csv`.
  * When the regional affinity factor is active, simulations show a statistically discernible (though potentially subtle) increase in votes for candidates from the same region as the voters, compared to simulations without this factor.
  * The model remains robust and produces plausible outcomes with the new affinity logic.
* **Considerations**:
  * Defining meaningful regions can be subjective and complex. The mapping should be transparent and justifiable.
  * The strength of regional affinity can vary significantly; the weighting factor will need careful consideration and potential calibration.
  * Overly strong regional effects could lead to unrealistic bloc voting; the model should balance this with ideological and other factors.

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

## 7. Simulation Parameter Tuning Log (as of 2025-05-07)

**Objective:** Refine simulation parameters to achieve election of a Pope by a 2/3 majority in significantly fewer than 49 rounds, ideally targeting <10 rounds, while ensuring papabile candidates have a higher likelihood of winning. The `runoff_threshold_rounds` was consistently kept at 50.

**Key Strategies & Outcomes:**

Numerous parameter configurations were tested to influence convergence speed and winner profiles. Key parameters adjusted included:

* `papabile_bonus`: Modifies the attractiveness of designated papabile candidates.
* `bandwagon_strength`: Amplifies preference for candidates gaining votes.
* `stickiness_factor`: Influences elector loyalty to their previous vote.
* `beta_weight`: Controls the influence of base ideological scores versus random factors.
* `regional_bonus`: (Mostly kept at 0.0 to isolate other effects during this phase).

**Summary of Approaches Attempted:**

1. Initial Tweaks: Gradually increasing `papabile_bonus` and `bandwagon_strength` while decreasing `stickiness_factor`.
    * Outcome: Success rates improved, but average rounds remained high (~50-60). Elections consistently hit the runoff phase.
2. "Extreme Convergence": Pushing `papabile_bonus` (0.9), `bandwagon_strength` (0.9) very high, `stickiness_factor` (0.1) and `beta_weight` (0.2) very low.
    * Outcome: Still ~55 rounds for successful elections. Minimum rounds did not drop below 51.
3. "Ultra Convergence": More extreme values (`papabile_bonus=1.5`, `bandwagon_strength=2.0`, `stickiness_factor=0.01`, `beta_weight=0.05`).
    * Outcome: Similar to above, minimum rounds around 51. Success rate ~58%. Indicated possible vote churning among many strong papabile candidates.
4. Moderated "Ultra" with Higher Stickiness: (`papabile_bonus=1.2`, `bandwagon_strength=1.5`, `stickiness_factor=0.1`).
    * Outcome: Still ~55 rounds minimum.
5. "High-Stickiness, High-Momentum": (`papabile_bonus=1.5`, `bandwagon_strength=2.0`, `stickiness_factor=0.7`). Aimed to make electors "lock-in" on emerging leaders.
    * Outcome: Still ~57 rounds, min 51.
6. "Rapid Kingmaker": Lower initial `papabile_bonus` (0.3), very high `bandwagon_strength` (2.5) and `stickiness_factor` (0.8), slightly higher `beta_weight` (0.2). Aimed to create variance then rapidly amplify a leader.
    * Outcome: Still ~59 rounds, min 51. Elector '1' (Parolin) became a frequent winner but still too late.
7. "Overwhelming Favorite": Massive `papabile_bonus` (3.0), strong bandwagon (1.5), moderate stickiness (0.5). Aimed to make papabile group the only viable option from round 1.
    * Outcome: Success rate dropped significantly (10%), min rounds 51. Likely "flattened" the field too much among papabile candidates.

**Current Status & Next Steps:**
Despite aggressive parameter tuning, simulations consistently reach the 2/3 majority threshold only after ~50 rounds, usually triggering the runoff mechanism. The goal of achieving a natural consensus in <10 rounds has not yet been met. Future efforts may require rethinking the core preference or transition model logic, or exploring different interaction effects between parameters.

---

## 8. Dataset Creation Plan

**Summary:** You’ll (1) identify and catalogue authoritative cardinal rosters (GCatholic, Catholic-Hierarchy, Vatican press), (2) define a clear schema for each elector’s attributes, (3) write Python scrapers to fetch and parse HTML/CSV, (4) clean and normalize names/dates/locations, (5) validate record counts against official reports, (6) export raw and cleaned CSVs, and (7) document the full workflow in code and notebooks.

---

### 8.1. Identify Authoritative Sources

1. **GCatholic.org** – Maintains a “Living Cardinals” table with ages and birth dates (137 entries, including non-electors) ([GCatholic][1]).
2. **Catholic-Hierarchy.org** – Structured view of “Cardinal Electors” vs. non-voting cardinals ([Catholic Hierarchy][2]).
3. **Wikipedia – List of current cardinals** – Community-maintained roster (use cautiously, cross-check) ([Wikipedia][5]).
4. **Conclavoscope**

---

### 8.2. Define Data Schema

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

### 8.3. Acquisition & Extraction

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

## 9. Transition Model Specification

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

## 10. Parsing & Cleaning

* **Name normalization**: strip accents (e.g. `unidecode`), consistent casing.
* **Date parsing**: `datetime.strptime` → ISO 8601 strings.
* **Continent mapping**: use an ISO country-to-continent lookup (e.g. `pycountry` + custom map).
* **Elector filter**: calculate age on May 5, 2025 and set `elector_status = (age < 80)`.
* **Deduplication**: merge entries by exact `name` and `birth_date`.

---

## 11. Ideology Score Calculation

The `ideology_score` for each elector is a primary driver of voting preference in the simulation. It aims to represent their position on a conservative-to-progressive spectrum.

### 11.1 Primary Data Source

* **Conclavoscope Alignment Score (`cs_alignment_score`):** The main input for the `ideology_score` will be the `cs_alignment_score` obtained from the merged dataset (`data/merged_electors.csv`). This score reflects an assessment of the cardinal's alignment based on Conclavoscope's methodology.

### 11.2 Calculation

1. **Handle Missing Values:** Address missing `cs_alignment_score` values in the input data. Common strategies include:
    * Imputation: Replace missing scores with the mean or median `cs_alignment_score` of all electors who have a score.
    * Default Value: Assign a neutral score (e.g., 0 if the final scale is -1 to +1).
2. **Normalization:** Normalize the (potentially imputed) `cs_alignment_score` to a consistent range, typically **[-1.0, +1.0]**, where -1.0 represents the most conservative and +1.0 represents the most progressive end of the scale relative to the dataset.
    * Min-Max Scaling is a suitable method:
        `norm_score = -1 + 2 * (score - min_score) / (max_score - min_score)`
3. **Assignment:** The resulting normalized score is assigned to the `ideology_score` column in the `elector_data` DataFrame used by the simulation models.

(Previous complex methods involving LLMs, external website scraping, and multi-factor weighting (Sections 8.2-8.3 in earlier versions) are deprecated in favor of this more direct approach based on available Conclavoscope data.)

### 11.3 Validation & Calibration (Simplified)

#### 11.3.1 Sanity Checks & Outlier Review

* Assert –1 ≤ `ideology_score` ≤ +1 after calculation.
* Review the distribution of the final `ideology_score` (e.g., histogram) to ensure it appears reasonable.
* Examine electors with extreme scores (close to -1 or +1) to ensure they align with qualitative expectations, if possible.

#### 11.3.2 Calibration Considerations

* While direct external benchmarking might be difficult without the previously planned data sources, the `beta_weight` parameter in the `TransitionModel` can be calibrated during simulation runs (e.g., by comparing simulation outcomes against historical conclaves or expert predictions) to tune the influence of the `ideology_score`.
* If simulation results consistently deviate from expectations, revisiting the normalization method or the handling of missing values for the `ideology_score` may be necessary.

#### 11.3.3 Integration into Simulation Model

* The final `ideology_score` is a key input for the `TransitionModel` in `src/model.py`, influencing voting probabilities between cardinals based on ideological distance.

## 12. Simulation Engine (`src/simulate.py`)

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

```mermaid
{{ ... }}
