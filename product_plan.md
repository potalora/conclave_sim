## Windsurf-Based Conclave Simulation Product Plan

### Background

* **Papal Vacancy Triggered:** On May 4, 2025, Pope Francis passed away, initiating the need for a conclave of 135 cardinal electors to choose his successor.
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
| **Phase 1** | 2 weeks  | Elector ingestion; parameter module   |
| **Phase 2** | 2 weeks  | Diagnostics; basic visualizations     |
| **Phase 3** | 2 weeks  | Scenario runner; extended tests       |
| **Phase 4** | 1 week   | Full docs; example notebooks          |

### 5. Metrics & Success Criteria

* **Code Velocity:** Time to create/refactor core modules (<30 min per feature).
* **Accuracy:** Simulated 2013 conclave win probability within ±5 % of historical estimates.
* **Adoption:** ≥5 external contributors in first month.
* **Performance:** 20 000 simulations in <1 min on standard laptop CPU.

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
3. **Holy See Press Office (Vatican.va)** – Official bulletins listing confirmed cardinals present at consistories ([Vatican Press][3]).
4. **Acta Apostolicae Sedis** – Archive of official creation dates in successive volumes ([Vatican][4]).
5. **Wikipedia – List of current cardinals** – Community-maintained roster (use cautiously, cross-check) ([Wikipedia][5]).

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

---

## 8. Timeline & Ownership

* **Day 1–2:** Source cataloging and schema finalization.
* **Day 3–5:** Scraper scripts for GCatholic, Catholic-Hierarchy, Vatican press.
* **Day 6:** Cleaning pipeline and initial export.
* **Day 7:** QA, tests, documentation completion.
* **Owner:** \[Name], with review by tech lead on Day 7.

[1]: https://www.gcatholic.org/
[2]: https://www.catholic-hierarchy.org/
[3]: https://press.vatican.va/
[4]: https://www.vatican.va/archive/aas/index.htm
[5]: https://en.wikipedia.org/wiki/List_of_current_cardinals
[6]: https://en.wikipedia.org/wiki/Cardinals_created_by_Francis
