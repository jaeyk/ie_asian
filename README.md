# ie_asian: Chong Wa vs IDIA (Pre-1980)

Primary workflow for the International Examiner analysis focused on:
- Actor comparison: `Chong Wa` vs `ID Improvement Association`
- Time cut: earliest available date to before `1980`
- Main output figure: `outputs/fig_enhanced_v2_chongwa_idia_direction_ci.png`

## One Command (Recommended)

```bash
./run_enhanced_v2_direction_ci.sh
```

## Primary Outputs

- `outputs/fig_enhanced_v2_chongwa_idia_direction_ci.png`
- `outputs/enhanced_v2_chongwa_idia_claims.csv`
- `outputs/enhanced_v2_chongwa_idia_unique_claims.csv`
- `outputs/enhanced_v2_chongwa_idia_heatmap_values.csv`

## Processing Workflow (Detailed)

### 1) Corpus + time window
- Input corpus is `raw_data/ie.csv` (International Examiner).
- Analysis window is constrained to rows with canonical dates before `1980`.
- In the current processed panel (`outputs/ie_chongwa_id_pre1980_paragraph_panel_filtered.csv`), years span `1976` to `1979`.

### 2) Panel construction and filtering
- `src/build_ie_chongwa_id_panel.py` builds paragraph-level rows and metadata.
- `src/filter_paragraph_panel.py` removes low-quality rows and writes:
  - `outputs/ie_chongwa_id_pre1980_paragraph_panel_qc.csv`
  - `outputs/ie_chongwa_id_pre1980_paragraph_panel_filtered.csv`

### 3) Core actor-position extraction (CI figure)
- `src/analyze_fulltext_enhanced_v2_chongwa_idia.py`:
  - sentence segmentation
  - actor mention detection (`Chong Wa` vs `ID Improvement Association`)
  - stance event extraction from dependency patterns + implicit anti-eviction rules
  - frame assignment into:
    - `Resident: Housing Affordability`
    - `Resident: Anti-Eviction/Displacement`
    - `State Resource Access / Service Delivery`
    - `Commercial Growth`
  - confidence scoring per claim
  - claim clustering into unique claim units
- Direction metric:
  - confidence-weighted support-minus-oppose index per actor x frame.
- Uncertainty:
  - bootstrap CI (95%) over unique claims per cell.

### 4) Allies-network extraction
- `src/analyze_chongwa_idia_allies_network.py` builds an organization-only network centered on:
  - `Chong Wa (CBA)`
  - `ID Improvement Association`

Pipeline details:
- NER extraction restricted to `ORG` entities for allies.
- Aggressive alias normalization:
  - `Chong Wa`/`Chong Wah`/`Chinese Benevolent Association` -> `Chong Wa (CBA)`
  - `International District Economic Association` + `IDEA` -> `International District Economic Association (IDEA)`
  - `HUD` + spelled-out variant -> `Department of Housing and Urban Development (HUD)`
  - `CSA` + spelled-out variant -> `Community Services Administration (CSA)`
- Non-organization artifact filtering:
  - removes role/fragment/noise entities (e.g., ordinance/task-force/role fragments, board-role strings, OCR-like artifacts).
- Relationship assignment (sentence-level):
  - `aligned`/`opposed` from high-precision cue rules (`cues`)
  - ambiguous cases use a lightweight in-corpus classifier (`model`)
  - guardrails force `neutral` for procedural/service contexts to avoid false polarity.
- Relationship assignment (edge-level historical aggregation):
  - sentence labels are aggregated by edge over all months.
  - final edge label is derived from historical aligned/opposed balance (`relation_score`) with month-level stability checks.
- Explicit project prior:
  - `Chong Wa (CBA)` ↔ `ID Improvement Association` is forced as `opposed` in final network outputs.

### Cues vs Model (Exact Definition)

`Cue` layer (rule-based):
- A sentence is `opposed` if it contains explicit conflict lexicon (e.g., `against`, `oppose`, `reject`, `criticize`, `conflict`, `dispute`).
- A sentence is `aligned` if it contains explicit cooperation/support lexicon (e.g., `partner`, `collaborate`, `cooperate`, `support`, `back`, `endorse`).
- If neither is present, sentence is provisionally `neutral`.

`Model` layer (only for ambiguous/no-cue sentences):
- Classifier type: multinomial Naive Bayes built from this same corpus slice (pre-1980 panel).
- Training labels: only high-confidence cue-labeled sentences (`aligned` or `opposed`) are used as pseudo-labeled training data.
- Features: lowercase token counts from regex tokenization (`[a-z][a-z-]{2,}`) with stopword removal.
- Smoothing: Laplace/add-one smoothing on token likelihoods.
- Priors: class priors from training document counts with add-one prior smoothing.
- Decision rule:
  - compute posterior probabilities for `aligned` vs `opposed`
  - if max posterior < confidence threshold (`0.62`), return `neutral`
  - otherwise return the argmax class.
- Guardrails before model inference:
  - service/program administration contexts are forced to `neutral`
  - meeting/discussion process language (e.g., `met with`, `discussed with`) is forced to `neutral`
  - stadium/traffic/planning procedural contexts are forced to `neutral` unless explicit conflict cues are present.

`Historical edge` layer:
- For each edge, sentence-level labels are aggregated over the full period and by month.
- `relation_score = (n_aligned - n_opposed) / (n_aligned + n_opposed)` (if denominator > 0, else 0).
- Final edge label:
  - `aligned` if score >= `0.25`
  - `opposed` if score <= `-0.25`
  - otherwise `neutral`
  - if monthly polarity flips and global magnitude is weak (`|score| < 0.6`), edge is set to `neutral`.

### 5) Visualization conventions for allies network
- Node color:
  - `Federal Agency` (dark blue)
  - `Local Government Agency` (teal)
  - `Community Organization` (light blue)
  - Hub colors: `Chong Wa (CBA)` orange, `ID Improvement Association` blue
- Edge style:
  - `aligned` = green solid
  - `opposed` = red dashed
  - `neutral` = gray dotted
- Node size:
  - proportional to number of sentences co-mentioning that organization with one of the hubs.
- Source annotation:
  - `Source: Seattle's International Examiner (1976-1979)`

## Allies Network (Chong Wa vs IDIA)

Builds an organization-only allies network with:
- Hubs: `Chong Wa (CBA)` and `ID Improvement Association`
- Edge types: `aligned`, `opposed`, `neutral`
  - `opposed` is explicitly enforced for `Chong Wa (CBA)` vs `ID Improvement Association`
  - Other edge labels are based on historical aggregation across the full period
- Organization type coloring:
  - `Federal Agency` (dark blue): includes `Department of Housing and Urban Development (HUD)` and `Community Services Administration (CSA)`
  - `Local Government Agency` (teal): includes `City Building Department`
  - `Community Organization` (light blue)
- Labeling and cleanup:
  - `Chong Wa`, `Chong Wah`, and Chinese Benevolent Association/Society aliases are normalized to `Chong Wa (CBA)`
  - `International District Economic Association` is normalized to `International District Economic Association (IDEA)`
  - Non-organization artifacts (e.g., ordinance/task-force/role fragments) are filtered
- Node size: sentence-level co-mention count with `Chong Wa (CBA)` or `ID Improvement Association`

Run:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mplconfig XDG_CACHE_HOME=/tmp/.cache FONTCONFIG_PATH=/tmp/fontconfig \
.venv_spacy312/bin/python src/analyze_chongwa_idia_allies_network.py \
  --panel outputs/ie_chongwa_id_pre1980_paragraph_panel_filtered.csv \
  --before-year 1980 \
  --top-allies-per-side 10 \
  --min-ally-mentions 2
```

Outputs:
- `outputs/fig_chongwa_idia_allies_network.png`
- `outputs/chongwa_idia_allies_nodes.csv`
- `outputs/chongwa_idia_allies_edges.csv`
- `outputs/chongwa_idia_allies_edge_evidence.csv`
- `outputs/chongwa_idia_idea_edge_evidence.csv`

## Data Inputs

- Source CSV: `raw_data/ie.csv`
- Generated panel: `outputs/ie_chongwa_id_pre1980_paragraph_panel_filtered.csv`

## Notes

- Current issue set used in v2 figures:
  - `Resident: Housing Affordability`
  - `Resident: Anti-Eviction/Displacement`
  - `State Resource Access / Service Delivery`
  - `Commercial Growth`
- `Environmental/Traffic` and `Representation/Identity` are excluded from the v2 figure set.
- For edge diagnostics, inspect `outputs/chongwa_idia_allies_edge_evidence.csv` to trace sentence-level relation assignments.
