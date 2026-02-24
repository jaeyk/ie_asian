# Method Note: `fig_chongwa_idia_allies_network_largefont.png`

This note documents how `src/analyze_chongwa_idia_allies_network_largefont.py` builds the figure `outputs/fig_chongwa_idia_allies_network_largefont.png`.

## 1) Input data and scope

- Input panel: `outputs/ie_chongwa_id_pre1980_paragraph_panel_filtered.csv`
- Time filter: keep rows with `canonical_year < 1980` (in practice, 1976-1979 in this corpus slice).
- Unit of analysis: sentence (paragraph text is split into sentences).

## 2) Hub detection (Chong Wa vs IDIA)

The script detects mentions with regex patterns:

- Chong Wa hub: variants like `Chong Wa(h)`, `Chinese Benevolent Association/Society`, `CBA`, `CWBA`.
- IDIA hub: variants like `Improvement Association`, `IDIA`, `Inter*im`, `Interim`, `IDHA`, `housing alliance`.

Sentence-side assignment:

- If sentence mentions Chong Wa but not IDIA -> Chong Wa side.
- If sentence mentions IDIA but not Chong Wa -> IDIA side.
- If both or neither -> not assigned to a side for ally extraction.

## 3) Ally extraction and cleaning

- spaCy NER (`en_core_web_sm`) extracts only `ORG` entities from each sentence.
- Entity names are normalized (e.g., `HUD`, `CSA`, `IDEA`, long-form names mapped to canonical labels).
- Multiple cleaning guards remove noisy entities:
  - generic/undesired names (blocklist),
  - likely person-like all-caps artifacts,
  - place/street context artifacts,
  - selected exclusions (e.g., `hotel`, `station` contexts).

Result: a cleaned set of candidate ally organizations per sentence.

## 4) Sentence-level relation labeling (`aligned` / `opposed` / `neutral`)

Each sentence gets a relation label by combining rule cues + a tiny in-corpus model:

1. High-precision cue rules:
   - conflict words (`against`, `oppose`, `criticize`, `reject`, `clash`, `dispute`, etc.) -> `opposed`
   - alliance words (`jointly`, `partner`, `collaborate`, `cooperate`, `support`, `endorse`, etc.) -> `aligned`
2. If no explicit cue, infer with a Naive Bayes model:
   - trained on this same corpus slice from cue-labeled aligned/opposed sentences,
   - token-level multinomial model with Laplace smoothing,
   - confidence threshold (`min_conf=0.62`), otherwise `neutral`.
3. Guardrails force `neutral` in procedural/service/planning contexts (meeting/process language, service-program language, stadium/traffic/planning context) to reduce false polarity.

## 5) Edge construction

For each eligible sentence:

- Hub-ally edges:
  - connect side hub (`Chong Wa` or `IDIA`) to each extracted ally.
- Ally-ally edges:
  - for every pair of allies co-mentioned in the same sentence.
- For every edge, store:
  - sentence-level relation count (`aligned/opposed/neutral`),
  - month-level counts (`YYYY-MM`) for temporal aggregation,
  - sentence audit evidence row (`file_name`, `article_id`, `paragraph_id`, `sentence_text`, label).

Separately, when a sentence mentions both hubs, the script tallies Chong Wa-IDIA co-mention relation counts in `hub_hub`.

## 6) Node selection

Defaults:

- `top-allies-per-side = 12`
- `min-ally-mentions = 2`

For each side, only allies meeting the mention threshold and within top-N mentions are kept.
Final network nodes = two hubs + selected allies.

## 7) Historical relation aggregation for final edge type

For each retained edge:

- Let `a = n_aligned`, `o = n_opposed`, `active = a + o`.
- If `active == 0` -> `neutral`.
- Score = `(a - o) / active`.
- If monthly polarity flips (some months positive, some negative) and `|score| < 0.6`, down-weight to `neutral`.
- Otherwise classify:
  - `score >= 0.25` -> `aligned`
  - `score <= -0.25` -> `opposed`
  - else -> `neutral`

## 8) Forced hub-hub edge rule

After normal edge assembly, Chong Wa <-> IDIA is explicitly inserted as `opposed` (project intent), with:

- `n_sentences = max(1, total hub-hub co-mention sentences)`
- `n_opposed = max(1, observed opposed count)`
- `relation_score = -1.0`

This guarantees the central antagonistic hub link appears in the final graph.

## 9) Visualization mapping

- Layout: `networkx.spring_layout` with slight anchor initialization for the two hubs.
- Node sizes:
  - fixed larger size for hubs,
  - otherwise scaled by sentence co-mention counts.
- Node colors:
  - hubs with distinct colors,
  - federal agencies (dark blue),
  - local government agencies/programs (green),
  - community organizations (light blue).
- Edge style:
  - `aligned`: solid green
  - `opposed`: dashed red
  - `neutral`: dotted gray
- Edge width: `1 + log1p(n_sentences)`.

## 10) Outputs

- `outputs/chongwa_idia_allies_nodes_largefont.csv`
- `outputs/chongwa_idia_allies_edges_largefont.csv`
- `outputs/chongwa_idia_allies_edge_evidence_largefont.csv`
- `outputs/chongwa_idia_idea_edge_evidence_largefont.csv` (IDIA-IDEA audit subset)
- `outputs/fig_chongwa_idia_allies_network_largefont.png`

## 11) Repro command

```bash
.venv_spacy312/bin/python src/analyze_chongwa_idia_allies_network_largefont.py
```

Optional key args:

```bash
--panel outputs/ie_chongwa_id_pre1980_paragraph_panel_filtered.csv
--before-year 1980
--top-allies-per-side 12
--min-ally-mentions 2
--spacy-model en_core_web_sm
```
