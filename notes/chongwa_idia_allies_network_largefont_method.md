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

## Plain-Language Report (Full Process, Start to Finish)

### Goal

The goal is to build a network figure showing which organizations were connected to Chong Wa and IDIA in International Examiner coverage (1976-1979), and whether those relationships looked mostly cooperative, conflictual, or neutral.

### Step 1: Start with the prepared IE text data

The script starts from a cleaned paragraph-level dataset:

- `outputs/ie_chongwa_id_pre1980_paragraph_panel_filtered.csv`

This file already contains article metadata (article id, date, year/month) and paragraph text.

### Step 2: Keep only the target period

The script keeps rows before 1980 (`canonical_year < 1980`), which is the 1976-1979 period for this project slice.

### Step 3: Break paragraphs into sentences

Each paragraph is split into sentences. Sentences are the core unit used for coding relationships and building edges.

### Step 4: Detect whether a sentence is about Chong Wa, IDIA, or both

Using name-pattern matching, the script identifies mentions of:

- Chong Wa (including common name variants/acronyms)
- IDIA (including common variants)

Then it assigns sentence "side":

- Chong Wa mentioned (without IDIA) -> Chong Wa side
- IDIA mentioned (without Chong Wa) -> IDIA side
- both or neither -> no side assignment for ally extraction

### Step 5: Extract organization names in each sentence

For side-assigned sentences, a named-entity step pulls out organization mentions (ORG entities).  
Those names are normalized (for example, standardizing HUD/CSA/IDEA names) and cleaned to remove obvious noise (street/place artifacts, generic fragments, unwanted tokens).

### Step 6: Label sentence relationship tone

Each sentence gets one of three labels:

- `aligned`
- `opposed`
- `neutral`

How:

1. Clear keyword rules first (conflict words vs cooperation words).
2. If no clear keyword, a small Naive Bayes text model (trained on the same corpus slice) makes a best guess.
3. Guardrails force `neutral` in procedural contexts (meeting/process/service/planning language) to avoid over-interpreting tone.

### Step 7: Build edge evidence from sentence co-mentions

From each eligible sentence, the script creates:

- Hub-to-ally edges (Chong Wa or IDIA connected to each ally found in that sentence)
- Ally-to-ally edges (pairs of allies co-mentioned in the same sentence)

It records audit-level evidence for each edge instance:

- file name
- article id
- paragraph id
- sentence text
- sentence relation label
- month (`YYYY-MM`)

### Step 8: Track direct Chong Wa-IDIA co-mentions

If a sentence contains both hubs, the script separately tallies Chong Wa-IDIA counts (`hub_hub`) by relation label.

### Step 9: Select which allies stay in the final graph

By default:

- keep top 12 allies per side
- require at least 2 mentions

Final node set = two hubs + selected allies.

### Step 10: Aggregate sentence evidence into final edge type

For each retained edge, sentence counts are combined over time:

- aligned count
- opposed count
- neutral count

A direction score is computed from aligned vs opposed balance.  
If the edge flips direction across months and is not strongly one-sided, it is downgraded to neutral.  
Final edge label becomes `aligned`, `opposed`, or `neutral`.

### Step 11: Apply project rule for the central hub-hub link

The Chong Wa-IDIA edge is explicitly inserted as `opposed` (project intent), using observed co-mention totals and a forced negative relation score.  
This guarantees the core antagonistic link appears in the final network.

### Step 12: Draw the figure

The script draws a network graph where:

- node size reflects how often that organization is co-mentioned with a hub
- node color indicates organization type
- edge color/style indicates relation type
- edge width increases with supporting sentence volume

### Step 13: Write outputs

Main outputs are:

- `outputs/chongwa_idia_allies_nodes_largefont.csv`
- `outputs/chongwa_idia_allies_edges_largefont.csv`
- `outputs/chongwa_idia_allies_edge_evidence_largefont.csv`
- `outputs/chongwa_idia_idea_edge_evidence_largefont.csv`
- `outputs/fig_chongwa_idia_allies_network_largefont.png`

### Repro

```bash
.venv_spacy312/bin/python src/analyze_chongwa_idia_allies_network_largefont.py
```
