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
