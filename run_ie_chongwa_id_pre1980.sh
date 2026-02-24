#!/usr/bin/env bash
set -euo pipefail

python3 src/build_ie_chongwa_id_panel.py \
  --before-year 1980 \
  --source "International Examiner" \
  --require-date \
  --out outputs/ie_chongwa_id_pre1980_paragraph_panel.csv

python3 src/filter_paragraph_panel.py \
  --panel outputs/ie_chongwa_id_pre1980_paragraph_panel.csv \
  --qc-out outputs/ie_chongwa_id_pre1980_paragraph_panel_qc.csv \
  --filtered-out outputs/ie_chongwa_id_pre1980_paragraph_panel_filtered.csv

python3 src/analyze_paragraph_ethnic_panethnic_themes.py \
  --panel outputs/ie_chongwa_id_pre1980_paragraph_panel_filtered.csv \
  --out-prevalence outputs/ie_chongwa_id_pre1980_theme_prevalence_by_group.csv \
  --out-contrast outputs/ie_chongwa_id_pre1980_theme_contrast_chongwa_minus_id.csv \
  --out-monthly outputs/ie_chongwa_id_pre1980_theme_monthly_by_group.csv

/Users/jaeyeonkim/.virtualenvs/r-reticulate/bin/python src/analyze_dual_hub_policy_context.py \
  --panel outputs/ie_chongwa_id_pre1980_paragraph_panel_filtered.csv \
  --out-nodes outputs/ie_chongwa_id_pre1980_dual_hub_nodes.csv \
  --out-edges outputs/ie_chongwa_id_pre1980_dual_hub_edges.csv \
  --out-membership outputs/ie_chongwa_id_pre1980_dual_hub_community_membership.csv \
  --out-community-summary outputs/ie_chongwa_id_pre1980_dual_hub_community_summary.csv \
  --min-n 2 \
  --max-n 4 \
  --min-any-paragraphs 4 \
  --max-terms 90

MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mplconfig_ie XDG_CACHE_HOME=/tmp/.cache \
/Users/jaeyeonkim/.virtualenvs/r-reticulate/bin/python src/plot_dual_hub_policy_context.py \
  --nodes outputs/ie_chongwa_id_pre1980_dual_hub_nodes.csv \
  --edges outputs/ie_chongwa_id_pre1980_dual_hub_edges.csv \
  --out outputs/fig_ie_chongwa_id_pre1980_dual_hub_policy_context.png \
  --pan-label "Chong Wa / Benevolent" \
  --eth-label "International District" \
  --max-nodes 20 \
  --period-label "1976-01 to 1979-12" \
  --source-caption "International Examiner digitized archive (Seattle)"

echo "Chong Wa/Benevolent Association vs International District pre-1980 pipeline complete."
