#!/usr/bin/env bash
set -euo pipefail

# Focused pipeline for the main figure:
# outputs/fig_enhanced_v2_chongwa_idia_direction_ci.png

python3 src/build_ie_chongwa_id_panel.py \
  --before-year 1980 \
  --require-date \
  --out outputs/ie_chongwa_id_pre1980_paragraph_panel.csv

python3 src/filter_paragraph_panel.py \
  --panel outputs/ie_chongwa_id_pre1980_paragraph_panel.csv \
  --qc-out outputs/ie_chongwa_id_pre1980_paragraph_panel_qc.csv \
  --filtered-out outputs/ie_chongwa_id_pre1980_paragraph_panel_filtered.csv

mkdir -p /tmp/mplconfig /tmp/.cache /tmp/fontconfig
MPLBACKEND=Agg \
MPLCONFIGDIR=/tmp/mplconfig \
XDG_CACHE_HOME=/tmp/.cache \
FONTCONFIG_PATH=/tmp/fontconfig \
.venv_spacy312/bin/python src/analyze_fulltext_enhanced_v2_chongwa_idia.py \
  --panel outputs/ie_chongwa_id_pre1980_paragraph_panel_filtered.csv \
  --before-year 1980

echo "Done."
echo "Primary figure: outputs/fig_enhanced_v2_chongwa_idia_direction_ci.png"
