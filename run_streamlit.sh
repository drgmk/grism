#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python -m streamlit run "${SCRIPT_DIR}/grism/ui_streamlit.py"
